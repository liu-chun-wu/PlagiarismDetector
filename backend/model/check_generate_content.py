# ================================================================
#  rag_ai_detector.py
#
#  • RAG back-end: FAISS paragraph index (from Program 1)
#  • AI-vs-Human detector: multi-agent design (from Program 2)
# ================================================================
#  (0) Optional one-shot installs (uncomment for Colab)
# ------------------------------------------------
# !pip install -U openai python-dotenv \
#     torch transformers sentence-transformers \
#     langchain langchain-community faiss-cpu \
#     accelerate bitsandbytes jieba nltk tqdm

# ------------------------------------------------
#  (1) Imports & setup
# ------------------------------------------------
import os, re, gc, math, json, itertools, warnings, contextlib
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import torch, nltk, jieba
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv
import psutil
import time
import threading

# 正式部署用
NLTK_DIR = "nltk_data"

# 測試用路徑
# NLTK_DIR = "/home/undergrad/PlagiarismDetector/backend/nltk_data"

nltk.download('punkt', download_dir=NLTK_DIR, quiet=True)
warnings.filterwarnings("ignore", category=UserWarning)

# def ram_watcher(max_gb=40):
#     while True:
#         total_ram = psutil.virtual_memory().used / (1024**3)
#         print(f"[RAM Monitor] System: {total_ram:.2f} GB in use")  # Comment this line if too spammy!
#         if total_ram >= max_gb:
#             print(f"[RAM Monitor] System RAM usage above {max_gb:.2f}GB! Killing process NOW.")
#             os._exit(1)
#         time.sleep(1)

# threading.Thread(target=ram_watcher, args=(40,), daemon=True).start()

# ------------------------------------------------
#  (2) Runtime devices & model names
# ------------------------------------------------
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
print(f"Using device ▶ {DEVICE}")

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
PPL_LM_ID = "Qwen/Qwen3-4B"  # perplexity model
GPT4_STYLE_MODEL = "gpt-4.1"
GPT4_DECISION = "gpt-4.1"

# ------------------------------------------------
#  (3) OpenAI key (reads .env or OS env)
# ------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))

# ------------------------------------------------
#  (4) Corpus paths & vector-DB cache
# ------------------------------------------------
SOURCE_DIRS = [
    # 正式部署用
    "dataset/paraphrased_dataset/source/ccu",
    "dataset/paraphrased_dataset/source/nccu_2018",
    "dataset/paraphrased_dataset/source/nccu_2019",
    "dataset/paraphrased_dataset/source/ncu_2019",
    "dataset/paraphrased_dataset/source/ncu_2020",
    "dataset/paraphrased_dataset/source/nsyu_2019",
    "dataset/paraphrased_dataset/source/nycu",

    # # 測試用路徑
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ccu",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nccu_2018",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nccu_2019",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2019",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/ncu_2020",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nsyu_2019",
    # "/mnt/Agents4Financial/PlagiarismDetector/backend/dataset/paraphrased_dataset/source/nycu",
]

# auto-created on first run
VECTOR_DB_DIR = Path(
    # 正式部署用
    "kb_faiss_index")

# # 測試用路徑
# VECTOR_DB_DIR = Path(
#     "/home/undergrad/PlagiarismDetector/backend/kb_faiss_index")

_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                    model_kwargs={"device": DEVICE},
                                    encode_kwargs={"dtype": np.float16})


def split_into_paragraphs(txt: str) -> List[str]:
    return [blk.strip() for blk in txt.split("\n\n") if blk.strip()]


def build_generate_vector_db(dirs: List[str], save_dir: Path) -> FAISS:
    docs = []
    for d in dirs:
        for fn in Path(d).glob("*.txt"):
            with open(fn, encoding="utf-8", errors="ignore") as f:
                for para in split_into_paragraphs(f.read()):
                    docs.append(
                        Document(page_content=para,
                                 metadata={
                                     "file_path": str(fn),
                                     "source": "human_reference"
                                 }))
    print(f"✓ Loaded {len(docs):,} paragraphs.")
    db = FAISS.from_documents(docs, _embeddings)
    db.save_local(str(save_dir))
    return db


def load_or_build_db() -> FAISS:
    if VECTOR_DB_DIR.exists():
        print("✓ FAISS index found → loading …")
        return FAISS.load_local(str(VECTOR_DB_DIR),
                                _embeddings,
                                allow_dangerous_deserialization=True)
    print("⚙️  Building FAISS index …")
    return build_generate_vector_db(SOURCE_DIRS, VECTOR_DB_DIR)


vector_db = load_or_build_db()
cross_encoder = CrossEncoder(RERANKER_MODEL, device=DEVICE)


# ------------------------------------------------
#  (5) Helpers
# ------------------------------------------------
def segment_cn(text: str) -> List[str]:
    sents = re.split(r"(?<=[。！？!?；;])\s*", re.sub(r"\s+", " ", text))
    return [s for s in sents if s]


# =================================================================
#                 ──  A G E N T   C L A S S E S  ──
# =================================================================
class PreprocessAgent:

    def __call__(self, txt: str) -> List[str]:
        return segment_cn(re.sub(r"\u3000| ", " ", txt))


class StyleAgent:

    def __init__(self, model_name: str, temperature: float = 0.3):
        self.model, self.temp = model_name, temperature

    def __call__(self, sample: str) -> str:
        prompt = ("你是 AI 內容鑑別專家，列出 3 個文風特徵並判斷偏向 (Human)/(AI)。\n\n"
                  f"【待檢測文本】\n{sample}")
        rsp = client.chat.completions.create(model=self.model,
                                             temperature=self.temp,
                                             max_tokens=512,
                                             messages=[{
                                                 "role": "system",
                                                 "content": prompt
                                             }])
        return rsp.choices[0].message.content.strip()


class HeuristicsAgent:

    def __init__(self, tok, lm):
        self.tok, self.lm = tok, lm

    @contextlib.contextmanager
    def _nograd(self):
        with torch.no_grad():
            yield

    def perplexity(self, txt):
        ids = self.tok(txt,
                       return_tensors="pt",
                       truncation=True,
                       max_length=2048).input_ids.to(self.lm.device)
        with self._nograd():
            loss = self.lm(ids, labels=ids).loss
        return min(max(math.exp(loss.item()) / 100, 0), 1)

    def repetition(self, sents):
        words = list(itertools.chain.from_iterable(
            jieba.cut(s) for s in sents))
        top10 = Counter(words).most_common(10)
        return min(max(np.mean([c for _, c in top10]) / 10, 0), 1)

    def burstiness(self, sents):
        lens = [len(s.split()) for s in sents]
        return min(max((np.std(lens) + 1e-6) / (np.mean(lens) + 1e-6) / 2, 0),
                   1)

    def __call__(self, sents):
        txt = " ".join(sents[:300])
        return {
            "perplexity": self.perplexity(txt),
            "repetition": self.repetition(sents),
            "burstiness": self.burstiness(sents)
        }


class RAGAgent:

    def __init__(self, db: FAISS, k: int = 5):
        self.db, self.k = db, k

    def __call__(self, text):
        hits = self.db.similarity_search_with_score(text, k=self.k)
        if not hits:
            return {
                "style_match_score": 0.5,
                "ai_ratio": 0.5,
                "retrieved_texts": [],
                "matched_human_source": False
            }

        q = _embeddings.embed_query(text)
        matched_human = False
        retrieved, sims = [], []

        for d, _ in hits:
            d_emb = _embeddings.embed_query(d.page_content)
            sim = np.dot(
                q, d_emb) / (np.linalg.norm(q) * np.linalg.norm(d_emb) + 1e-8)

            # Flag as matched if high sim to human reference
            if getattr(d, 'metadata',
                       {}).get("source") == "human_reference" and sim > 0.98:
                matched_human = True

            # Keep non-human texts for scoring
            if getattr(d, 'metadata', {}).get("source") != "human_reference":
                retrieved.append(d.page_content)
                sims.append(sim)

        # Fallback: if no non-human content retrieved
        if not retrieved:
            retrieved = [d.page_content for d, _ in hits]
            sims = [0.0] * len(retrieved)

        style_score = (len(retrieved) / self.k +
                       sum(sims) / len(sims)) / 2 if sims else 0.5

        return {
            "style_match_score": style_score,
            "ai_ratio": 0.5,
            "retrieved_texts": retrieved,
            "matched_human_source": matched_human
        }


class DecisionAgent:

    def __init__(self, model_name: str, temperature: float = 0.3):
        self.model, self.temp = model_name, temperature

    def __call__(self, heur, style_obs, rag, sample):
        if rag.get("matched_human_source", False):
            return {
                "pred": "HUMAN",
                "plagiarism_confidence": 100,
                "reason": "與資料庫中的人類撰寫來源完全相符，判定為人類撰寫。",
                "force_human_match": True  # ← helps downstream logic
            }

        prompt = f"""
根據以下統計特徵與分析，判斷文本來源（AI/HUMAN）並給信心值 0-100：
【統計特徵】
Perplexity {heur['perplexity']:.2f} · Repetition {heur['repetition']:.2f} · Burstiness {heur['burstiness']:.2f}
【GPT-Style】\n{style_obs}
【RAG】 Style-match {rag['style_match_score']:.2f}
【文本樣例】\n{sample[:200]}…
輸出格式：
1. Verdict: AI 或 HUMAN
2. Confidence: 整數 0-100
3. Reason: 三行內中文"""
        rsp = client.chat.completions.create(
            model=self.model,
            temperature=self.temp,
            max_tokens=512,
            messages=[{
                "role": "system",
                "content": prompt
            }]).choices[0].message.content.splitlines()
        verdict = "AI" if "AI" in rsp[0].upper() else "HUMAN"
        conf = int(re.findall(r"\d+", rsp[1])[-1]) if re.findall(
            r"\d+", rsp[1]) else 50
        reason = " ".join(rsp[2:]) if len(rsp) >= 3 else "N/A"
        return {
            "pred": verdict,
            "plagiarism_confidence": conf,
            "reason": reason
        }


# ------------------------------------------------
#  (6) Highlight helper
# ------------------------------------------------
def highlight_ai_segments(text, rag, min_sim=0.7):
    if rag.get("matched_human_source"):
        # Skip all highlighting if known human source is matched
        sents = segment_cn(text)
        return " ".join(sents), {
            "total": len(sents),
            "ai_sents": 0,
            "ai_pct": 0.0,
            "sents": sents,
            "similarities": [0.0] * len(sents)
        }

    sents = segment_cn(text)
    q = _embeddings.embed_query(text)
    flags, sims, srcs = [False
                         ] * len(sents), [0.0] * len(sents), [""] * len(sents)

    for i, s in enumerate(sents):
        if len(s) < 10: continue
        s_emb = _embeddings.embed_query(s)
        for rt in rag["retrieved_texts"]:
            rt_emb = _embeddings.embed_query(rt)
            sim = np.dot(s_emb, rt_emb) / (
                np.linalg.norm(s_emb) * np.linalg.norm(rt_emb) + 1e-8)
            if sim > min_sim and sim > sims[i]:
                flags[i], sims[i], srcs[i] = True, sim, rt[:100] + "…"

    out = ["🤖" + s if f else s for f, s in zip(flags, sents)]

    return " ".join(out), {
        "total": len(sents),
        "ai_sents": sum(flags),
        "ai_pct": round(sum(flags) / len(sents) * 100, 2) if sents else 0,
        "sents": sents,
        "similarities": sims
    }


# ------------------------------------------------
#  (7) Heavy model init
# ------------------------------------------------
print("🔄 Loading Qwen-4B for perplexity …")
q_tok = AutoTokenizer.from_pretrained(PPL_LM_ID, trust_remote_code=True)
q_model = AutoModelForCausalLM.from_pretrained(
    PPL_LM_ID,
    device_map={
        "": 0
    },
    trust_remote_code=True,
    torch_dtype=torch.float16).eval()

# ------------------------------------------------
#  (8) Compose pipeline
# ------------------------------------------------
preA = PreprocessAgent()
styA = StyleAgent(GPT4_STYLE_MODEL)
heuA = HeuristicsAgent(q_tok, q_model)
ragA = RAGAgent(vector_db, k=5)
decA = DecisionAgent(GPT4_DECISION)


def detect_from_text(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {
            "pred": "HUMAN",
            "plagiarism_confidence": 100,
            "reason": "Empty text."
        }

    sents = preA(text)

    # -- (A) Run RAG retrieval on full text
    rag = ragA(" ".join(sents))

    # -- (B) Run heuristics on full text
    heur = heuA(sents)

    # -- (C) Run style judgment + final AI/Human decision on full text
    #       Use a middle chunk (500–1000 tokens) if text is very long
    sample_sents = sents  # Use all sentences

    sample = " ".join(sample_sents)
    style = styA(sample)
    dec = decA(heur, style, rag, sample)

    ai_prob = 0.0 if dec.get("force_human_match",
                             False) else (dec["plagiarism_confidence"] /
                                          100 if dec["pred"] == "AI" else 1 -
                                          dec["plagiarism_confidence"] / 100)

    # -- (D) After decision: highlight and extract AI-like region
    hl_text, hl_stats = highlight_ai_segments(text, rag)
    sim_scores = hl_stats["similarities"]
    sents_all = hl_stats["sents"]

    # Extract longest AI-like segment (sim > threshold)
    threshold = 0.80
    start, end, max_len = 0, 0, 0
    best_block = (0, 1)
    while start < len(sim_scores):
        if sim_scores[start] < threshold:
            start += 1
            continue
        end = start
        while end < len(sim_scores) and sim_scores[end] >= threshold:
            end += 1
        if end - start > max_len:
            max_len = end - start
            best_block = (start, end)
        start = end
    snippet_sents = sents_all[best_block[0]:best_block[1]]

    # -- (E) Final output
    dec.update({
        "plagiarism_percentage": round(ai_prob * 100, 2),
        **heur,
        "style_obs": style,
        "rag_style_match": rag["style_match_score"],
        "plagiarism_snippet": snippet_sents,  # for interpretability
        "timestamp": datetime.now().isoformat(),
        "highlighted_text": hl_text,
        "highlight_stats": hl_stats
    })

    return dec


# =================================================================
#  (9) CLI entry-point
# =================================================================
if __name__ == "__main__":
    # CHANGE HERE: put your test file path
    test_file_path = "/home/undergrad/PlagiarismDetector/backend/model/generate_test.txt"  # Or the absolute/relative path you want
    with open(test_file_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if txt:
        print("\n分析結果：")
        print(json.dumps(detect_from_text(txt), ensure_ascii=False, indent=2))
