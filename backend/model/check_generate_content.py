import os
import re
import glob
import math
import json
import itertools
import random
import warnings
import contextlib
import time
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Union
import pickle, gzip, gc             #  <── new
import faiss
import numpy as np
import pandas as pd
import torch
import nltk
import jieba
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai

nltk.download('punkt', quiet=True)
warnings.filterwarnings("ignore", category=UserWarning)

# SOURCE_JSON_DIRS = [
#     "/content/drive/MyDrive/extracted_keywords/source_ncu_2020"
# ]
# PARAPHRASED_JSON_DIRS = [
#     "/content/drive/MyDrive/extracted_keywords/MyDrive_ncu_2020",
#     "/content/drive/MyDrive/extracted_keywords/MyDrive_nsyu_2019"
# ]

# Folder prefix that the *builder notebook* saved
SAVE_BASE     = Path("/home/undergrad/PlagiarismDetector/backend/dataset/ai_database/hyperrag_kb_fast_identical")
DOCSTORE_PATH = Path("/home/undergrad/PlagiarismDetector/backend/dataset/ai_database/hyperrag_kb_fast_identical.docstore.jsonl")
openai.api_key = "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"
GPT4_STYLE_MODEL = "gpt-4.1"
GPT4_DECISION = "gpt-4.1"
PPL_LM_ID = "Qwen/Qwen3-4B"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_SIMILAR = 5

DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"
# put this right after DEVICE_MAP is defined
_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE_MAP},
    encode_kwargs={"dtype": np.float16}
)


qwen_tokenizer = AutoTokenizer.from_pretrained(PPL_LM_ID, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    PPL_LM_ID,
    device_map=DEVICE_MAP,
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval()

ENTITY_PROMPT = """
你是专业关键词抽取助手。
请从下述文本中全面分析，提取5-15个最核心、相关的关键词或技术短语（如名词、方法、术语等）。
仅输出如下JSON格式，不能输出任何其他内容：

{"keywords": ["关键词1", "关键词2", ..., "关键词N"]}

注意：
- 所有关键词必须与下方文本实际内容紧密相关。
- 严禁输出注释、格式说明、范例、引号、冒号、编号或任何额外文字。

正文如下：
{TEXT}
"""

RELATION_PROMPT = """
請閱讀下列正文，說明 **這些實體之間的共同關係**，
用一句中文概述，控制在 100 個字以內（勿換行）：
{ENT_LIST}

正文：
{TEXT}
【只輸出關係概述，不要額外文字】
"""

# -----------------------------------------------------------------
# Minimal JSON-line doc-store that IS AddableMixin-compatible
# -----------------------------------------------------------------
import json, io, os
from typing import Dict, Optional
from langchain_community.docstore.base import AddableMixin, Docstore
from langchain.schema import Document

class JSONDocstore(AddableMixin, Docstore):
    """
    • Append-only JSONL file (one doc per line).
    • Keeps NO full Document objects in RAM.
    • Implements only the bits FAISS needs: add / __contains__ / search.
      (The last two are dummies — FAISS never calls them at build time.)
    """
    def __init__(self, filename: str):
        # ensure parent dir exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename
        # reopen for appending on every add → avoids keeping a handle open
        # build a tiny in-memory index only if you later need random search
        self._seen_keys = set()

    # ---------- AddableMixin ------------------------------------
    def add(self, texts: Dict[str, Document]):
        """Persist {key: Document} mapping and return list(keys)."""
        with open(self.filename, "a", encoding="utf-8") as f:
            for k, doc in texts.items():
                out = {
                    "id":          k,
                    "page_content": doc.page_content,
                    "metadata":     doc.metadata,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                self._seen_keys.add(k)
        return list(texts.keys())

    # ---------- Docstore API stubs (not needed for building) ----
    def __contains__(self, key: str) -> bool:
        return key in self._seen_keys          # cheap membership check

    def search(self, search: str) -> Optional[Document]:
        # optional: implement if you want doc-store look-ups later
        return None

all_snippets: List[str] = []          # every chunk string stored exactly once
# ─────────────── save / load helpers ───────────────
def save_hyperrag(hg, faiss_db, base_path: str):
    """
    Persists:  • <base>.index   – FAISS vectors
              • <base>.pkl.gz  – graph, postings, snippets
    """
    from pathlib import Path
    base = Path(base_path)

    # 1) FAISS
    faiss_db.save_local(str(base.with_suffix(".index")))

    # 2) graph + postings
    payload = {
        "all_snippets":  all_snippets,
        "entity_texts":  dict(hg.entity_texts),
        "hyperedges":    hg.hyperedges,
        "edges":         list(hg.graph.edges())
    }
    with gzip.open(base.with_suffix(".pkl.gz"), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ KB saved → {base}.index / .pkl.gz")

# ================================================================
# load_hyperrag – loads the disk-based FAISS + JSONL doc-store
# ================================================================
def load_hyperrag(base_path: Path):
    """
    Return (hg, faiss_db) if files exist, else (None, None).

    • Loads the JSONL doc-store (very light-weight, append-only)
    • Memory-maps the FAISS index using faiss.read_index with IO_FLAG_MMAP
    • Rebuilds the hyper-graph object from the .pkl.gz
    """
    idx = base_path.with_suffix(".index")
    pkl = base_path.with_suffix(".pkl.gz")
    if not (idx.exists() and pkl.exists() and DOCSTORE_PATH.exists()):
        return None, None

    # 1) doc-store
    docstore = JSONDocstore(str(DOCSTORE_PATH))

    # 2) FAISS index (memory-mapped)
    index_path = str(idx / "index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at {index_path}")
    
    # Load the FAISS index with memory mapping
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)

    # Load the index-to-docstore-id mapping
    index_to_docstore_id_path = str(idx / "index.pkl")
    if not os.path.exists(index_to_docstore_id_path):
        raise FileNotFoundError(f"index_to_docstore_id file not found at {index_to_docstore_id_path}")
    
    with open(index_to_docstore_id_path, "rb") as f:
        index_to_docstore_id = pickle.load(f)

    # Create the LangChain FAISS vectorstore instance
    faiss_db = FAISS(
        embedding_function=_embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    # 3) hyper-graph payload
    with gzip.open(pkl, "rb") as f:
        payload = pickle.load(f)

    global all_snippets
    all_snippets = payload["all_snippets"]

    hg = HyperGraphDB(entity_extractor)
    hg.entity_texts = defaultdict(list, payload["entity_texts"])
    hg.hyperedges = payload["hyperedges"]
    hg.graph.add_edges_from(payload["edges"])

    print("✓ KB loaded from disk with memory mapping")
    return hg, faiss_db

def read_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore").strip()

def segment_cn(text: str) -> List[str]:
    sents = re.split(r"(?<=[。！？!?；;])\s*", re.sub(r"\s+", " ", text))
    return [s for s in sents if s]

@contextlib.contextmanager
def no_grad_ctx():
    with torch.no_grad():
        yield

class EntityExtractor:
    def __init__(self, api_model="gpt-4.1", max_tokens=256):
        self.api_model = api_model
        self.max_tokens = max_tokens

    def extract_entities(self, text: str):
        prompt = ENTITY_PROMPT.replace("{TEXT}", text)
        try:
            response = openai.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            answer = response.choices[0].message.content
            parsed = json.loads(answer)
            return [k for k in parsed.get("keywords", []) if 1 < len(k) < 15]
        except Exception as e:
            print(f"[GPT-4.1 extraction error] {e}\nRaw LLM output:\n{answer if 'answer' in locals() else ''}")
            return []

    def __call__(self, text: str) -> list:
        return self.extract_entities(text)

    def batch(self, texts: list) -> list:
        return [self.__call__(txt) for txt in texts]

entity_extractor = EntityExtractor(api_model="gpt-4.1", max_tokens=256)

# ───────────────────────── AFTER  ─────────────────────────
class HyperGraphDB:
    """
    • entity_texts[e] stores List[int] of snippet IDs, not the text itself
    • hyperedges[key]["texts"] stores List[int] IDs as well
    """
    def __init__(self, entity_extractor=None):
        self.entity_extractor = entity_extractor
        self.graph        = nx.Graph()
        self.hyperedges   = {}
        self.entity_texts = defaultdict(list)      # entity → [sid, sid, …]

    def _add_hyperedge(self, combo: tuple, sid: int, summary: str):
        key = frozenset(combo)
        item = self.hyperedges.setdefault(key, {"summary": summary, "texts": []})
        item["texts"].append(sid)

    def add(self, text: str, entities: list):
        if not entities:
            return

        #  assign a unique snippet-id
        sid = len(all_snippets)
        all_snippets.append(text)

        # classic nodes + pair edges
        for e in entities:
            self.graph.add_node(e)
            self.entity_texts[e].append(sid)       # store ID only
        for a, b in itertools.combinations(set(entities), 2):
            self.graph.add_edge(a, b)

        # hyper-edges (unchanged, but store sid)
        for k in range(3, min(len(entities), 6) + 1):
            for combo in itertools.combinations(sorted(set(entities)), k):
                key = frozenset(combo)
                if key in self.hyperedges:                # duplicate edge
                    self.hyperedges[key]["texts"].append(sid)
                    continue

                ent_list = ", ".join(combo)
                prompt   = RELATION_PROMPT.format(ENT_LIST=ent_list, TEXT=text[:800])
                try:
                    rsp = openai.chat.completions.create(
                        model="gpt-4.1",
                        messages=[{"role": "system", "content": prompt}],
                        temperature=0.0, max_tokens=64
                    )
                    summary = rsp.choices[0].message.content.strip()
                except Exception as e:
                    summary = f"摘要失敗:{str(e)[:20]}"

                self._add_hyperedge(combo, sid, summary)

    # helper to turn ID lists → text list
    def _ids_to_texts(self, id_list: List[int]) -> List[str]:
        return [all_snippets[s] for s in id_list]

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        ents = (self.entity_extractor(query) if self.entity_extractor
                else [w for w in re.split(r"[，,、;；\s]+", query) if w.strip()])
        sid_hits = []
        for e in ents:
            if not self.graph.has_node(e):
                continue
            sid_hits.extend(self.entity_texts.get(e, []))
            for n in self.graph.neighbors(e):
                sid_hits.extend(self.entity_texts.get(n, []))

        sid_hits = list(dict.fromkeys(sid_hits))[:top_k]
        return self._ids_to_texts(sid_hits)

# ─── load KB produced by the separate builder notebook ──────────
hg, faiss_db = load_hyperrag(SAVE_BASE)
if hg is None:
    raise RuntimeError(
        "Knowledge-base files not found.  "
        "Run the fast-builder notebook first to generate them."
    )

class KBManagerFromCache:
    def __init__(self, hg, faiss):
        self.hg = hg
        self.faiss = faiss

    def retrieve(self, query: str, k: int = TOP_K_SIMILAR):
        # 1) vector search across chunks + node + hyper summaries
        hits = self.faiss.similarity_search_with_score(query, k)

        # 2) if hit is a node, pull its supporting chunk texts
        expanded = []
        for d, _ in hits:
            expanded.append(d.page_content)
            if d.metadata.get("type") == "node":
                ent = d.metadata["entity"]
                sid_list = self.hg.entity_texts.get(ent, [])
                expanded.extend([all_snippets[sid] for sid in sid_list])

        expanded = list(dict.fromkeys(expanded))[:k]

        labels = [d.metadata.get("label", "HUMAN") for d, _ in hits]
        ai_ratio = labels.count("AI") / len(labels) if labels else 0.5
        return expanded, ai_ratio


kb_manager = KBManagerFromCache(hg, faiss_db)

class PreprocessAgent:
    def __init__(self, kb_manager):
        self.kb_manager = kb_manager

    def __call__(self, txt: str) -> List[str]:
        sents = segment_cn(re.sub(r"\u3000| ", " ", txt))
        return sents

class StyleAgent:
    def __init__(self, model_name: str, kb_manager, temperature=0.3):
        self.model = model_name
        self.temperature = temperature
        self.kb_manager = kb_manager

    def __call__(self, sample: str) -> str:
        examples, _ = self.kb_manager.retrieve(sample, k=2)
        example_text = "\n".join([f"- {ex[:100]}..." for ex in examples]) if examples else "無相關文本樣例"
        prompt = f"""
你是一位 AI 內容鑑別專家。請分析下列文字，列出三個文風特徵，並判斷偏向 Human 或 AI（以括號標註 (Human)/(AI)）。

【參考文本樣例】
{example_text}

【待檢測文本】
{sample}
"""
        try:
            rsp = openai.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=512,
                messages=[{"role": "system", "content": prompt}]
            )
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            print(f" StyleAgent GPT-4 call failed: {e}")
            return "StyleAgent Error: GPT-4 unavailable"

class HeuristicsAgent:
    def __init__(self, tokenizer, lm, kb_manager):
        self.tok = tokenizer
        self.lm = lm
        self.kb_manager = kb_manager

    def perplexity(self, txt: str) -> float:
        try:
            ids = self.tok(txt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(self.lm.device)
            with no_grad_ctx():
                loss = self.lm(ids, labels=ids).loss
            return min(max((math.exp(loss.item()) / 100), 0), 1)
        except:
            return 0.5

    def repetition(self, sents: List[str]) -> float:
        words = list(itertools.chain.from_iterable(jieba.cut(s) for s in sents))
        top10 = Counter(words).most_common(10)
        return min(max(np.mean([c for _, c in top10]) / 10, 0), 1)

    def burstiness(self, sents: List[str]) -> float:
        lens = [len(s.split()) for s in sents]
        return min(max((np.std(lens) + 1e-6) / (np.mean(lens) + 1e-6) / 2, 0), 1)

    def __call__(self, sents: List[str]) -> Dict[str, float]:
        text = " ".join(sents[:300])
        return {
            "perplexity": self.perplexity(text),
            "repetition": self.repetition(sents),
            "burstiness": self.burstiness(sents)
        }

class RAGAgent:
    def __init__(self, kb_manager, top_k=5):
        self.kb_manager = kb_manager
        self.top_k = top_k

    def __call__(self, text: str) -> Dict[str, Any]:
        retrieved_texts, ai_ratio = self.kb_manager.retrieve(text, k=self.top_k)

        if not retrieved_texts:
            return {
                "style_match_score": 0.5,
                "ai_ratio": 0.5,
                "retrieved_texts": []
            }

        retrieval_ratio = len(retrieved_texts) / self.top_k
        similarity_scores = []

        query_embedding = _embeddings.embed_query(text)

        for retrieved_text in retrieved_texts:
            doc_embedding = _embeddings.embed_query(retrieved_text)
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarity_scores.append(similarity)

        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.5
        style_score = (retrieval_ratio + avg_similarity) / 2

        return {
            "style_match_score": style_score,
            "ai_ratio": ai_ratio,
            "retrieved_texts": retrieved_texts
        }

class DecisionAgent:
    def __init__(self, model_name: str, temperature=0.3):
        self.model = model_name
        self.temperature = temperature

    def __call__(self, heur: Dict[str, float], style_obs: str,
                 rag_result: Dict[str, Any], text_sample: str) -> Dict[str, Union[str, int]]:
        prompt = f"""
根據以下統計特徵與分析，判斷文本來源：

【統計特徵】
- Perplexity: {heur['perplexity']:.2f}
- Repetition: {heur['repetition']:.2f}
- Burstiness: {heur['burstiness']:.2f}
- AI Neighbor Ratio: {rag_result['ai_ratio']:.2f}

【GPT-Style 分析】
{style_obs}

【RAG 檢索分析】
- Style match score: {rag_result['style_match_score']:.2f}

【待檢測文本】
{text_sample[:200]}...

請用以下格式輸出：
1. Verdict: AI / HUMAN
2. Confidence: 0-100（整數）
3. Reason: 中文說明三行以內
"""
        rsp = openai.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=1024,
            messages=[{
                "role": "system",
                "content": prompt
            }]
        ).choices[0].message.content.splitlines()

        verdict = "AI" if "AI" in rsp[0].upper() else "HUMAN"
        conf = int(re.findall(r"\d+", rsp[1])[-1]) if len(rsp) > 1 and re.findall(r"\d+", rsp[1]) else 50
        reason = " ".join(rsp[2:]) if len(rsp) >= 3 else "N/A"

        return {"pred": verdict, "confidence": conf, "reason": reason}


def highlight_ai_segments(text, rag_results, min_similarity=0.7):
    from difflib import SequenceMatcher

    sents = segment_cn(text)
    ai_sentences = [False] * len(sents)
    matched_sources = [""] * len(sents)
    similarity_scores = [0.0] * len(sents)

    retrieved_texts = rag_results.get("retrieved_texts", [])
    for i, sent in enumerate(sents):
        for rt in retrieved_texts:
            if len(sent) < 10:
                continue
            similarity = SequenceMatcher(None, sent, rt).ratio()
            if similarity > min_similarity and similarity > similarity_scores[i]:
                ai_sentences[i] = True
                matched_sources[i] = rt[:100] + "..." if len(rt) > 100 else rt
                similarity_scores[i] = similarity

    highlighted_parts = []
    for i, (is_ai, sent, score, source) in enumerate(zip(ai_sentences, sents, similarity_scores, matched_sources)):
        if is_ai:
            highlighted_parts.append(
                f'<span style="background-color: #ffcccc; padding: 2px;" '
                f'title="AI Similarity: {score:.2f}\\nMatched: {source}">🤖 {sent}</span>'
            )
        else:
            highlighted_parts.append(sent)

    highlighted_text = " ".join(highlighted_parts)
    stats = {
        "total_sentences": len(sents),
        "ai_sentences": sum(ai_sentences),
        "human_sentences": len(sents) - sum(ai_sentences),
        "ai_percentage": round((sum(ai_sentences) / len(sents)) * 100, 2) if sents else 0,
        "average_similarity": round(sum(similarity_scores) / len(similarity_scores), 3) if similarity_scores else 0
    }

    return highlighted_text, stats


def detect_ai(file_path: str) -> Dict:
    raw_txt = read_txt(file_path)
    if not raw_txt.strip():
        return {"pred": "HUMAN", "confidence": 0, "reason": "Empty file."}

    sents = preA(raw_txt)
    mid_sample = " ".join(sents[len(sents) // 3:len(sents) // 3 + 2]) if len(sents) > 2 else raw_txt[:300]

    try:
        style_obs = styA(mid_sample)
    except Exception as e:
        style_obs = f"GPT 呼叫失敗，錯誤：{str(e)}"

    heuristics = heuA(sents)

    try:
        rag_result = ragA(mid_sample)
    except Exception as e:
        rag_result = {
            "style_match_score": 0.5,
            "ai_ratio": 0.5,
            "retrieved_texts": []
        }

    decision = decA(heuristics, style_obs, rag_result, mid_sample)
    highlighted_text, highlight_stats = highlight_ai_segments(raw_txt, rag_result)

    decision.update({
        "file": Path(file_path).name,
        "perplexity": heuristics.get("perplexity", 0),
        "repetition": heuristics.get("repetition", 0),
        "burstiness": heuristics.get("burstiness", 0),
        "style_obs": style_obs,
        "rag_style_match": rag_result.get("style_match_score", 0),
        "rag_ai_ratio": rag_result.get("ai_ratio", 0),
        "highlighted_text": highlighted_text,
        "highlight_stats": highlight_stats
    })
    return decision

from datetime import datetime

def detect_from_text(text: str) -> Dict[str, Any]:
    """
    Modified detect_from_text function that includes AI highlighting
    """
    if not text:
        return {
            "pred": "HUMAN",
            "confidence": 0,
            "reason": "Empty text.",
            "ai_prob": 0.0,
            "snippet": ""
        }

    sents = preA(text)
    mid = " ".join(sents[len(sents) // 3:len(sents) // 3 + 2]) if len(sents) > 2 else text[:300]

    try:
        style = styA(mid)
    except Exception as e:
        style = f"StyleAgent Error: {str(e)}"

    heur = heuA(sents)

    try:
        rag = ragA(mid)
    except Exception as e:
        rag = {
            "style_match_score": 0.5,
            "ai_ratio": 0.5,
            "retrieved_texts": []
        }

    dec = decA(heur, style, rag, mid)
    ai_prob = dec["confidence"] / 100 if dec["pred"] == "AI" else 1 - dec["confidence"] / 100

    highlighted_text, highlight_stats = highlight_ai_segments(text, rag)

    dec.update({
        "ai_prob": ai_prob,
        "perplexity": heur.get("perplexity", 0),
        "repetition": heur.get("repetition", 0),
        "burstiness": heur.get("burstiness", 0),
        "style_obs": style,
        "rag_style_match": rag.get("style_match_score", 0),
        "rag_ai_ratio": rag.get("ai_ratio", 0),
        "snippet": mid,
        "timestamp": datetime.now().isoformat(),
        "highlighted_text": highlighted_text,
        "highlight_stats": highlight_stats
    })
    return dec


# ================================================================
# Entry Point: User Paste Text
# ================================================================
if __name__ == "__main__":
    preA = PreprocessAgent(kb_manager)
    styA = StyleAgent(GPT4_STYLE_MODEL, kb_manager)
    heuA = HeuristicsAgent(qwen_tokenizer, qwen_model, kb_manager)
    ragA = RAGAgent(kb_manager, TOP_K_SIMILAR)
    decA = DecisionAgent(GPT4_DECISION)

    print("\n請貼上要分析的文字（按兩次 Enter 結束輸入）:")
    buf = []
    while True:
        line = input()
        if not line.strip():
            break
        buf.append(line)
    user_text = "\n".join(buf).strip()

    if user_text:
        print("\n分析結果：")
        user_result = detect_from_text(user_text)
        print(json.dumps(user_result, ensure_ascii=False, indent=2))

        # # Optional: Save result to file
        # output_path = "/content/drive/MyDrive/user_test_result.json"
        # with open(output_path, "w", encoding="utf-8") as f:
        #     json.dump(user_result, f, ensure_ascii=False, indent=2)
        # print(f"\n已儲存至 Google Drive：{output_path}")

# import os
# import re
# import glob
# import math
# import json
# import itertools
# import random
# import warnings
# import contextlib
# import time
# import shutil
# from pathlib import Path
# from collections import Counter, defaultdict
# from typing import List, Dict, Tuple, Any, Union

# import numpy as np
# import pandas as pd
# import torch
# import nltk
# import jieba
# import networkx as nx
# from tqdm import tqdm
# from sklearn.metrics import classification_report, accuracy_score, f1_score
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.schema import Document
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import openai

# nltk.download('punkt', quiet=True)
# warnings.filterwarnings("ignore", category=UserWarning)

# SOURCE_JSON_DIRS = [
#     "/content/drive/MyDrive/extracted_keywords/source_ncu_2019",
#     "/content/drive/MyDrive/extracted_keywords/source_ncu_2020"
# ]
# PARAPHRASED_JSON_DIRS = [
#     "/content/drive/MyDrive/extracted_keywords/ncu_2020",
#     "/content/drive/MyDrive/extracted_keywords/nsyu_2019"
# ]

# GPT4_STYLE_MODEL = "gpt-4.1"
# GPT4_DECISION = "gpt-4.1"
# PPL_LM_ID = "Qwen/Qwen3-4B"
# EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# CHUNK_SIZE = 512
# CHUNK_OVERLAP = 50
# TOP_K_SIMILAR = 5

# DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"

# qwen_tokenizer = AutoTokenizer.from_pretrained(PPL_LM_ID, trust_remote_code=True)
# qwen_model = AutoModelForCausalLM.from_pretrained(
#     PPL_LM_ID,
#     device_map=DEVICE_MAP,
#     trust_remote_code=True,
#     torch_dtype=torch.float16
# ).eval()

# ENTITY_PROMPT = """
# 你是专业关键词抽取助手。
# 请从下述文本中全面分析，提取5-15个最核心、相关的关键词或技术短语（如名词、方法、术语等）。
# 仅输出如下JSON格式，不能输出任何其他内容：

# {"keywords": ["关键词1", "关键词2", ..., "关键词N"]}

# 注意：
# - 所有关键词必须与下方文本实际内容紧密相关。
# - 严禁输出注释、格式说明、范例、引号、冒号、编号或任何额外文字。

# 正文如下：
# {TEXT}
# """

# RELATION_PROMPT = """
# 請閱讀下列正文，說明 **這些實體之間的共同關係**，
# 用一句中文概述，控制在 100 個字以內（勿換行）：
# {ENT_LIST}

# 正文：
# {TEXT}
# 【只輸出關係概述，不要額外文字】
# """

# def read_txt(path: str) -> str:
#     return Path(path).read_text(encoding="utf-8", errors="ignore").strip()

# def segment_cn(text: str) -> List[str]:
#     sents = re.split(r"(?<=[。！？!?；;])\s*", re.sub(r"\s+", " ", text))
#     return [s for s in sents if s]

# @contextlib.contextmanager
# def no_grad_ctx():
#     with torch.no_grad():
#         yield

# class EntityExtractor:
#     def __init__(self, api_model="gpt-4.1", max_tokens=256):
#         self.api_model = api_model
#         self.max_tokens = max_tokens

#     def extract_entities(self, text: str):
#         prompt = ENTITY_PROMPT.replace("{TEXT}", text)
#         try:
#             response = openai.chat.completions.create(
#                 model=self.api_model,
#                 messages=[{"role": "system", "content": prompt}],
#                 temperature=0.0,
#                 max_tokens=self.max_tokens,
#                 response_format={"type": "json_object"}
#             )
#             answer = response.choices[0].message.content
#             parsed = json.loads(answer)
#             return [k for k in parsed.get("keywords", []) if 1 < len(k) < 15]
#         except Exception as e:
#             print(f"[GPT-4.1 extraction error] {e}\nRaw LLM output:\n{answer if 'answer' in locals() else ''}")
#             return []

#     def __call__(self, text: str) -> list:
#         return self.extract_entities(text)

#     def batch(self, texts: list) -> list:
#         return [self.__call__(txt) for txt in texts]

# entity_extractor = EntityExtractor(api_model="gpt-4.1", max_tokens=256)

# class HyperGraphDB:
#     """
#     • graph ........ pair-wise edges  (NetworkX)
#     • hyperedges ... dict[frozenset] ➜ {'summary': str, 'texts': [str]}
#     • entity_texts.. every chunk where an entity appears (for neighbour crawl)
#     """
#     def __init__(self, entity_extractor=None):
#         self.entity_extractor = entity_extractor
#         self.graph        = nx.Graph()
#         self.hyperedges   = {}
#         self.entity_texts = defaultdict(list)

#     # -------- internal helper ------------------------------------
#     def _add_hyperedge(self, combo: tuple, text: str, summary: str):
#         key = frozenset(combo)
#         item = self.hyperedges.setdefault(key, {"summary": summary, "texts": []})
#         item["texts"].append(text)

#     # -------- public API -----------------------------------------
#     def add(self, text: str, entities: list):
#         if not entities:
#             return

#         # Nodes + pair-wise edges
#         for e in entities:
#             self.graph.add_node(e)
#             self.entity_texts[e].append(text)
#         for a, b in itertools.combinations(set(entities), 2):
#             self.graph.add_edge(a, b)

#         # Hyper-edges (≥3) with GPT-4 summary
#         for k in range(3, min(len(entities), 6) + 1):
#             for combo in itertools.combinations(sorted(set(entities)), k):
#                 key = frozenset(combo)
#                 if key in self.hyperedges:              # already summarised
#                     self.hyperedges[key]["texts"].append(text)
#                     continue
#                 ent_list = ", ".join(combo)
#                 prompt   = RELATION_PROMPT.format(ENT_LIST=ent_list, TEXT=text[:800])
#                 try:
#                     rsp = openai.chat.completions.create(
#                         model="gpt-4.1",
#                         messages=[{"role": "system", "content": prompt}],
#                         temperature=0.0,
#                         max_tokens=64
#                     )
#                     summary = rsp.choices[0].message.content.strip()
#                 except Exception as e:
#                     summary = f"摘要失敗:{str(e)[:20]}"
#                 self._add_hyperedge(combo, text, summary)

#     def retrieve(self, query: str, top_k: int = 5):
#         ents = (self.entity_extractor(query) if self.entity_extractor
#                 else [w for w in re.split(r"[，,、;；\s]+", query) if w.strip()])
#         hits = []
#         for e in ents:
#             if not self.graph.has_node(e):
#                 continue
#             hits.extend(self.entity_texts.get(e, []))
#             for n in self.graph.neighbors(e):
#                 hits.extend(self.entity_texts.get(n, []))
#         return list(dict.fromkeys(hits))[:top_k]
    
# # class HyperGraphDB:
# #     def __init__(self, entity_extractor=None):
# #         self.entity_extractor = entity_extractor
# #         self.graph = nx.Graph()
# #         self.entity_texts = defaultdict(list)

# #     def add(self, text: str, entities: list):
# #         if not entities:
# #             return
# #         for e in entities:
# #             self.graph.add_node(e)
# #             self.entity_texts[e].append(text)
# #         for a, b in itertools.combinations(set(entities), 2):
# #             self.graph.add_edge(a, b)

# #     def retrieve(self, query: str, top_k: int = 5):
# #         if self.entity_extractor:
# #             ents = self.entity_extractor(query)
# #         else:
# #             ents = [w.strip() for w in re.split(r"[，,、;；\s]+", query) if w.strip()]
# #         hits = []
# #         for e in ents:
# #             hits.extend(self.entity_texts.get(e, []))
# #             for n in self.graph.neighbors(e):
# #                 hits.extend(self.entity_texts.get(n, []))
# #         return list(dict.fromkeys(hits))[:top_k]

# source_jsons = []
# for d in SOURCE_JSON_DIRS:
#     source_jsons.extend(glob.glob(f"{d}/**/*.json", recursive=True))
# paraphrased_jsons = []
# for d in PARAPHRASED_JSON_DIRS:
#     paraphrased_jsons.extend(glob.glob(f"{d}/**/*.json", recursive=True))

# print("Found", len(source_jsons), "source JSONs and", len(paraphrased_jsons), "paraphrased JSONs")


# hg   = HyperGraphDB(entity_extractor)
# docs = []
# processed_hyperedges = set()            # avoid duplicate edge docs

# def ingest(fp_list, label_tag):
#     for fp in tqdm(fp_list, desc=f"Loading {label_tag} JSONs"):
#         with open(fp, encoding="utf-8") as f:
#             data = json.load(f)

#         for chunk in data["chunks"]:
#             text      = chunk["text"]
#             keywords  = chunk.get("keywords", [])
#             hg.add(text, keywords)                       # builds graph + edges

#             # 1) raw chunk
#             docs.append(Document(text, metadata={"label":label_tag,
#                                                  "type":"chunk",
#                                                  "path":fp}))

#             # 2) entity nodes
#             for ent in keywords:
#                 docs.append(Document(f"實體: {ent}",
#                                      metadata={"label":"GRAPH",
#                                                "type":"node",
#                                                "entity":ent}))

#             # 3) hyper-edge summaries (only once per unique edge)
#             for k in range(3, min(len(keywords),6)+1):
#                 for combo in itertools.combinations(sorted(set(keywords)), k):
#                     key = frozenset(combo)
#                     if key in hg.hyperedges and key not in processed_hyperedges:
#                         docs.append(Document(hg.hyperedges[key]["summary"],
#                                              metadata={"label":"GRAPH",
#                                                        "type":"hyper",
#                                                        "entities":list(combo)}))
#                         processed_hyperedges.add(key)

# ingest(source_jsons,      "HUMAN")
# ingest(paraphrased_jsons, "AI")

# # hg = HyperGraphDB()
# # docs = []

# # for fp in tqdm(source_jsons, desc="Loading source JSONs"):
# #     with open(fp, encoding='utf-8') as f:
# #         data = json.load(f)
# #     for chunk in data['chunks']:
# #         text = chunk['text']
# #         keywords = chunk.get('keywords', [])
# #         hg.add(text, keywords)
# #         docs.append(Document(page_content=text, metadata={"label": "HUMAN"}))

# # for fp in tqdm(paraphrased_jsons, desc="Loading paraphrased JSONs"):
# #     with open(fp, encoding='utf-8') as f:
# #         data = json.load(f)
# #     for chunk in data['chunks']:
# #         text = chunk['text']
# #         keywords = chunk.get('keywords', [])
# #         hg.add(text, keywords)
# #         docs.append(Document(page_content=text, metadata={"label": "AI"}))

# print(f"Vectorising {len(docs)} KB chunks … (this may take a while)")
# _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": DEVICE_MAP})
# faiss_db = FAISS.from_documents(docs, embedding=_embeddings)

# class KBManagerFromCache:
#     def __init__(self, hg, faiss):
#         self.hg = hg
#         self.faiss = faiss

#     def retrieve(self, query: str, k: int = TOP_K_SIMILAR):
#         # 1) vector search across chunks + node + hyper summaries
#         hits = self.faiss.similarity_search_with_score(query, k)

#         # 2) if hit is a node, pull its supporting chunk texts
#         expanded = []
#         for d, _ in hits:
#             expanded.append(d.page_content)
#             if d.metadata.get("type") == "node":
#                 ent = d.metadata["entity"]
#                 expanded.extend(self.hg.entity_texts.get(ent, []))

#         expanded = list(dict.fromkeys(expanded))[:k]

#         labels = [d.metadata.get("label", "HUMAN") for d, _ in hits]
#         ai_ratio = labels.count("AI") / len(labels) if labels else 0.5
#         return expanded, ai_ratio
#     # def retrieve(self, query: str, k: int = TOP_K_SIMILAR):
#     #     ents = entity_extractor(query)
#     #     hg_texts = []
#     #     for e in ents:
#     #         hg_texts.extend(self.hg.entity_texts.get(e, []))
#     #         for n in self.hg.graph.neighbors(e):
#     #             hg_texts.extend(self.hg.entity_texts.get(n, []))
#     #     hg_texts = list(dict.fromkeys(hg_texts))[:k]
#     #     docs = self.faiss.similarity_search_with_score(query, k)
#     #     all_docs = []
#     #     if hg_texts:
#     #         for text in hg_texts:
#     #             for doc in self.faiss.index_to_docstore_id.values():
#     #                 doc_obj = self.faiss.docstore.search(doc)
#     #                 if doc_obj.page_content == text:
#     #                     all_docs.append((doc_obj, 0))
#     #                     break
#     #     if docs: all_docs.extend(docs)
#     #     if not all_docs: return [], 0.5
#     #     labels = [doc[0].metadata.get("label", "HUMAN") for doc in all_docs]
#     #     ai_ratio = labels.count("AI") / len(labels)
#     #     return [doc[0].page_content for doc in all_docs], ai_ratio

# kb_manager = KBManagerFromCache(hg, faiss_db)

# class PreprocessAgent:
#     def __init__(self, kb_manager):
#         self.kb_manager = kb_manager

#     def __call__(self, txt: str) -> List[str]:
#         sents = segment_cn(re.sub(r"\u3000| ", " ", txt))
#         return sents

# class StyleAgent:
#     def __init__(self, model_name: str, kb_manager, temperature=0.3):
#         self.model = model_name
#         self.temperature = temperature
#         self.kb_manager = kb_manager

#     def __call__(self, sample: str) -> str:
#         examples, _ = self.kb_manager.retrieve(sample, k=2)
#         example_text = "\n".join([f"- {ex[:100]}..." for ex in examples]) if examples else "無相關文本樣例"
#         prompt = f"""
# 你是一位 AI 內容鑑別專家。請分析下列文字，列出三個文風特徵，並判斷偏向 Human 或 AI（以括號標註 (Human)/(AI)）。

# 【參考文本樣例】
# {example_text}

# 【待檢測文本】
# {sample}
# """
#         try:
#             rsp = openai.chat.completions.create(
#                 model=self.model,
#                 temperature=self.temperature,
#                 max_tokens=512,
#                 messages=[{"role": "system", "content": prompt}]
#             )
#             return rsp.choices[0].message.content.strip()
#         except Exception as e:
#             print(f" StyleAgent GPT-4 call failed: {e}")
#             return "StyleAgent Error: GPT-4 unavailable"

# class HeuristicsAgent:
#     def __init__(self, tokenizer, lm, kb_manager):
#         self.tok = tokenizer
#         self.lm = lm
#         self.kb_manager = kb_manager

#     def perplexity(self, txt: str) -> float:
#         try:
#             ids = self.tok(txt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(self.lm.device)
#             with no_grad_ctx():
#                 loss = self.lm(ids, labels=ids).loss
#             return min(max((math.exp(loss.item()) / 100), 0), 1)
#         except:
#             return 0.5

#     def repetition(self, sents: List[str]) -> float:
#         words = list(itertools.chain.from_iterable(jieba.cut(s) for s in sents))
#         top10 = Counter(words).most_common(10)
#         return min(max(np.mean([c for _, c in top10]) / 10, 0), 1)

#     def burstiness(self, sents: List[str]) -> float:
#         lens = [len(s.split()) for s in sents]
#         return min(max((np.std(lens) + 1e-6) / (np.mean(lens) + 1e-6) / 2, 0), 1)

#     def __call__(self, sents: List[str]) -> Dict[str, float]:
#         text = " ".join(sents[:300])
#         return {
#             "perplexity": self.perplexity(text),
#             "repetition": self.repetition(sents),
#             "burstiness": self.burstiness(sents)
#         }

# class RAGAgent:
#     def __init__(self, kb_manager, top_k=5):
#         self.kb_manager = kb_manager
#         self.top_k = top_k

#     def __call__(self, text: str) -> Dict[str, Any]:
#         retrieved_texts, ai_ratio = self.kb_manager.retrieve(text, k=self.top_k)

#         if not retrieved_texts:
#             return {
#                 "style_match_score": 0.5,
#                 "ai_ratio": 0.5,
#                 "retrieved_texts": []
#             }

#         retrieval_ratio = len(retrieved_texts) / self.top_k
#         similarity_scores = []

#         query_embedding = _embeddings.embed_query(text)

#         for retrieved_text in retrieved_texts:
#             doc_embedding = _embeddings.embed_query(retrieved_text)
#             similarity = np.dot(query_embedding, doc_embedding) / (
#                 np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
#             )
#             similarity_scores.append(similarity)

#         avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.5
#         style_score = (retrieval_ratio + avg_similarity) / 2

#         return {
#             "style_match_score": style_score,
#             "ai_ratio": ai_ratio,
#             "retrieved_texts": retrieved_texts
#         }

# class DecisionAgent:
#     def __init__(self, model_name: str, temperature=0.3):
#         self.model = model_name
#         self.temperature = temperature

#     def __call__(self, heur: Dict[str, float], style_obs: str,
#                  rag_result: Dict[str, Any], text_sample: str) -> Dict[str, Union[str, int]]:
#         prompt = f"""
# 根據以下統計特徵與分析，判斷文本來源：

# 【統計特徵】
# - Perplexity: {heur['perplexity']:.2f}
# - Repetition: {heur['repetition']:.2f}
# - Burstiness: {heur['burstiness']:.2f}
# - AI Neighbor Ratio: {rag_result['ai_ratio']:.2f}

# 【GPT-Style 分析】
# {style_obs}

# 【RAG 檢索分析】
# - Style match score: {rag_result['style_match_score']:.2f}

# 【待檢測文本】
# {text_sample[:200]}...

# 請用以下格式輸出：
# 1. Verdict: AI / HUMAN
# 2. Confidence: 0-100（整數）
# 3. Reason: 中文說明三行以內
# """
#         rsp = openai.chat.completions.create(
#             model=self.model,
#             temperature=self.temperature,
#             max_tokens=1024,
#             messages=[{
#                 "role": "system",
#                 "content": prompt
#             }]
#         ).choices[0].message.content.splitlines()

#         verdict = "AI" if "AI" in rsp[0].upper() else "HUMAN"
#         conf = int(re.findall(r"\d+", rsp[1])[-1]) if len(rsp) > 1 and re.findall(r"\d+", rsp[1]) else 50
#         reason = " ".join(rsp[2:]) if len(rsp) >= 3 else "N/A"

#         return {"pred": verdict, "confidence": conf, "reason": reason}


# def highlight_ai_segments(text, rag_results, min_similarity=0.7):
#     from difflib import SequenceMatcher

#     sents = segment_cn(text)
#     ai_sentences = [False] * len(sents)
#     matched_sources = [""] * len(sents)
#     similarity_scores = [0.0] * len(sents)

#     retrieved_texts = rag_results.get("retrieved_texts", [])
#     for i, sent in enumerate(sents):
#         for rt in retrieved_texts:
#             if len(sent) < 10:
#                 continue
#             similarity = SequenceMatcher(None, sent, rt).ratio()
#             if similarity > min_similarity and similarity > similarity_scores[i]:
#                 ai_sentences[i] = True
#                 matched_sources[i] = rt[:100] + "..." if len(rt) > 100 else rt
#                 similarity_scores[i] = similarity

#     highlighted_parts = []
#     for i, (is_ai, sent, score, source) in enumerate(zip(ai_sentences, sents, similarity_scores, matched_sources)):
#         if is_ai:
#             highlighted_parts.append(
#                 f'<span style="background-color: #ffcccc; padding: 2px;" '
#                 f'title="AI Similarity: {score:.2f}\\nMatched: {source}">🤖 {sent}</span>'
#             )
#         else:
#             highlighted_parts.append(sent)

#     highlighted_text = " ".join(highlighted_parts)
#     stats = {
#         "total_sentences": len(sents),
#         "ai_sentences": sum(ai_sentences),
#         "human_sentences": len(sents) - sum(ai_sentences),
#         "ai_percentage": round((sum(ai_sentences) / len(sents)) * 100, 2) if sents else 0,
#         "average_similarity": round(sum(similarity_scores) / len(similarity_scores), 3) if similarity_scores else 0
#     }

#     return highlighted_text, stats


# def detect_ai(file_path: str) -> Dict:
#     raw_txt = read_txt(file_path)
#     if not raw_txt.strip():
#         return {"pred": "HUMAN", "confidence": 0, "reason": "Empty file."}

#     sents = preA(raw_txt)
#     mid_sample = " ".join(sents[len(sents) // 3:len(sents) // 3 + 2]) if len(sents) > 2 else raw_txt[:300]

#     try:
#         style_obs = styA(mid_sample)
#     except Exception as e:
#         style_obs = f"GPT 呼叫失敗，錯誤：{str(e)}"

#     heuristics = heuA(sents)

#     try:
#         rag_result = ragA(mid_sample)
#     except Exception as e:
#         rag_result = {
#             "style_match_score": 0.5,
#             "ai_ratio": 0.5,
#             "retrieved_texts": []
#         }

#     decision = decA(heuristics, style_obs, rag_result, mid_sample)
#     highlighted_text, highlight_stats = highlight_ai_segments(raw_txt, rag_result)

#     decision.update({
#         "file": Path(file_path).name,
#         "perplexity": heuristics.get("perplexity", 0),
#         "repetition": heuristics.get("repetition", 0),
#         "burstiness": heuristics.get("burstiness", 0),
#         "style_obs": style_obs,
#         "rag_style_match": rag_result.get("style_match_score", 0),
#         "rag_ai_ratio": rag_result.get("ai_ratio", 0),
#         "highlighted_text": highlighted_text,
#         "highlight_stats": highlight_stats
#     })
#     return decision

# from datetime import datetime

# def detect_from_text(text: str) -> Dict[str, Any]:
#     """
#     Modified detect_from_text function that includes AI highlighting
#     """
#     if not text:
#         return {
#             "pred": "HUMAN",
#             "confidence": 0,
#             "reason": "Empty text.",
#             "ai_prob": 0.0,
#             "snippet": ""
#         }

#     sents = preA(text)
#     mid = " ".join(sents[len(sents) // 3:len(sents) // 3 + 2]) if len(sents) > 2 else text[:300]

#     try:
#         style = styA(mid)
#     except Exception as e:
#         style = f"StyleAgent Error: {str(e)}"

#     heur = heuA(sents)

#     try:
#         rag = ragA(mid)
#     except Exception as e:
#         rag = {
#             "style_match_score": 0.5,
#             "ai_ratio": 0.5,
#             "retrieved_texts": []
#         }

#     dec = decA(heur, style, rag, mid)
#     ai_prob = dec["confidence"] / 100 if dec["pred"] == "AI" else 1 - dec["confidence"] / 100

#     highlighted_text, highlight_stats = highlight_ai_segments(text, rag)

#     dec.update({
#         "ai_prob": ai_prob,
#         "perplexity": heur.get("perplexity", 0),
#         "repetition": heur.get("repetition", 0),
#         "burstiness": heur.get("burstiness", 0),
#         "style_obs": style,
#         "rag_style_match": rag.get("style_match_score", 0),
#         "rag_ai_ratio": rag.get("ai_ratio", 0),
#         "snippet": mid,
#         "timestamp": datetime.now().isoformat(),
#         "highlighted_text": highlighted_text,
#         "highlight_stats": highlight_stats
#     })
#     return dec


# # ================================================================
# # Entry Point: User Paste Text
# # ================================================================
# if __name__ == "__main__":
#     preA = PreprocessAgent(kb_manager)
#     styA = StyleAgent(GPT4_STYLE_MODEL, kb_manager)
#     heuA = HeuristicsAgent(qwen_tokenizer, qwen_model, kb_manager)
#     ragA = RAGAgent(kb_manager, TOP_K_SIMILAR)
#     decA = DecisionAgent(GPT4_DECISION)

#     print("\n請貼上要分析的文字（按兩次 Enter 結束輸入）:")
#     buf = []
#     while True:
#         line = input()
#         if not line.strip():
#             break
#         buf.append(line)
#     user_text = "\n".join(buf).strip()

#     if user_text:
#         print("\n分析結果：")
#         user_result = detect_from_text(user_text)
#         print(json.dumps(user_result, ensure_ascii=False, indent=2))

#         # Optional: Save result to file
#         output_path = "/content/drive/MyDrive/user_test_result.json"
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(user_result, f, ensure_ascii=False, indent=2)
#         print(f"\n已儲存至 Google Drive：{output_path}")

# # ================================================================
# # (0) Install / Upgrade Packages
# # ================================================================
# # !pip install -U openai tiktoken transformers accelerate sentencepiece sentence-transformers nltk jieba pandas tqdm scikit-learn bitsandbytes chromadb langchain networkx langchain-huggingface langchain-community

# # ================================================================
# # (1) Imports
# # ================================================================
# # import sys
# # import sys
# # sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# import os, re, glob, math, json, itertools, random, warnings, contextlib
# from pathlib import Path
# from collections import Counter, defaultdict
# from typing import List, Dict, Tuple, Any, Union

# import numpy as np
# import pandas as pd
# import torch, nltk, jieba, networkx as nx
# from tqdm.auto import tqdm
# from sklearn.metrics import classification_report, accuracy_score, f1_score

# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings  # <-- 改這裡
# from langchain.vectorstores import FAISS
# from langchain.schema import Document
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import openai
# import shutil
# import time

# nltk.download('punkt', quiet=True)
# warnings.filterwarnings("ignore", category=UserWarning)

# # ================================================================
# #  (2) Basic Settings
# # ================================================================
# SOURCE_DIRS = [
#     "/content/drive/MyDrive/paraphrased_dataset/source/ncu_2019",
#     "/content/drive/MyDrive/paraphrased_dataset/source/ncu_2020"
# ]
# PARAPHRASED_DIRS = [
#     "/content/drive/MyDrive/ncu_2020", "/content/drive/MyDrive/nsyu_2019"
# ]

# # openai.api_key = "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"
# GPT4_STYLE_MODEL = "gpt-4.1"
# GPT4_DECISION = "gpt-4.1"
# PPL_LM_ID = "Qwen/Qwen3-4B"
# EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# CHUNK_SIZE = 512
# CHUNK_OVERLAP = 50
# TOP_K_SIMILAR = 5

# RAG_KNOWLEDGE_DIR = Path("/content/rag_knowledge")
# RAG_KNOWLEDGE_DIR.mkdir(exist_ok=True, parents=True)
# OUTPUT_DIR = Path("/content/ai_detection_outputs")
# OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
# DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"

# # Updated on 4/30/2025
# # QWEN_MODEL_NAME = "Qwen/Qwen3-4B"
# qwen_tokenizer = AutoTokenizer.from_pretrained(PPL_LM_ID,
#                                                trust_remote_code=True)
# qwen_model = AutoModelForCausalLM.from_pretrained(
#     PPL_LM_ID,
#     device_map="cuda",
#     trust_remote_code=True,
#     torch_dtype=torch.float16).eval()


# ENTITY_PROMPT = """
# 你是专业关键词抽取助手。
# 请从下述文本中全面分析，提取5-15个最核心、相关的关键词或技术短语（如名词、方法、术语等）。
# 仅输出如下JSON格式，不能输出任何其他内容：

# {"keywords": ["关键词1", "关键词2", ..., "关键词N"]}

# 注意：
# - 所有关键词必须与下方文本实际内容紧密相关。
# - 严禁输出注释、格式说明、范例、引号、冒号、编号或任何额外文字。

# 正文如下：
# {TEXT}
# """

# import re, json

# def robust_json_extract(text):
#     # Find the first valid [ ... ] block in the text for robust extraction
#     match = re.search(r'\[(?:.|\n)*?\]', text)
#     if match:
#         try:
#             return robust_json_extract(match.group(0))
#         except Exception as e:
#             print(f"JSON parse error: {e}\nRaw answer:\n{text}")
#             return []
#     print(f"No JSON array in answer! Raw:\n{text}")
#     return []
# # Updated on 4/30/2025
# # ================================================================
# # (3) Utility Functions
# # ================================================================
# def read_txt(path: str) -> str:
#     return Path(path).read_text(encoding="utf-8", errors="ignore").strip()


# def segment_cn(text: str) -> List[str]:
#     sents = re.split(r"(?<=[。！？!?；;])\s*", re.sub(r"\s+", " ", text))
#     return [s for s in sents if s]


# @contextlib.contextmanager
# def no_grad_ctx():
#     with torch.no_grad():
#         yield


# def extract_json_keywords(llm_output):
#     # Find the first JSON block with a "keywords" field
#     match = re.search(r'\{[^{}]*"keywords"\s*:\s*\[[^\[\]]*\][^{}]*\}', llm_output, re.DOTALL)
#     if not match:
#         print("No valid JSON block found:\n", llm_output)
#         return []
#     try:
#         data = json.loads(match.group(0))
#         kw = data.get("keywords", [])
#         # Only keep plausible keywords
#         return [k for k in kw if 1 < len(k) < 15]
#     except Exception as e:
#         print(f"Failed to parse JSON: {e}\nllm_output: {llm_output}")
#         return []

# # ================================================================
# # (4) Entity Extractor
# # ================================================================
# # print("\u23F3 Loading spaCy model …")
# # _spacy_nlp = spacy.load("zh_core_web_sm")

# class EntityExtractor:
#     def __init__(self, api_model="gpt-4.1", max_tokens=256):
#         self.api_model = api_model
#         self.max_tokens = max_tokens

#     def extract_entities(self, text: str):
#         prompt = ENTITY_PROMPT.replace("{TEXT}", text)
#         try:
#             response = openai.chat.completions.create(
#                 model=self.api_model,
#                 messages=[{"role": "system", "content": prompt}],
#                 temperature=0.0,
#                 max_tokens=self.max_tokens,
#                 response_format={"type": "json_object"}
#             )
#             answer = response.choices[0].message.content
#             # Try to parse output as JSON
#             parsed = json.loads(answer)
#             return [k for k in parsed.get("keywords", []) if 1 < len(k) < 15]
#         except Exception as e:
#             print(f"[GPT-4.1 extraction error] {e}\nRaw LLM output:\n{answer if 'answer' in locals() else ''}")
#             return []

#     def __call__(self, text: str) -> list:
#         return self.extract_entities(text)

#     def batch(self, texts: list) -> list:
#         # No real batch with API, just map one by one (rate limit may apply!)
#         return [self.__call__(txt) for txt in texts]

# entity_extractor = EntityExtractor(api_model="gpt-4.1", max_tokens=256)

# # Updated on 4/30/2025

# # Updated on 4/30/2025
# # # ================================================================
# # # (5) HyperGraph Database
# # # ================================================================
# # class HyperGraphDB:

# #     def __init__(self):
# #         self.graph = nx.Graph()
# #         self.entity_texts = defaultdict(list)

# #     def add(self, text: str, entities: List[str]):
# #         if not entities: return
# #         for e in entities:
# #             self.graph.add_node(e)
# #             self.entity_texts[e].append(text)
# #         for a, b in itertools.combinations(set(entities), 2):
# #             self.graph.add_edge(a, b)

# #     def retrieve(self, query: str, top_k: int = 5) -> List[str]:
# #         ents = EntityExtractor()(query)
# #         hits = []
# #         for e in ents:
# #             hits.extend(self.entity_texts.get(e, []))
# #             for n in self.graph.neighbors(e):
# #                 hits.extend(self.entity_texts.get(n, []))
# #         return list(dict.fromkeys(hits))[:top_k]


# # # ================================================================
# # # (5) HyperGraph Database
# # # ================================================================
# # class HyperGraphDB:

# #     def __init__(self, entity_extractor):
# #         self.entity_extractor = entity_extractor  # Store the shared extractor
# #         self.graph = nx.Graph()
# #         self.entity_texts = defaultdict(list)

# #     def add(self, text: str, entities: List[str]):
# #         if not entities:
# #             return
# #         for e in entities:
# #             self.graph.add_node(e)
# #             self.entity_texts[e].append(text)
# #         for a, b in itertools.combinations(set(entities), 2):
# #             self.graph.add_edge(a, b)

# #     def retrieve(self, query: str, top_k: int = 5) -> List[str]:
# #         ents = self.entity_extractor(
# #             query)  # Use the shared extractor instance here
# #         hits = []
# #         for e in ents:
# #             hits.extend(self.entity_texts.get(e, []))
# #             for n in self.graph.neighbors(e):
# #                 hits.extend(self.entity_texts.get(n, []))
# #         # Remove duplicates, return up to top_k
# #         return list(dict.fromkeys(hits))[:top_k]


# # Updated on 4/30/2025

# # ================================================================
# # (6) Knowledge Base Manager (New Version)
# # ================================================================
# print("\u23F3 Building embedding model …")
# _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
#                                     model_kwargs={"device": DEVICE_MAP})

# # Updated on 4/30/2025
# # class KBManager:

# #     def __init__(self):
# #         self.text_splitter = RecursiveCharacterTextSplitter(
# #             chunk_size=CHUNK_SIZE,
# #             chunk_overlap=CHUNK_OVERLAP,
# #             separators=["\n\n", "\n", "。", "，", " "])
# #         self.hg = HyperGraphDB()
# #         self.faiss = None
# #         self._build_kb()

# #     def _build_kb(self):
# #         docs = []
# #         for d in SOURCE_DIRS:
# #             for fp in glob.glob(f"{d}/**/*.txt", recursive=True):
# #                 content = read_txt(fp)
# #                 for chunk in self.text_splitter.split_text(content):
# #                     ents = EntityExtractor()(chunk)
# #                     self.hg.add(chunk, ents)
# #                     docs.append(
# #                         Document(page_content=chunk,
# #                                  metadata={"label": "HUMAN"}))
# #         for d in PARAPHRASED_DIRS:
# #             for fp in glob.glob(f"{d}/**/*.txt", recursive=True):
# #                 content = read_txt(fp)
# #                 for chunk in self.text_splitter.split_text(content):
# #                     ents = EntityExtractor()(chunk)
# #                     self.hg.add(chunk, ents)
# #                     docs.append(
# #                         Document(page_content=chunk, metadata={"label": "AI"}))
# #         print(
# #             f"\ud83d\udcda  Vectorising {len(docs)} KB chunks … (this may take a while)"
# #         )
# #         self.faiss = FAISS.from_documents(docs, embedding=_embeddings)

# #     def retrieve(self,
# #                  query: str,
# #                  k: int = TOP_K_SIMILAR) -> Tuple[List[str], float]:
# #         # Get results from HyperGraphDB
# #         hg_texts = self.hg.retrieve(query, k)

# #         # Get results from FAISS vector search
# #         docs = self.faiss.similarity_search_with_score(query, k)

# #         # Combine results
# #         all_docs = []

# #         # Add HyperGraphDB results with their labels
# #         if hg_texts:
# #             for text in hg_texts:
# #                 # Find this text in your original docs to get its label
# #                 for doc in self.faiss.index_to_docstore_id.values():
# #                     doc_obj = self.faiss.docstore.search(doc)
# #                     if doc_obj.page_content == text:
# #                         all_docs.append(
# #                             (doc_obj,
# #                              0))  # Use similarity score of 0 (perfect match)
# #                         break

# #         # Add FAISS results
# #         if docs:
# #             all_docs.extend(docs)

# #         # If no documents were found at all
# #         if not all_docs:
# #             return [], 0.5

# #         # Calculate AI ratio based on all retrieved documents
# #         labels = [doc[0].metadata.get("label", "HUMAN") for doc in all_docs]
# #         ai_ratio = labels.count("AI") / len(labels)

# #         return [doc[0].page_content for doc in all_docs], ai_ratio

# # kb_manager = KBManager()

# # class KBManager:

# #     def __init__(self, entity_extractor):
# #         self.text_splitter = RecursiveCharacterTextSplitter(
# #             chunk_size=CHUNK_SIZE,
# #             chunk_overlap=CHUNK_OVERLAP,
# #             separators=["\n\n", "\n", "。", "，", " "])
# #         self.hg = HyperGraphDB(entity_extractor)  # Use shared instance here!
# #         self.faiss = None
# #         self.entity_extractor = entity_extractor  # Store for use in _build_kb
# #         self._build_kb()

# #     def _build_kb(self):
# #         docs = []
# #         output_root = Path("/content/drive/MyDrive/extracted_keywords")
# #         print("Indexing SOURCE_DIRS ...")
# #         for d in SOURCE_DIRS:
# #             file_paths = glob.glob(f"{d}/**/*.txt", recursive=True)
# #             for fp in tqdm(file_paths, desc=f"[SOURCE] {Path(d).name}"):
# #                 content = read_txt(fp)
# #                 chunks = self.text_splitter.split_text(content)
# #                 ents_list = self.entity_extractor.batch(chunks)
# #                 for chunk, ents in zip(chunks, ents_list):
# #                     self.hg.add(chunk, ents)
# #                     docs.append(
# #                         Document(page_content=chunk,
# #                                  metadata={"label": "HUMAN"}))
# #                 # Save extracted keywords to Drive
# #                 relative_path = Path(fp).relative_to(d)  # e.g., "paper1.txt"
# #                 school_year = Path(d).name  # e.g., "ncu_2019"
# #                 save_dir = output_root / school_year / relative_path.parent
# #                 save_dir.mkdir(parents=True, exist_ok=True)
# #                 save_path = save_dir / f"{relative_path.stem}_keywords.json"

# #             with open(save_path, "w", encoding="utf-8") as f:
# #                 json.dump({
# #                     "file": str(relative_path),
# #                     "keywords": ents_list
# #                 },
# #                           f,
# #                           ensure_ascii=False,
# #                           indent=2)

# #         print("Indexing PARAPHRASED_DIRS ...")
# #         for d in PARAPHRASED_DIRS:
# #             file_paths = glob.glob(f"{d}/**/*.txt", recursive=True)
# #             for fp in tqdm(file_paths, desc=f"[PARA] {Path(d).name}"):
# #                 content = read_txt(fp)
# #                 chunks = self.text_splitter.split_text(content)
# #                 ents_list = self.entity_extractor.batch(chunks)
# #                 for chunk, ents in zip(chunks, ents_list):
# #                     self.hg.add(chunk, ents)
# #                     docs.append(
# #                         Document(page_content=chunk, metadata={"label": "AI"}))
# #                 # Save extracted keywords to Drive
# #                 relative_path = Path(fp).relative_to(d)  # e.g., "paper1.txt"
# #                 school_year = Path(d).name  # e.g., "ncu_2019"
# #                 save_dir = output_root / school_year / relative_path.parent
# #                 save_dir.mkdir(parents=True, exist_ok=True)
# #                 save_path = save_dir / f"{relative_path.stem}_keywords.json"
# #         print(f"Vectorising {len(docs)} KB chunks … (this may take a while)")
# #         self.faiss = FAISS.from_documents(docs, embedding=_embeddings)

# #     def retrieve(self,
# #                  query: str,
# #                  k: int = TOP_K_SIMILAR) -> Tuple[List[str], float]:
# #         # Get results from HyperGraphDB
# #         hg_texts = self.hg.retrieve(query, k)

# #         # Get results from FAISS vector search
# #         docs = self.faiss.similarity_search_with_score(query, k)

# #         # Combine results
# #         all_docs = []

# #         # Add HyperGraphDB results with their labels
# #         if hg_texts:
# #             for text in hg_texts:
# #                 # Find this text in your original docs to get its label
# #                 for doc in self.faiss.index_to_docstore_id.values():
# #                     doc_obj = self.faiss.docstore.search(doc)
# #                     if doc_obj.page_content == text:
# #                         all_docs.append(
# #                             (doc_obj,
# #                              0))  # Use similarity score of 0 (perfect match)
# #                         break

# #         # Add FAISS results
# #         if docs:
# #             all_docs.extend(docs)

# #         # If no documents were found at all
# #         if not all_docs:
# #             return [], 0.5

# #         # Calculate AI ratio based on all retrieved documents
# #         labels = [doc[0].metadata.get("label", "HUMAN") for doc in all_docs]
# #         ai_ratio = labels.count("AI") / len(labels)

# #         return [doc[0].page_content for doc in all_docs], ai_ratio
# #  kb_manager = KBManager(entity_extractor)

# # ================================================================
# # (2) Basic Settings
# # ================================================================
# SOURCE_JSON_DIRS = [
#     "/content/drive/MyDrive/extracted_keywords/source_ncu_2019",
#     "/content/drive/MyDrive/extracted_keywords/source_ncu_2020"
# ]
# PARAPHRASED_JSON_DIRS = [
#     "/content/drive/MyDrive/extracted_keywords/ncu_2020",
#     "/content/drive/MyDrive/extracted_keywords/nsyu_2019"
# ]

# # Gather paths
# source_jsons = []
# for d in SOURCE_JSON_DIRS:
#     source_jsons.extend(glob.glob(f"{d}/**/*.json", recursive=True))
# paraphrased_jsons = []
# for d in PARAPHRASED_JSON_DIRS:
#     paraphrased_jsons.extend(glob.glob(f"{d}/**/*.json", recursive=True))

# print("Found", len(source_jsons), "source JSONs and", len(paraphrased_jsons),
#       "paraphrased JSONs")


# # Build the hypergraph and docs list from JSONs
# class HyperGraphDB:
#     def __init__(self, entity_extractor=None):
#         self.entity_extractor = entity_extractor  # Can be None in JSON mode
#         self.graph = nx.Graph()
#         self.entity_texts = defaultdict(list)

#     def add(self, text: str, entities: list):
#         if not entities:
#             return
#         for e in entities:
#             self.graph.add_node(e)
#             self.entity_texts[e].append(text)
#         for a, b in itertools.combinations(set(entities), 2):
#             self.graph.add_edge(a, b)

#     def retrieve(self, query: str, top_k: int = 5):
#         """Entity-based retrieval. Use the shared extractor if not None—for query-time."""
#         # If using a prebuilt entity_extractor, use it on the query, otherwise, you could use jieba or a fallback.
#         if self.entity_extractor:
#             ents = self.entity_extractor(query)
#         else:
#             ents = [w.strip() for w in re.split(r"[，,、;；\s]+", query) if w.strip()]
#         hits = []
#         for e in ents:
#             hits.extend(self.entity_texts.get(e, []))
#             for n in self.graph.neighbors(e):
#                 hits.extend(self.entity_texts.get(n, []))
#         return list(dict.fromkeys(hits))[:top_k]


# hg = HyperGraphDB()
# docs = []

# for fp in tqdm(source_jsons, desc="Loading source JSONs"):
#     with open(fp, encoding='utf-8') as f:
#         data = json.load(f)
#     for chunk in data['chunks']:
#         text = chunk['text']
#         keywords = chunk.get('keywords', [])
#         hg.add(text, keywords)
#         docs.append(Document(page_content=text, metadata={"label": "HUMAN"}))

# for fp in tqdm(paraphrased_jsons, desc="Loading paraphrased JSONs"):
#     with open(fp, encoding='utf-8') as f:
#         data = json.load(f)
#     for chunk in data['chunks']:
#         text = chunk['text']
#         keywords = chunk.get('keywords', [])
#         hg.add(text, keywords)
#         docs.append(Document(page_content=text, metadata={"label": "AI"}))

# print(f"Vectorising {len(docs)} KB chunks … (this may take a while)")
# _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
#                                     model_kwargs={"device": DEVICE_MAP})
# faiss_db = FAISS.from_documents(docs, embedding=_embeddings)


# class KBManagerFromCache:

#     def __init__(self, hg, faiss):
#         self.hg = hg
#         self.faiss = faiss

#     def retrieve(self, query: str, k: int = TOP_K_SIMILAR):
#         ents = entity_extractor(query)
#         hg_texts = []
#         for e in ents:
#             hg_texts.extend(self.hg.entity_texts.get(e, []))
#             for n in self.hg.graph.neighbors(e):
#                 hg_texts.extend(self.hg.entity_texts.get(n, []))
#         hg_texts = list(dict.fromkeys(hg_texts))[:k]
#         docs = self.faiss.similarity_search_with_score(query, k)
#         all_docs = []
#         if hg_texts:
#             for text in hg_texts:
#                 for doc in self.faiss.index_to_docstore_id.values():
#                     doc_obj = self.faiss.docstore.search(doc)
#                     if doc_obj.page_content == text:
#                         all_docs.append((doc_obj, 0))
#                         break
#         if docs: all_docs.extend(docs)
#         if not all_docs: return [], 0.5
#         labels = [doc[0].metadata.get("label", "HUMAN") for doc in all_docs]
#         ai_ratio = labels.count("AI") / len(labels)
#         return [doc[0].page_content for doc in all_docs], ai_ratio


# # Use your same entity_extractor from above (unchanged)
# kb_manager = KBManagerFromCache(hg, faiss_db)

# # Updated on 4/30/2025


# # (7) Agents (Updated to use new kb_manager)
# class PreprocessAgent:

#     def __init__(self, kb_manager):
#         self.kb_manager = kb_manager

#     def __call__(self, txt: str) -> List[str]:
#         sents = segment_cn(re.sub(r"\u3000| ", " ", txt))
#         return sents


# class StyleAgent:

#     def __init__(self, model_name: str, kb_manager, temperature=0.3):
#         self.model = model_name
#         self.temperature = temperature
#         self.kb_manager = kb_manager

#     def __call__(self, sample: str) -> str:
#         examples, _ = self.kb_manager.retrieve(sample, k=2)
#         example_text = "\n".join([f"- {ex[:100]}..." for ex in examples
#                                   ]) if examples else "無相關文本樣例"
#         prompt = f"""
#     你是一位 AI 內容鑑別專家。請分析下列文字，列出三個文風特徵，並判斷偏向 Human 或 AI（以括號標註 (Human)/(AI)）。\n\n【參考文本樣例】\n{example_text}\n\n【待檢測文本】\n{sample}
#     """
#         try:
#             rsp = openai.chat.completions.create(model=self.model,
#                                                  temperature=self.temperature,
#                                                  max_tokens=512,
#                                                  messages=[{
#                                                      "role": "system",
#                                                      "content": prompt
#                                                  }])
#             return rsp.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"⚠️ StyleAgent GPT-4 call failed: {e}")
#             return "StyleAgent Error: GPT-4 unavailable"


# class HeuristicsAgent:

#     def __init__(self, tokenizer, lm, kb_manager):
#         self.tok, self.lm = tokenizer, lm
#         self.kb_manager = kb_manager

#     def perplexity(self, txt: str) -> float:
#         try:
#             ids = self.tok(txt,
#                            return_tensors="pt",
#                            truncation=True,
#                            max_length=2048).input_ids.to(self.lm.device)
#             with no_grad_ctx():
#                 loss = self.lm(ids, labels=ids).loss
#             return min(max((math.exp(loss.item()) / 100), 0), 1)
#         except:
#             return 0.5

#     def repetition(self, sents: List[str]) -> float:
#         words = list(itertools.chain.from_iterable(
#             jieba.cut(s) for s in sents))
#         top10 = Counter(words).most_common(10)
#         return min(max(np.mean([c for _, c in top10]) / 10, 0), 1)

#     def burstiness(self, sents: List[str]) -> float:
#         lens = [len(s.split()) for s in sents]
#         return min(max((np.std(lens) + 1e-6) / (np.mean(lens) + 1e-6) / 2, 0),
#                    1)

#     def __call__(self, sents: List[str]) -> Dict[str, float]:
#         text = " ".join(sents[:300])
#         return {
#             "perplexity": self.perplexity(text),
#             "repetition": self.repetition(sents),
#             "burstiness": self.burstiness(sents)
#         }


# class RAGAgent:

#     def __init__(self, kb_manager, top_k=5):
#         self.kb_manager = kb_manager
#         self.top_k = top_k

#     def __call__(self, text: str) -> Dict[str, Any]:
#         retrieved_texts, ai_ratio = self.kb_manager.retrieve(text,
#                                                              top_k=self.top_k)

#         if not retrieved_texts:
#             return {
#                 "style_match_score": 0.5,
#                 "ai_ratio": 0.5,
#                 "retrieved_texts": []
#             }

#         # Calculate a better style match score based on how many documents were actually retrieved
#         # compared to how many were requested
#         retrieval_ratio = len(retrieved_texts) / self.top_k

#         # Optional: Calculate similarity scores between the query and retrieved texts
#         # This would be more meaningful than just counting
#         similarity_scores = []

#         # Use the embedding model to compute similarity
#         query_embedding = _embeddings.embed_query(text)

#         for retrieved_text in retrieved_texts:
#             doc_embedding = _embeddings.embed_query(retrieved_text)
#             # Calculate cosine similarity
#             similarity = np.dot(query_embedding, doc_embedding) / (
#                 np.linalg.norm(query_embedding) *
#                 np.linalg.norm(doc_embedding))
#             similarity_scores.append(similarity)

#         # Average similarity score (if any were calculated)
#         avg_similarity = sum(similarity_scores) / len(
#             similarity_scores) if similarity_scores else 0.5

#         # Combine retrieval ratio and similarity for a more meaningful score
#         style_score = (retrieval_ratio + avg_similarity) / 2

#         return {
#             "style_match_score": style_score,
#             "ai_ratio": ai_ratio,
#             "retrieved_texts": retrieved_texts
#         }


# class DecisionAgent:

#     def __init__(self, model_name: str, temperature=0.3):
#         self.model = model_name
#         self.temperature = temperature

#     def __call__(self, heur: Dict[str, float], style_obs: str,
#                  rag_result: Dict[str, Any],
#                  text_sample: str) -> Dict[str, Union[str, int]]:
#         prompt = f"""
# 根據以下統計特徵與分析，判斷文本來源：

# 【統計特徵】
# - Perplexity: {heur['perplexity']:.2f}
# - Repetition: {heur['repetition']:.2f}
# - Burstiness: {heur['burstiness']:.2f}
# - AI Neighbor Ratio: {rag_result['ai_ratio']:.2f}

# 【GPT-Style 分析】
# {style_obs}

# 【RAG 檢索分析】
# - Style match score: {rag_result['style_match_score']:.2f}

# 【待檢測文本】
# {text_sample[:200]}...

# 請用以下格式輸出：
# 1. Verdict: AI / HUMAN
# 2. Confidence: 0-100（整數）
# 3. Reason: 中文說明三行以內
# """
#         rsp = openai.chat.completions.create(
#             model=self.model,
#             temperature=self.temperature,
#             max_tokens=1024,
#             messages=[{
#                 "role": "system",
#                 "content": prompt
#             }]).choices[0].message.content.splitlines()

#         verdict = "AI" if "AI" in rsp[0].upper() else "HUMAN"
#         conf = int(re.findall(
#             r"\d+",
#             rsp[1])[-1]) if len(rsp) > 1 and re.findall(r"\d+", rsp[1]) else 50
#         reason = " ".join(rsp[2:]) if len(rsp) >= 3 else "N/A"
#         return {"pred": verdict, "confidence": conf, "reason": reason}


# # ================================================================
# # (8) Main Detection Function
# # ================================================================
# preA = PreprocessAgent(kb_manager)
# styA = StyleAgent(GPT4_STYLE_MODEL, kb_manager)

# # updated on 4/30/2025 1706
# # heuA = HeuristicsAgent(
# #     AutoTokenizer.from_pretrained(PPL_LM_ID, trust_remote_code=True),
# #     AutoModelForCausalLM.from_pretrained(PPL_LM_ID,
# #                                          device_map=DEVICE_MAP,
# #                                          trust_remote_code=True,
# #                                          torch_dtype=torch.float16).eval(),
# #     kb_manager)
# heuA = HeuristicsAgent(qwen_tokenizer, qwen_model, kb_manager)
# # updated on 4/30/2025 1706

# ragA = RAGAgent(kb_manager, TOP_K_SIMILAR)
# decA = DecisionAgent(GPT4_DECISION)


# # ================================================================
# # (12) AI Snippet Highlighter - Improved Version
# # ================================================================
# def highlight_ai_segments(text, rag_results, min_similarity=0.7):
#     """
#     Highlights potential AI-generated segments in text based on RAG retrieval results.
#     Uses the improved highlighting method from the first program.
    
#     Args:
#         text (str): The input text to analyze
#         rag_results (dict): Results from RAGAgent including retrieved_texts
#         min_similarity (float): Minimum similarity threshold for highlighting
        
#     Returns:
#         tuple: (highlighted_text, highlight_stats)
#     """
#     from difflib import SequenceMatcher
#     import re

#     # Split input text into sentences
#     sents = segment_cn(text)

#     # Map to track which sentences might be AI-generated
#     ai_sentences = [False] * len(sents)
#     matched_sources = [""] * len(sents)
#     similarity_scores = [0.0] * len(sents)

#     # Check each sentence against retrieved AI examples
#     retrieved_texts = rag_results.get("retrieved_texts", [])
#     for i, sent in enumerate(sents):
#         for rt in retrieved_texts:
#             # Skip very short sentences (less than 10 chars)
#             if len(sent) < 10:
#                 continue

#             # Calculate similarity using SequenceMatcher
#             similarity = SequenceMatcher(None, sent, rt).ratio()

#             # If we find a high similarity match with an AI example
#             if similarity > min_similarity and similarity > similarity_scores[
#                     i]:
#                 ai_sentences[i] = True
#                 matched_sources[i] = rt[:100] + "..." if len(rt) > 100 else rt
#                 similarity_scores[i] = similarity

#     # Create highlighted version with HTML
#     highlighted_parts = []
#     for i, (is_ai, sent, score, source) in enumerate(
#             zip(ai_sentences, sents, similarity_scores, matched_sources)):
#         if is_ai:
#             # Add HTML highlighting for suspected AI text
#             highlighted_parts.append(
#                 f'<span style="background-color: #ffcccc; padding: 2px;" title="AI Similarity: {score:.2f}\nMatched: {source}">🤖 {sent}</span>'
#             )
#         else:
#             highlighted_parts.append(sent)

#     highlighted_text = " ".join(highlighted_parts)

#     # Generate statistics
#     stats = {
#         "total_sentences":
#         len(sents),
#         "ai_sentences":
#         sum(ai_sentences),
#         "human_sentences":
#         len(sents) - sum(ai_sentences),
#         "ai_percentage":
#         round((sum(ai_sentences) / len(sents)) * 100, 2) if sents else 0,
#         "average_similarity":
#         round(sum(similarity_scores) /
#               len(similarity_scores), 3) if similarity_scores else 0
#     }

#     return highlighted_text, stats


# # ================================================================
# # (15) Add Highlighting to the detect_ai Function
# # ================================================================
# def detect_ai(file_path: str) -> Dict:
#     """
#     Main detection function with highlighting capability
#     """
#     raw_txt = read_txt(file_path)
#     if not raw_txt.strip():
#         return {"pred": "HUMAN", "confidence": 0, "reason": "Empty file."}

#     sents = preA(raw_txt)
#     mid_sample = " ".join(sents[len(sents) // 3:len(sents) // 3 + 2])

#     try:
#         style_obs = styA(mid_sample)
#     except Exception as e:
#         style_obs = f"GPT 呼叫失敗，錯誤：{str(e)}"

#     heuristics = heuA(sents)

#     try:
#         rag_result = ragA(mid_sample)
#     except Exception as e:
#         rag_result = {
#             "style_match_score": 0.5,
#             "ai_ratio": 0.5,
#             "retrieved_texts": []
#         }

#     decision = decA(heuristics, style_obs, rag_result, mid_sample)

#     # Add highlighting using the improved method
#     highlighted_text, highlight_stats = highlight_ai_segments(
#         raw_txt, rag_result)

#     decision.update({
#         "file": Path(file_path).name,
#         "perplexity": heuristics.get("perplexity", 0),
#         "repetition": heuristics.get("repetition", 0),
#         "burstiness": heuristics.get("burstiness", 0),
#         "style_obs": style_obs,
#         "rag_style_match": rag_result.get("style_match_score", 0),
#         "rag_ai_ratio": rag_result.get("ai_ratio", 0),
#         "highlighted_text": highlighted_text,
#         "highlight_stats": highlight_stats
#     })
#     return decision


# # ================================================================
# # (8.5) ── User-Test Mode: paste raw text ────────────────────────
# #        (run this block separately for user interaction)
# # ================================================================
# from datetime import datetime

# print("\n 請貼上要分析的文字（按兩次 Enter 結束輸入）:")
# buf = []
# while True:
#     line = input()
#     if not line.strip():
#         break
#     buf.append(line)
# user_text = "\n".join(buf).strip()


# # ================================================================
# # (13) Add Highlighting to detect_from_text Function
# # ================================================================
# def detect_from_text(text: str) -> Dict[str, Any]:
#     """
#     Modified detect_from_text function that includes AI highlighting
#     """
#     if not text:
#         return {
#             "pred": "HUMAN",
#             "confidence": 0,
#             "reason": "Empty text.",
#             "ai_prob": 0.0,
#             "snippet": ""
#         }

#     sents = preA(text)
#     mid = " ".join(sents[len(sents) // 3:len(sents) // 3 +
#                          2]) if sents else text[:300]

#     try:
#         style = styA(mid)
#     except Exception as e:
#         style = f"StyleAgent Error: {str(e)}"

#     heur = heuA(sents)

#     try:
#         rag = ragA(mid)
#     except:
#         rag = {
#             "style_match_score": 0.5,
#             "ai_ratio": 0.5,
#             "retrieved_texts": []
#         }

#     dec = decA(heur, style, rag, mid)
#     ai_prob = dec["confidence"] / 100 if dec[
#         "pred"] == "AI" else 1 - dec["confidence"] / 100

#     # Add highlighting using the improved method
#     highlighted_text, highlight_stats = highlight_ai_segments(text, rag)

#     dec.update({
#         "ai_prob": ai_prob,
#         "perplexity": heur.get("perplexity", 0),
#         "repetition": heur.get("repetition", 0),
#         "burstiness": heur.get("burstiness", 0),
#         "style_obs": style,
#         "rag_style_match": rag.get("style_match_score", 0),
#         "rag_ai_ratio": rag.get("ai_ratio", 0),
#         "snippet": mid,
#         "timestamp": datetime.now().isoformat(),
#         "highlighted_text": highlighted_text,
#         "highlight_stats": highlight_stats
#     })
#     return dec


# # 👉 Run detection
# if user_text:
#     print("\n 分析結果：")
#     user_result = detect_from_text(user_text)
#     print(json.dumps(user_result, ensure_ascii=False, indent=2))

#     # Optional: Save to Drive
#     test_output_path = "/content/drive/MyDrive/user_test_result.json"
#     with open(test_output_path, "w", encoding="utf-8") as f:
#         json.dump(user_result, f, ensure_ascii=False, indent=2)
#     print(f"\n 已儲存至 Google Drive：{test_output_path}")



# # # ================================================================
# # # (10) Backup Results to Google Drive
# # # ================================================================
# # from google.colab import drive
# # drive.mount('/content/drive')

# # BACKUP_DIR = '/content/drive/MyDrive/ai_detection_outputs_backup'
# # shutil.copytree('/content/ai_detection_outputs', BACKUP_DIR, dirs_exist_ok=True)
# # print(f"\n\u2705 Results saved to Google Drive: {BACKUP_DIR}")

# # # ================================================================
# # # (11) Auto-Disconnect Runtime
# # # ================================================================
# # print("\u23F3 Disconnecting runtime in 5 seconds...")
# # time.sleep(5)
# # os.kill(os.getpid(), 9)

# # DONE!
