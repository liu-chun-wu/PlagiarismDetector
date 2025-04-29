# ================================================================
# (0) Install / Upgrade Packages
# ================================================================
# !pip install -U openai tiktoken transformers accelerate sentencepiece sentence-transformers nltk jieba pandas tqdm scikit-learn bitsandbytes chromadb langchain spacy networkx
# !pip install -U langchain-huggingface
# !python -m spacy download zh_core_web_sm
# !pip install -U langchain-community

# ================================================================
# (1) Imports
# ================================================================
# import sys
# import sys
# sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, glob, math, json, itertools, random, warnings, contextlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch, nltk, jieba, spacy, networkx as nx
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # <-- 改這裡
from langchain.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import shutil
import time

nltk.download('punkt', quiet=True)
warnings.filterwarnings("ignore", category=UserWarning)

# ================================================================
#  (2) Basic Settings
# ================================================================
SOURCE_DIRS = [
    "/content/drive/MyDrive/paraphrased_dataset/source/ncu_2019",
    "/content/drive/MyDrive/paraphrased_dataset/source/ncu_2020"
]
PARAPHRASED_DIRS = [
    "/content/drive/MyDrive/ncu_2020", "/content/drive/MyDrive/nsyu_2019"
]

openai.api_key = "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"
GPT4_STYLE_MODEL = "gpt-4.1"
GPT4_DECISION = "gpt-4.1"
PPL_LM_ID = "Qwen/Qwen2.5-7B"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_SIMILAR = 5

RAG_KNOWLEDGE_DIR = Path("/content/rag_knowledge")
RAG_KNOWLEDGE_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR = Path("/content/ai_detection_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# (3) Utility Functions
# ================================================================
def read_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore").strip()


def segment_cn(text: str) -> List[str]:
    sents = re.split(r"(?<=[。！？!?；;])\s*", re.sub(r"\s+", " ", text))
    return [s for s in sents if s]


@contextlib.contextmanager
def no_grad_ctx():
    with torch.no_grad():
        yield


# ================================================================
# (4) Entity Extractor
# ================================================================
print("\u23F3 Loading spaCy model …")
_spacy_nlp = spacy.load("zh_core_web_sm")


class EntityExtractor:

    def __call__(self, text: str) -> List[str]:
        doc = _spacy_nlp(text)
        return [ent.text for ent in doc.ents]


# ================================================================
# (5) HyperGraph Database
# ================================================================
class HyperGraphDB:

    def __init__(self):
        self.graph = nx.Graph()
        self.entity_texts = defaultdict(list)

    def add(self, text: str, entities: List[str]):
        if not entities: return
        for e in entities:
            self.graph.add_node(e)
            self.entity_texts[e].append(text)
        for a, b in itertools.combinations(set(entities), 2):
            self.graph.add_edge(a, b)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        ents = EntityExtractor()(query)
        hits = []
        for e in ents:
            hits.extend(self.entity_texts.get(e, []))
            for n in self.graph.neighbors(e):
                hits.extend(self.entity_texts.get(n, []))
        return list(dict.fromkeys(hits))[:top_k]


# ================================================================
# (6) Knowledge Base Manager (New Version)
# ================================================================
print("\u23F3 Building embedding model …")
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                    model_kwargs={"device": DEVICE_MAP})


class KBManager:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "，", " "])
        self.hg = HyperGraphDB()
        self.faiss = None
        self._build_kb()

    def _build_kb(self):
        docs = []
        for d in SOURCE_DIRS:
            for fp in glob.glob(f"{d}/**/*.txt", recursive=True):
                content = read_txt(fp)
                for chunk in self.text_splitter.split_text(content):
                    ents = EntityExtractor()(chunk)
                    self.hg.add(chunk, ents)
                    docs.append(
                        Document(page_content=chunk,
                                 metadata={"label": "HUMAN"}))
        for d in PARAPHRASED_DIRS:
            for fp in glob.glob(f"{d}/**/*.txt", recursive=True):
                content = read_txt(fp)
                for chunk in self.text_splitter.split_text(content):
                    ents = EntityExtractor()(chunk)
                    self.hg.add(chunk, ents)
                    docs.append(
                        Document(page_content=chunk, metadata={"label": "AI"}))
        print(
            f"\ud83d\udcda  Vectorising {len(docs)} KB chunks … (this may take a while)"
        )
        self.faiss = FAISS.from_documents(docs, embedding=_embeddings)

    def retrieve(self,
                 query: str,
                 k: int = TOP_K_SIMILAR) -> Tuple[List[str], float]:
        # Get results from HyperGraphDB
        hg_texts = self.hg.retrieve(query, k)

        # Get results from FAISS vector search
        docs = self.faiss.similarity_search_with_score(query, k)

        # Combine results
        all_docs = []

        # Add HyperGraphDB results with their labels
        if hg_texts:
            for text in hg_texts:
                # Find this text in your original docs to get its label
                for doc in self.faiss.index_to_docstore_id.values():
                    doc_obj = self.faiss.docstore.search(doc)
                    if doc_obj.page_content == text:
                        all_docs.append(
                            (doc_obj,
                             0))  # Use similarity score of 0 (perfect match)
                        break

        # Add FAISS results
        if docs:
            all_docs.extend(docs)

        # If no documents were found at all
        if not all_docs:
            return [], 0.5

        # Calculate AI ratio based on all retrieved documents
        labels = [doc[0].metadata.get("label", "HUMAN") for doc in all_docs]
        ai_ratio = labels.count("AI") / len(labels)

        return [doc[0].page_content for doc in all_docs], ai_ratio


kb_manager = KBManager()


# (7) Agents (Updated to use new kb_manager)
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
        example_text = "\n".join([f"- {ex[:100]}..." for ex in examples
                                  ]) if examples else "無相關文本樣例"
        prompt = f"""
    你是一位 AI 內容鑑別專家。請分析下列文字，列出三個文風特徵，並判斷偏向 Human 或 AI（以括號標註 (Human)/(AI)）。\n\n【參考文本樣例】\n{example_text}\n\n【待檢測文本】\n{sample}
    """
        try:
            rsp = openai.chat.completions.create(model=self.model,
                                                 temperature=self.temperature,
                                                 max_tokens=512,
                                                 messages=[{
                                                     "role": "system",
                                                     "content": prompt
                                                 }])
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ StyleAgent GPT-4 call failed: {e}")
            return "StyleAgent Error: GPT-4 unavailable"


class HeuristicsAgent:

    def __init__(self, tokenizer, lm, kb_manager):
        self.tok, self.lm = tokenizer, lm
        self.kb_manager = kb_manager

    def perplexity(self, txt: str) -> float:
        try:
            ids = self.tok(txt,
                           return_tensors="pt",
                           truncation=True,
                           max_length=2048).input_ids.to(self.lm.device)
            with no_grad_ctx():
                loss = self.lm(ids, labels=ids).loss
            return min(max((math.exp(loss.item()) / 100), 0), 1)
        except:
            return 0.5

    def repetition(self, sents: List[str]) -> float:
        words = list(itertools.chain.from_iterable(
            jieba.cut(s) for s in sents))
        top10 = Counter(words).most_common(10)
        return min(max(np.mean([c for _, c in top10]) / 10, 0), 1)

    def burstiness(self, sents: List[str]) -> float:
        lens = [len(s.split()) for s in sents]
        return min(max((np.std(lens) + 1e-6) / (np.mean(lens) + 1e-6) / 2, 0),
                   1)

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
        retrieved_texts, ai_ratio = self.kb_manager.retrieve(text,
                                                             top_k=self.top_k)

        if not retrieved_texts:
            return {
                "style_match_score": 0.5,
                "ai_ratio": 0.5,
                "retrieved_texts": []
            }

        # Calculate a better style match score based on how many documents were actually retrieved
        # compared to how many were requested
        retrieval_ratio = len(retrieved_texts) / self.top_k

        # Optional: Calculate similarity scores between the query and retrieved texts
        # This would be more meaningful than just counting
        similarity_scores = []

        # Use the embedding model to compute similarity
        query_embedding = _embeddings.embed_query(text)

        for retrieved_text in retrieved_texts:
            doc_embedding = _embeddings.embed_query(retrieved_text)
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) *
                np.linalg.norm(doc_embedding))
            similarity_scores.append(similarity)

        # Average similarity score (if any were calculated)
        avg_similarity = sum(similarity_scores) / len(
            similarity_scores) if similarity_scores else 0.5

        # Combine retrieval ratio and similarity for a more meaningful score
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
                 rag_result: Dict[str, Any],
                 text_sample: str) -> Dict[str, Union[str, int]]:
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
            }]).choices[0].message.content.splitlines()

        verdict = "AI" if "AI" in rsp[0].upper() else "HUMAN"
        conf = int(re.findall(
            r"\d+",
            rsp[1])[-1]) if len(rsp) > 1 and re.findall(r"\d+", rsp[1]) else 50
        reason = " ".join(rsp[2:]) if len(rsp) >= 3 else "N/A"
        return {"pred": verdict, "confidence": conf, "reason": reason}


# ================================================================
# (8) Main Detection Function
# ================================================================
preA = PreprocessAgent(kb_manager)
styA = StyleAgent(GPT4_STYLE_MODEL, kb_manager)
heuA = HeuristicsAgent(
    AutoTokenizer.from_pretrained(PPL_LM_ID, trust_remote_code=True),
    AutoModelForCausalLM.from_pretrained(PPL_LM_ID,
                                         device_map=DEVICE_MAP,
                                         trust_remote_code=True,
                                         torch_dtype=torch.float16).eval(),
    kb_manager)
ragA = RAGAgent(kb_manager, TOP_K_SIMILAR)
decA = DecisionAgent(GPT4_DECISION)


# ================================================================
# (12) AI Snippet Highlighter - Improved Version
# ================================================================
def highlight_ai_segments(text, rag_results, min_similarity=0.7):
    """
    Highlights potential AI-generated segments in text based on RAG retrieval results.
    Uses the improved highlighting method from the first program.
    
    Args:
        text (str): The input text to analyze
        rag_results (dict): Results from RAGAgent including retrieved_texts
        min_similarity (float): Minimum similarity threshold for highlighting
        
    Returns:
        tuple: (highlighted_text, highlight_stats)
    """
    from difflib import SequenceMatcher
    import re

    # Split input text into sentences
    sents = segment_cn(text)

    # Map to track which sentences might be AI-generated
    ai_sentences = [False] * len(sents)
    matched_sources = [""] * len(sents)
    similarity_scores = [0.0] * len(sents)

    # Check each sentence against retrieved AI examples
    retrieved_texts = rag_results.get("retrieved_texts", [])
    for i, sent in enumerate(sents):
        for rt in retrieved_texts:
            # Skip very short sentences (less than 10 chars)
            if len(sent) < 10:
                continue

            # Calculate similarity using SequenceMatcher
            similarity = SequenceMatcher(None, sent, rt).ratio()

            # If we find a high similarity match with an AI example
            if similarity > min_similarity and similarity > similarity_scores[
                    i]:
                ai_sentences[i] = True
                matched_sources[i] = rt[:100] + "..." if len(rt) > 100 else rt
                similarity_scores[i] = similarity

    # Create highlighted version with HTML
    highlighted_parts = []
    for i, (is_ai, sent, score, source) in enumerate(
            zip(ai_sentences, sents, similarity_scores, matched_sources)):
        if is_ai:
            # Add HTML highlighting for suspected AI text
            highlighted_parts.append(
                f'<span style="background-color: #ffcccc; padding: 2px;" title="AI Similarity: {score:.2f}\nMatched: {source}">🤖 {sent}</span>'
            )
        else:
            highlighted_parts.append(sent)

    highlighted_text = " ".join(highlighted_parts)

    # Generate statistics
    stats = {
        "total_sentences":
        len(sents),
        "ai_sentences":
        sum(ai_sentences),
        "human_sentences":
        len(sents) - sum(ai_sentences),
        "ai_percentage":
        round((sum(ai_sentences) / len(sents)) * 100, 2) if sents else 0,
        "average_similarity":
        round(sum(similarity_scores) /
              len(similarity_scores), 3) if similarity_scores else 0
    }

    return highlighted_text, stats


# ================================================================
# (15) Add Highlighting to the detect_ai Function
# ================================================================
def detect_ai(file_path: str) -> Dict:
    """
    Main detection function with highlighting capability
    """
    raw_txt = read_txt(file_path)
    if not raw_txt.strip():
        return {"pred": "HUMAN", "confidence": 0, "reason": "Empty file."}

    sents = preA(raw_txt)
    mid_sample = " ".join(sents[len(sents) // 3:len(sents) // 3 + 2])

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

    # Add highlighting using the improved method
    highlighted_text, highlight_stats = highlight_ai_segments(
        raw_txt, rag_result)

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


# ================================================================
# (8.5) ── User-Test Mode: paste raw text ────────────────────────
#        (run this block separately for user interaction)
# ================================================================
from datetime import datetime

print("\n📋 請貼上要分析的文字（按兩次 Enter 結束輸入）:")
buf = []
while True:
    line = input()
    if not line.strip():
        break
    buf.append(line)
user_text = "\n".join(buf).strip()


# ================================================================
# (13) Add Highlighting to detect_from_text Function
# ================================================================
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
    mid = " ".join(sents[len(sents) // 3:len(sents) // 3 +
                         2]) if sents else text[:300]

    try:
        style = styA(mid)
    except Exception as e:
        style = f"StyleAgent Error: {str(e)}"

    heur = heuA(sents)

    try:
        rag = ragA(mid)
    except:
        rag = {
            "style_match_score": 0.5,
            "ai_ratio": 0.5,
            "retrieved_texts": []
        }

    dec = decA(heur, style, rag, mid)
    ai_prob = dec["confidence"] / 100 if dec[
        "pred"] == "AI" else 1 - dec["confidence"] / 100

    # Add highlighting using the improved method
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


# 👉 Run detection
if user_text:
    print("\n📊 分析結果：")
    user_result = detect_from_text(user_text)
    print(json.dumps(user_result, ensure_ascii=False, indent=2))

    # Optional: Save to Drive
    test_output_path = "/content/drive/MyDrive/user_test_result.json"
    with open(test_output_path, "w", encoding="utf-8") as f:
        json.dump(user_result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已儲存至 Google Drive：{test_output_path}")

# # ================================================================
# # (9) Batch Detection and Evaluation
# # ================================================================
# all_files = (
#     [(p, "HUMAN") for d in SOURCE_DIRS for p in glob.glob(f"{d}/**/*.txt", recursive=True)] +
#     [(p, "AI") for d in PARAPHRASED_DIRS for p in glob.glob(f"{d}/**/*.txt", recursive=True)]
# )
# print(f"\U0001f5c2️ Total files: {len(all_files)}")

# # (NEW) Detect the first file and print out
# first_file_path = all_files[0][0]  # 第1個 file 的路徑
# first_report = detect_ai(first_file_path)

# # 印出來看 StyleAgent / DecisionAgent 有沒有正常
# print("\n📋 First detection report:")
# print(json.dumps(first_report, ensure_ascii=False, indent=2))

# # 同時存一份到 Google Drive 以防
# first_save_path = '/content/drive/MyDrive/first_detect_result.json'
# with open(first_save_path, 'w', encoding='utf-8') as f:
#     json.dump(first_report, f, ensure_ascii=False, indent=2)

# print(f"\n✅ First detection report saved to {first_save_path}\n")

# y_true, y_pred = [], []
# for fp, label in tqdm(all_files, desc="Detecting"):
#     report = detect_ai(fp)
#     y_true.append(label)
#     y_pred.append(report["pred"])
#     with open(OUTPUT_DIR / (Path(fp).stem + ".json"), "w", encoding="utf-8") as f:
#         json.dump(report, f, ensure_ascii=False, indent=2)

# print("\n\ud83d\udcca Classification Report")
# print(classification_report(y_true, y_pred, labels=["AI", "HUMAN"], zero_division=0, digits=3))

# print(f"Accuracy  : {accuracy_score(y_true, y_pred):.3f}")
# print(f"Macro-F1  : {f1_score(y_true, y_pred, average='macro'):.3f}")

# # ================================================================
# # (10) Backup Results to Google Drive
# # ================================================================
# from google.colab import drive
# drive.mount('/content/drive')

# BACKUP_DIR = '/content/drive/MyDrive/ai_detection_outputs_backup'
# shutil.copytree('/content/ai_detection_outputs', BACKUP_DIR, dirs_exist_ok=True)
# print(f"\n\u2705 Results saved to Google Drive: {BACKUP_DIR}")

# # ================================================================
# # (11) Auto-Disconnect Runtime
# # ================================================================
# print("\u23F3 Disconnecting runtime in 5 seconds...")
# time.sleep(5)
# os.kill(os.getpid(), 9)

# DONE!
