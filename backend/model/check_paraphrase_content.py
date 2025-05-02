#@title agentic_rag_4o_plagiarism_detection_demo_with_flask

# !pip install torch transformers sentence-transformers langchain langchain-community faiss-cpu openai

import os
import re
import gc
import torch
import torch.nn.functional as F
from difflib import SequenceMatcher
from typing import List
import numpy

from dotenv import load_dotenv

# For the pipeline
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

# For AI detection (optional)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# For calling GPT-4o
from openai import OpenAI

device = torch.device("cuda:0")
torch.cuda.set_device(0)
print(f"Using device: {device}")

# ========== DEMO AI Detector Model Setup ==========
detector_model_name = "roberta-base-openai-detector"
det_tokenizer = AutoTokenizer.from_pretrained(detector_model_name)
det_model = AutoModelForSequenceClassification.from_pretrained(
    detector_model_name).to(device)
det_model.eval()

# Using your GPT-4o model (replace with your actual key as needed)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))


# ===============================================================
# 1) OpenAI-based text generation function (using gpt-4o)
# ===============================================================
def generate_with_openai_api(prompt: str,
                             max_tokens: int = 512,
                             temperature: float = 0.7,
                             top_p: float = 0.9,
                             frequency_penalty: float = 0.0,
                             presence_penalty: float = 0.0) -> str:
    """
    Sends 'prompt' to the GPT-4o model via the openai API client.
    Returns the model-generated string.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI API Error:", e)
        return "【API Error】"


# ===============================================================
# 2) AI Content Detection Helper (OPTIONAL)
# ===============================================================
def compute_ai_generated_probability(text: str) -> float:
    """
    Roughly estimate how likely a text is AI-generated using 'roberta-base-openai-detector'.
    Returns a float in [0,1], where 1 means "very likely AI-generated".
    """
    max_length = 512  # cutoff
    if len(text) > max_length * 4:
        text = text[:max_length * 4]

    inputs = det_tokenizer(text,
                           truncation=True,
                           padding=True,
                           return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = det_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    # index 0 => "Real/Human", index 1 => "Fake/AI"
    ai_prob = probs[0][1].item()
    return ai_prob


# ===============================================================
# 3) Similarity Helpers
# ===============================================================
def split_into_paragraphs(text: str):
    paragraphs = []
    for blk in text.split("\n\n"):
        blk = blk.strip()
        if blk:
            paragraphs.append(blk)
    return paragraphs


# Updated on 04/22/2025
def split_into_sentences_chinese(text):
    pattern = re.compile(r'([^。！？]*[。！？])', re.U)
    sentences = pattern.findall(text)
    left = pattern.sub('', text)
    if left:
        sentences.append(left)
    return [s for s in sentences if s.strip()]


# Updated on 04/22/2025


def highlight_matches(original: str, submitted: str, min_len=4):
    """
    Return matched segments (>= min_len) found in `submitted`
    that also appear in `original`.
    """
    matcher = SequenceMatcher(None, original, submitted)
    matched_segments = []
    for match in matcher.get_matching_blocks():
        if match.size >= min_len:
            segment = submitted[match.b:match.b + match.size]
            matched_segments.append(segment)
    return matched_segments


def compute_embedding_similarity(embedding_model, text1, text2):
    emb1 = embedding_model.embed_query(text1)
    emb2 = embedding_model.embed_query(text2)
    t1 = torch.tensor(emb1).to(device)
    t2 = torch.tensor(emb2).to(device)
    return F.cosine_similarity(t1, t2, dim=0).item()


# ===============================================================
# 4) Build a FAISS Vector Store from source docs
# ===============================================================
def build_paraphrase_vector_db(dirs_list: List[str]):
    """
    Build a FAISS vector store from ALL .txt files in each directory.
    Each file is split into paragraphs.
    """
    all_paragraph_docs = []
    file_paths = []
    total_file_count = 0

    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cuda:0"})

    for pages_dir in dirs_list:
        processed_files = 0
        txt_files = sorted(
            [f for f in os.listdir(pages_dir) if f.endswith(".txt")])

        # Use all files instead of just limiting to files_per_folder
        for filename in txt_files:
            file_path = os.path.join(pages_dir, filename)
            file_paths.append(file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                paragraphs = split_into_paragraphs(text)
                for para in paragraphs:
                    doc = Document(page_content=para,
                                   metadata={"file_path": file_path})
                    all_paragraph_docs.append(doc)

            processed_files += 1
            total_file_count += 1

        print(f"Loaded {processed_files} .txt files from {pages_dir}")

    print(
        f"Total source files loaded: {total_file_count}, paragraphs: {len(all_paragraph_docs)}"
    )

    vector_db = FAISS.from_documents(all_paragraph_docs, embedding_model)
    return vector_db, embedding_model, file_paths


# ===============================================================
# 5) CrossEncoder for re-ranking
# ===============================================================
cross_encoder_model = "BAAI/bge-reranker-v2-m3"
cross_encoder = CrossEncoder(cross_encoder_model, device="cuda:0")


# ===============================================================
# 6) The main pipeline
# ===============================================================
def cooperate_plagiarism_check(user_text: str,
                               vector_db: FAISS,
                               embedding_model,
                               exclude_file_path: str = None,
                               initial_k=10,
                               rerank_top_k=3):
    """
    1) Similarity Search in FAISS
    2) Rerank with CrossEncoder
    3) 'Main' LLM analysis
    4) 'Expert' feedback x2
    5) 'Judge' verdict => "ACCEPT" or "ABSTAIN"

    Returns a dict including judge_output, etc.
    """
    # (1) Similarity search
    retrieved_initial = vector_db.similarity_search(user_text, k=initial_k)

    # Exclude same file (avoid self-match)
    if exclude_file_path is not None:
        retrieved_initial = [
            d for d in retrieved_initial
            if d.metadata.get("file_path", "") != exclude_file_path
        ]

    # (2) Rerank with CrossEncoder
    pairs = [(user_text, d.page_content) for d in retrieved_initial]
    cross_scores = cross_encoder.predict(pairs)
    docs_scores = list(zip(retrieved_initial, cross_scores))
    docs_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = docs_scores[:rerank_top_k]

    doc_info_list = []
    for doc, sc in top_docs:
        emb_sim = compute_embedding_similarity(embedding_model, user_text,
                                               doc.page_content)
        overlaps = highlight_matches(doc.page_content, user_text)
        doc_info_list.append({
            "content": doc.page_content,
            "cross_score": sc,
            "embedding_sim": emb_sim,
            "overlaps": overlaps,
            "file_path": doc.metadata.get("file_path", "")
        })

    top_texts = [info["content"] for info in doc_info_list]

    # (3) Main model
    prompt_main = ("你是主要分析模型 (Main Model)。以下是用户的段落，以及最相似的文献段落。\n"
                   "检查是否有文字重叠，是否存在抄袭倾向。\n\n"
                   f"[用户段落]:\n{user_text}\n\n"
                   f"[最相似文献段落Top {rerank_top_k}]:\n" + "\n".join(top_texts) +
                   "\n\n请用中文回答，并简要说明你的想法(1~3句话)。")
    main_analysis = generate_with_openai_api(prompt_main)

    # (4) Expert 1
    prompt_fb1 = ("你是检查专家(Expert 1)，请审阅以下资料：\n"
                  f"[用户段落]: {user_text}\n"
                  f"[Main Model 分析]: {main_analysis}\n"
                  f"[相似文献Top {rerank_top_k}]:\n" + "\n".join(top_texts) +
                  "\n\n请简短指出Main Model的分析是否可信，是否遗漏了什么。\n")
    fb1 = generate_with_openai_api(prompt_fb1)

    # (5) Expert 2
    prompt_fb2 = ("你是检查专家(Expert 2)，请审阅以下资料：\n"
                  f"[用户段落]: {user_text}\n"
                  f"[Main Model 分析]: {main_analysis}\n"
                  f"[相似文献Top {rerank_top_k}]:\n" + "\n".join(top_texts) +
                  "\n\n请简短指出Main Model的分析是否可信，是否遗漏了什么。\n")
    fb2 = generate_with_openai_api(prompt_fb2)

    # Combine feedbacks
    feedbacks = [f"Expert 1 Feedback: {fb1}", f"Expert 2 Feedback: {fb2}"]

    # (6) Judge
    judge_prompt = f"""
你是评审模型(Judge)。以下是资料：
[用户段落]: {user_text}
[Main Model 分析]: {main_analysis}
{feedbacks[0]}
{feedbacks[1]}

请你判断，是否应该"ABSTAIN"(因为怀疑有抄袭或不确定)或"ACCEPT"。
请用JSON格式回答:
{{
  "verdict": "...",
  "reason": "..."
}}

"verdict"只能是"ACCEPT"或"ABSTAIN"。
"reason"用中文简述根据。
    """
    judge_output = generate_with_openai_api(judge_prompt)

    # (7) Average cross-score
    if len(top_docs) > 0:
        avg_confidence = sum([sc for _, sc in top_docs]) / len(top_docs)
    else:
        avg_confidence = 0.0

    # Extract plagiarism percentage based on verdict and scores
    verdict_match = re.search(r'"verdict"\s*:\s*"(\w+)"', judge_output)
    verdict = verdict_match.group(1) if verdict_match else "UNKNOWN"

    # Calculate plagiarism percentage
    # Higher if ABSTAIN, lower if ACCEPT
    if verdict == "ABSTAIN":
        # Higher plagiarism score for ABSTAIN
        plagiarism_percentage = 50 + (avg_confidence * 50)  # Scale to 50-100%
    else:
        # Lower plagiarism score for ACCEPT
        plagiarism_percentage = avg_confidence * 40  # Scale to 0-40%

    # Ensure within 0-100 range
    plagiarism_percentage = max(0, min(100, plagiarism_percentage))

    # === Build snippet(s) from EVERY overlap =========================
    all_overlaps = []
    for info in doc_info_list:  # gather *all* matches
        all_overlaps.extend(info["overlaps"])

    all_overlaps = list(dict.fromkeys(all_overlaps))  # deduplicate, keep order

    # (A) complete concatenation for the UI
    # plagiarism_snippet_full = " ".join(all_overlaps)
    plagiarism_snippet_full = all_overlaps

    # (B) longest chunk – keeps old behaviour for backward‑compat
    plagiarism_snippet = max(all_overlaps, key=len) if all_overlaps else ""

    # fallback if we somehow had no overlaps at all
    # if not plagiarism_snippet_full and doc_info_list:
    #     snippet_len = len(doc_info_list[0]["content"])
    #     plagiarism_snippet_full = doc_info_list[0][
    #         "content"][:snippet_len] + "…"
    original_snippet = plagiarism_snippet_full

    # ----- GPT-4.1 Sentence-level Highlighting -----
    user_sentences = split_into_sentences_chinese(user_text)
    plag_highlight_prompt = (
        "你是学术剽窃检测AI。下面是用户逐句分割的段落，以及最相似的参考文献段落。\n"
        "请你判断哪些用户句子涉嫌剽窃，请只用[PLAG]...[/PLAG]包裹这些句子输出原文（不必解释，保持原句原格式，剩下句子不要输出）。\n"
        "如果没有任何句子涉嫌剽窃，则只输出空字符串。\n\n"
        "【用户分句】:\n" +
        "\n".join([f"{i+1}. {s}" for i, s in enumerate(user_sentences)]) +
        "\n\n【最相似文献段落】:\n" + "\n".join(top_texts))
    gpt4_plag_sentence_mark = generate_with_openai_api(plag_highlight_prompt,
                                                       max_tokens=256)
    plagiarized_sents = re.findall(r'\[PLAG\](.+?)\[/PLAG\]',
                                   gpt4_plag_sentence_mark,
                                   flags=re.S)

    def highlight_sentences(user_sentences, plag_sents):
        result = []
        for s in user_sentences:
            if s.strip() in [ps.strip() for ps in plag_sents]:
                result.append(f'<mark style="background:orange">{s}</mark>')
            else:
                result.append(s)
        return ''.join(result)

    gpt_sentence_highlight_html = highlight_sentences(user_sentences,
                                                      plagiarized_sents)
    # gpt4_1_snippet = " ".join(plagiarized_sents)
    gpt4_1_snippet = plagiarized_sents

    # updated on 04/23/2025
    # ----- SMART: PICK CLOSER TO PERCENTAGE -----
    def snippet_score_ratio(snippet, user_text):
        if not snippet or not user_text:
            return 0.0
        return len(snippet) / max(1, len(user_text))

    ratio_original = snippet_score_ratio(original_snippet, user_text)
    ratio_gpt41 = snippet_score_ratio(gpt4_1_snippet, user_text)
    target_ratio = plagiarism_percentage / 100

    if abs(ratio_original - target_ratio) < abs(ratio_gpt41 - target_ratio):
        plagiarism_snippet = original_snippet
    else:
        plagiarism_snippet = gpt4_1_snippet
    # updated on 04/23/2025
    # ✅ Forcefully return 0% and empty snippet if judged as human
    if verdict == "ACCEPT":
        plagiarism_percentage = 0.0
        plagiarism_snippet = []

    gc.collect()
    torch.cuda.empty_cache()

    def transfer_numpy_to_float(data):
        if isinstance(data, dict):
            return {k: transfer_numpy_to_float(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [transfer_numpy_to_float(item) for item in data]
        elif isinstance(data, (numpy.integer, )):
            return int(data)
        elif isinstance(data, (numpy.floating, )):
            return float(data)
        elif isinstance(data, numpy.ndarray):
            return data.tolist()
        else:
            return data

    return {
        # list 中如果有非 float型態 (如 numpy)，轉換成json時會報錯，這邊使用 transfer_numpy_to_float 轉換
        "top_docs_info": transfer_numpy_to_float(doc_info_list),
        "main_analysis": main_analysis,
        "feedbacks": transfer_numpy_to_float(feedbacks),
        "judge_output": judge_output,
        "plagiarism_confidence": round(float(round(avg_confidence * 100, 2))),
        "plagiarism_percentage": round(float(round(plagiarism_percentage, 2))),
        "plagiarism_snippet": plagiarism_snippet,
        "verdict": verdict
    }
