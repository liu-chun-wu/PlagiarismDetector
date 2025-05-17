# #@title agentic_rag_4o_plagiarism_detection_demo_with_flask

# # !pip install torch transformers sentence-transformers langchain langchain-community faiss-cpu openai

import os
import re
import gc
import torch
import torch.nn.functional as F
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv
import numpy

from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from openai import OpenAI

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# ========== Env and Model Setup ==========
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use only GPU 1
load_dotenv()
openai_api_key = os.getenv("OPENAI_APIKEY")
client = OpenAI(api_key=openai_api_key)

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": str(device)}
)
cross_encoder = CrossEncoder(RERANKER_MODEL_NAME, device=str(device))

# ========== Utilities ==========
def split_into_paragraphs(text: str):
    return [blk.strip() for blk in text.split("\n\n") if blk.strip()]

def compute_embedding_similarity(embedding_model, text1, text2):
    emb1 = embedding_model.embed_query(text1)
    emb2 = embedding_model.embed_query(text2)
    t1 = torch.tensor(emb1).to(device)
    t2 = torch.tensor(emb2).to(device)
    return F.cosine_similarity(t1, t2, dim=0).item()

def highlight_matches(original: str, submitted: str, min_len=4):
    matcher = SequenceMatcher(None, original, submitted)
    matched_segments = []
    for match in matcher.get_matching_blocks():
        if match.size >= min_len:
            segment = submitted[match.b:match.b + match.size]
            matched_segments.append(segment)
    return matched_segments

def build_paraphrase_vector_db(dirs_list: List[str]):
    all_paragraph_docs = []
    file_paths = []
    total_file_count = 0
    for pages_dir in dirs_list:
        txt_files = sorted([f for f in os.listdir(pages_dir) if f.endswith(".txt")])
        for filename in txt_files:
            file_path = os.path.join(pages_dir, filename)
            file_paths.append(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                paragraphs = split_into_paragraphs(text)
                for para in paragraphs:
                    doc = Document(page_content=para, metadata={"file_path": file_path})
                    all_paragraph_docs.append(doc)
            total_file_count += 1
        print(f"Loaded {len(txt_files)} .txt files from {pages_dir}")
    print(f"Total source files loaded: {total_file_count}, paragraphs: {len(all_paragraph_docs)}")
    vector_db = FAISS.from_documents(all_paragraph_docs, embedding_model)
    return vector_db, embedding_model, file_paths

def load_vector_db(txt_dirs: List[str], faiss_path: Optional[str] = None):
    if faiss_path and os.path.exists(faiss_path):
        print(f"Loading FAISS from {faiss_path}")
        return FAISS.load_local(faiss_path, embedding_model), embedding_model, []
    print("Building new FAISS index...")
    vdb, emb, files = build_paraphrase_vector_db(txt_dirs)
    if faiss_path:
        vdb.save_local(faiss_path)
    return vdb, emb, files

def compute_statistical_confidence(cross_scores: List[float]):
    # BAAI/bge-retranker: 0-5 rating, turn into 0-1
    norm_scores = [score / 5.0 for score in cross_scores]  # adjust if your model is different
    avg = sum(norm_scores)/len(norm_scores) if norm_scores else 0.0
    return round(avg, 4)

# ------------ AGENTIC LangGraph Setup --------------

class RagState(TypedDict, total=False):
    user_text: str
    retrieved: Optional[List[Document]]
    doc_scores: Optional[List[Dict[str, Any]]]
    main_analysis: Optional[str]
    expert_feedbacks: Annotated[List[str], add_messages]
    revised: Optional[bool]
    judge_output: Optional[str]
    sim_seq: Optional[int]
    sim_cosine: Optional[int]
    plagiarism_percent: Optional[int]
    plagiarism_snippet: Optional[List[str]]
    plagiarism_confidence: Optional[float]

def retrieve_node(state: RagState, vdb: FAISS, k: int = 10):
    user_text = state["user_text"]
    docs = vdb.similarity_search(user_text, k=k)
    scores = cross_encoder.predict([(user_text, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:3]
    top_paragraph = ranked[0][0].page_content if ranked else ""
    sim_seq = int(100 * SequenceMatcher(None, user_text, top_paragraph).ratio())
    overlaps = highlight_matches(top_paragraph, user_text, min_len=4)
    sim_cosine = int(100 * compute_embedding_similarity(embedding_model, user_text, top_paragraph))
    plagiarism_percent = int((sim_seq + sim_cosine) / 2)
    plagiarism_confidence = compute_statistical_confidence([s for _, s in ranked])
    return {
        "retrieved": [d for d, _ in ranked],
        "doc_scores": [{"doc": d, "cross_score": float(s)} for d, s in ranked],
        "sim_seq": sim_seq,
        "sim_cosine": sim_cosine,
        "plagiarism_percent": plagiarism_percent,
        "plagiarism_snippet": overlaps,
        "plagiarism_confidence": plagiarism_confidence
    }

def call_gpt(messages, *, max_tokens=1024, temperature=0.7):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def main_node(state: RagState):
    ctx = "\n".join(d.page_content for d in state["retrieved"])
    sim_seq = state.get("sim_seq", 0)
    sim_cosine = state.get("sim_cosine", 0)
    plagiarism_confidence = state.get("plagiarism_confidence", 0.0)
    similarities_desc = (f"字符级重合 sequence_match={sim_seq}%，"
                         f"语义相似 cosine={sim_cosine}%，"
                         f"置信度 confidence={plagiarism_confidence:.2f}")
    prompt = [
        {"role": "system",
        "content": '''
        你是高校学术查重初审AI（Main）。
        判定准则如下：

        1. 若sequence_match或cosine≥98%，自动判为“高风险”，is_plagiarism为true，无需推理和解释。
        2. 若两者均<50%，直接判为“低风险”，is_plagiarism为false。
        3. 若50%≤相似度<98%，或两个指标数值差异较大，请你像查重专家那样综合分析：结合表达方式、引用规范、内容重写、科研写作习惯、逻辑等，对“中风险”或“需补充引用”情形给出推理，并在"summary"字段用一句话说明风险来源与建议。
        4. 禁止使用“已规范引用”“规范引用”相关措辞。

        只输出如下JSON，无任何解释说明：
        {
        "plagiarism_risk": "...",
        "plagiarism_percent": 0-100,
        "confidence": 0.00-1.00,
        "is_plagiarism": true/false,
        "overlap_excerpt": ["..."],
        "summary": ""
        }
        '''
        },
        {"role": "user",
         "content":
            f"[用户段落]: {state['user_text']}\n"
            f"[最相似文献段落]: {ctx}\n"
            f"相似度：{similarities_desc}"}
    ]
    main_result = call_gpt(prompt)
    # Overwrite LLM "confidence" with our statistical value in JSON
    import json
    try:
        result_json = json.loads(main_result)
        result_json["confidence"] = round(plagiarism_confidence, 4)
        return {"main_analysis": json.dumps(result_json, ensure_ascii=False)}
    except Exception:
        return {"main_analysis": main_result}

def expert_node(state: RagState, expert_id: int):
    ctx = "\n".join(d.page_content for d in state["retrieved"])
    sim_seq = state.get("sim_seq", 0)
    sim_cosine = state.get("sim_cosine", 0)
    plagiarism_confidence = state.get("plagiarism_confidence", 0.0)
    similarities_desc = (f"字符级重合 sequence_match={sim_seq}%，"
                         f"语义相似 cosine={sim_cosine}%，"
                         f"置信度 confidence={plagiarism_confidence:.2f}")
    prompt = [
        {"role": "system",
        "content": f'''
        你是查重专家（Expert {expert_id}），判定标准如下：

        - sequence_match或cosine≥98%，直接判为“高风险”，is_plagiarism为true；
        - 两者均<50%，直接“低风险”，is_plagiarism为false；
        - 50%≤相似度<98% 或两指标判断分歧时，请像专家一样分析表达、引用、内容及专业背景，判为“中风险”或“需补充引用”时须在"summary"说明理由（如缺乏改写、部分引用不充分等）。
        - 禁止出现“已规范引用”等免责标签。

        始终输出与Main节点一致格式的JSON，"summary"字段用一句话阐明原因，无多余解释。
        '''
        },
        {"role": "user",
         "content":
            f"[用户段落]: {state['user_text']}\n"
            f"[Main分析]: {state['main_analysis']}\n"
            f"[相似文献段落]: {ctx}\n"
            f"相似度：{similarities_desc}"}
    ]
    fb = call_gpt(prompt)
    return {"expert_feedbacks": [f"Expert {expert_id}: {fb}"]}

def consensus_node(state: RagState):
    if state.get("revised") or not state.get("expert_feedbacks") or len(state["expert_feedbacks"]) < 2:
        return {}
    fb_text = "\n".join(
        m.content if hasattr(m, "content") else str(m)
        for m in state.get("expert_feedbacks", [])
    )
    sim_seq = state.get("sim_seq", 0)
    sim_cosine = state.get("sim_cosine", 0)
    plagiarism_confidence = state.get("plagiarism_confidence", 0.0)
    similarities_desc = (f"字符级重合 sequence_match={sim_seq}%，"
                         f"语义相似 cosine={sim_cosine}%，"
                         f"置信度 confidence={plagiarism_confidence:.2f}")
    prompt = [
        {"role": "system",
        "content": '''
        你是终审综合AI（Consensus），融合判决标准如下：

        1. 任一节点sequence_match或cosine≥98%，直接“高风险”，is_plagiarism为true；
        2. 两者均<70%，直接“低风险”，is_plagiarism为false；
        3. 其余情况，须根据表达、引用、内容归纳、专家建议综合推理风险等级，并用"summary"一行为理由说明（如“部分段落直接摘抄，建议重写”或“部分引用不充分，需补充说明”）。
        不能出现“已规范引用”等免责表述。只输出如下JSON，无其它说明：
        {
        "plagiarism_risk": "...",
        "plagiarism_percent": 0-100,
        "confidence": 0.00-1.00,
        "is_plagiarism": true/false,
        "overlap_excerpt": ["..."],
        "summary": ""
        }
        '''
        },
        {"role": "user",
         "content":
            f"[前一次结论]: {state['main_analysis']}\n"
            f"[专家意见]:\n{fb_text}\n"
            f"相似度: {similarities_desc}"}
    ]
    result = call_gpt(prompt, max_tokens=512)
    import json
    try:
        result_json = json.loads(result)
        result_json["confidence"] = round(plagiarism_confidence, 4)
        return {"main_analysis": json.dumps(result_json, ensure_ascii=False), "revised": True}
    except Exception:
        return {"main_analysis": result, "revised": True}

def judge_node(state: RagState):
    fb_text = "\n".join(msg.content for msg in state.get("expert_feedbacks", []))
    sim_seq = state.get("sim_seq", 0)
    sim_cosine = state.get("sim_cosine", 0)
    plagiarism_confidence = state.get("plagiarism_confidence", 0.0)
    similarities_desc = (f"字符级重合 sequence_match={sim_seq}%，"
                         f"语义相似 cosine={sim_cosine}%，"
                         f"置信度 confidence={plagiarism_confidence:.2f}")
    prompt = [
            {"role": "system",
    "content": '''
    你是中国高校学术诚信查重终审委员会Judge。

    规则如下：
    - sequence_match或cosine≥98%，直接“高风险”，is_plagiarism必须为true；
    - 两者均<70%，直接“低风险”，is_plagiarism为false；
    - 其余情况下，请针对50%≤相似度<98%或各项指标分歧时，像专家一样分析表达、引用、归纳等做判断，必须在"summary"字段补充一句理由或风险来源。

    严禁任何“已规范引用”表述。输出如下结构JSON，不得有多余说明：
    {
    "final_judgement": {
        "plagiarism_risk": "...",
        "plagiarism_percent": 0-100,
        "confidence": 0.00-1.00,
        "summary": ""
    },
    "similarity_metrics": {
        "sequence_match": "n%",
        "cosine": "n%"
    },
    "overlap_excerpt": "与文献重合最长片段",
    "expert_opinions": [
        {"expert": "Expert 1", "opinion": "..."},
        {"expert": "Expert 2", "opinion": "..."}
    ],
    "improvement_suggestions": [
        "...",
        "..."
    ],
    "conclusion": ""
    }
    '''
    },
        {"role": "user",
         "content":
            f"[用户段落]: {state['user_text']}\n"
            f"[最终Main分析]: {state['main_analysis']}\n"
            f"[专家反馈]: {fb_text}\n"
            f"相似度：{similarities_desc}"
        }
    ]
    result = call_gpt(prompt)
    import json
    try:
        result_json = json.loads(result)
        result_json["final_judgement"]["confidence"] = round(plagiarism_confidence, 4)
        return {"judge_output": json.dumps(result_json, ensure_ascii=False)}
    except Exception:
        return {"judge_output": result}

def build_plag_graph(vdb: FAISS):
    g = StateGraph(RagState)
    retrieve = g.add_node("retrieve", lambda s: retrieve_node(s, vdb))
    main = g.add_node("main", main_node)
    expert1 = g.add_node("expert1", lambda s: expert_node(s, 1))
    expert2 = g.add_node("expert2", lambda s: expert_node(s, 2))
    consensus = g.add_node("consensus", consensus_node)
    judge = g.add_node("judge", judge_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "main")
    g.add_edge("main", "expert1")
    g.add_edge("main", "expert2")
    g.add_edge("expert1", "consensus")
    g.add_edge("expert2", "consensus")
    g.add_edge("consensus", "judge")
    return g

def run_plagiarism_check(user_text: str, vector_db: FAISS):
    graph = build_plag_graph(vector_db)
    compiled_graph = graph.compile()
    initial_state = {"user_text": user_text, "revised": False}
    final_state: RagState = compiled_graph.invoke(initial_state)
    return final_state["judge_output"], final_state


# ------------ Drop-in Compatible API --------------
def cooperate_plagiarism_check(user_text: str, vector_db, embedding_model=None, *args, **kwargs):
    judge_json, state = run_plagiarism_check(user_text, vector_db)
    import json
    
    try:
        judge_parsed = json.loads(judge_json)
    except Exception:
        judge_parsed = {"raw_output": judge_json}
    # print(type(judge_parsed))
    # print(type(state.get('expert_feedbacks', [])))
    # print(type(state.get("plagiarism_confidence", None)))
    # print(type(state.get("plagiarism_percent", None)))
    # print(type(state.get("plagiarism_snippet", [])))
    # print(type(state.get("sim_seq", None)))
    # print(type(state.get("sim_cosine", None)))

    def map_plagiarism_risk_to_verdict(plagiarism_risk, summary=""):
        if plagiarism_risk is None:
            return {"result": "UNKNOWN", "reason": "未能判定相似度风险"}
        if plagiarism_risk == "低风险":
            return {"result": "ACCEPT", "reason": summary or "文本相似度低，无明显抄袭风险。"}
        # High, mid, need citation all = "ABSTAIN"
        return {"result": "ABSTAIN", "reason": summary or f"因判定为{plagiarism_risk}，建议补充引用、重写、或复查。"}
    
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

    result = {
        "judge_output": judge_parsed,
        "main_analysis": state.get('main_analysis', ''),
        "feedbacks": transfer_numpy_to_float(state.get('expert_feedbacks', [])),
        "plagiarism_confidence": round(float(round(state.get("plagiarism_confidence", None) * 100, 2))),
        "plagiarism_percentage": round(float(round(state.get("plagiarism_percent", None), 2))),
        "plagiarism_snippet": transfer_numpy_to_float(state.get("plagiarism_snippet", [])),
        "similarity_metrics": {
            "sequence_match": round(float(round(state.get("sim_seq", None), 2))),
            "cosine": round(float(round(state.get("sim_cosine", None), 2)))
        },
    }
    
    judge = result["judge_output"].get("final_judgement", {})
    risk = judge.get("plagiarism_risk")
    summary = judge.get("summary", "")
    result["verdict"] = map_plagiarism_risk_to_verdict(risk, summary)
    return result
# import os
# import re
# import gc
# import torch
# import torch.nn.functional as F
# from difflib import SequenceMatcher
# from typing import List
# import numpy

# from dotenv import load_dotenv

# # For the pipeline
# from sentence_transformers import CrossEncoder
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.schema import Document
# # from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS

# # For AI detection (optional)
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # For calling GPT-4o
# from openai import OpenAI

# device = torch.device("cuda:0")
# torch.cuda.set_device(0)
# print(f"Using device: {device}")

# # ========== DEMO AI Detector Model Setup ==========
# detector_model_name = "roberta-base-openai-detector"
# det_tokenizer = AutoTokenizer.from_pretrained(detector_model_name)
# det_model = AutoModelForSequenceClassification.from_pretrained(
#     detector_model_name).to(device)
# det_model.eval()

# # Using your GPT-4o model (replace with your actual key as needed)
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))


# # ===============================================================
# # 1) OpenAI-based text generation function (using gpt-4o)
# # ===============================================================
# def generate_with_openai_api(prompt: str,
#                              max_tokens: int = 512,
#                              temperature: float = 0.7,
#                              top_p: float = 0.9,
#                              frequency_penalty: float = 0.0,
#                              presence_penalty: float = 0.0) -> str:
#     """
#     Sends 'prompt' to the GPT-4o model via the openai API client.
#     Returns the model-generated string.
#     """
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4.1",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a helpful assistant."
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 },
#             ],
#             max_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             frequency_penalty=frequency_penalty,
#             presence_penalty=presence_penalty)
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         print("OpenAI API Error:", e)
#         return "【API Error】"


# # ===============================================================
# # 2) AI Content Detection Helper (OPTIONAL)
# # ===============================================================
# def compute_ai_generated_probability(text: str) -> float:
#     """
#     Roughly estimate how likely a text is AI-generated using 'roberta-base-openai-detector'.
#     Returns a float in [0,1], where 1 means "very likely AI-generated".
#     """
#     max_length = 512  # cutoff
#     if len(text) > max_length * 4:
#         text = text[:max_length * 4]

#     inputs = det_tokenizer(text,
#                            truncation=True,
#                            padding=True,
#                            return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = det_model(**inputs)
#         logits = outputs.logits
#         probs = torch.softmax(logits, dim=1)

#     # index 0 => "Real/Human", index 1 => "Fake/AI"
#     ai_prob = probs[0][1].item()
#     return ai_prob


# # ===============================================================
# # 3) Similarity Helpers
# # ===============================================================
# def split_into_paragraphs(text: str):
#     paragraphs = []
#     for blk in text.split("\n\n"):
#         blk = blk.strip()
#         if blk:
#             paragraphs.append(blk)
#     return paragraphs


# # Updated on 04/22/2025
# def split_into_sentences_chinese(text):
#     pattern = re.compile(r'([^。！？]*[。！？])', re.U)
#     sentences = pattern.findall(text)
#     left = pattern.sub('', text)
#     if left:
#         sentences.append(left)
#     return [s for s in sentences if s.strip()]


# # Updated on 04/22/2025


# def highlight_matches(original: str, submitted: str, min_len=4):
#     """
#     Return matched segments (>= min_len) found in `submitted`
#     that also appear in `original`.
#     """
#     matcher = SequenceMatcher(None, original, submitted)
#     matched_segments = []
#     for match in matcher.get_matching_blocks():
#         if match.size >= min_len:
#             segment = submitted[match.b:match.b + match.size]
#             matched_segments.append(segment)
#     return matched_segments


# def compute_embedding_similarity(embedding_model, text1, text2):
#     emb1 = embedding_model.embed_query(text1)
#     emb2 = embedding_model.embed_query(text2)
#     t1 = torch.tensor(emb1).to(device)
#     t2 = torch.tensor(emb2).to(device)
#     return F.cosine_similarity(t1, t2, dim=0).item()


# # ===============================================================
# # 4) Build a FAISS Vector Store from source docs
# # ===============================================================
# def build_paraphrase_vector_db(dirs_list: List[str]):
#     """
#     Build a FAISS vector store from ALL .txt files in each directory.
#     Each file is split into paragraphs.
#     """
#     all_paragraph_docs = []
#     file_paths = []
#     total_file_count = 0

#     embedding_model = HuggingFaceEmbeddings(
#         model_name="intfloat/multilingual-e5-base",
#         model_kwargs={"device": "cuda:0"})

#     for pages_dir in dirs_list:
#         processed_files = 0
#         txt_files = sorted(
#             [f for f in os.listdir(pages_dir) if f.endswith(".txt")])

#         # Use all files instead of just limiting to files_per_folder
#         for filename in txt_files:
#             file_path = os.path.join(pages_dir, filename)
#             file_paths.append(file_path)

#             with open(file_path, "r", encoding="utf-8") as f:
#                 text = f.read()
#                 paragraphs = split_into_paragraphs(text)
#                 for para in paragraphs:
#                     doc = Document(page_content=para,
#                                    metadata={"file_path": file_path})
#                     all_paragraph_docs.append(doc)

#             processed_files += 1
#             total_file_count += 1

#         print(f"Loaded {processed_files} .txt files from {pages_dir}")

#     print(
#         f"Total source files loaded: {total_file_count}, paragraphs: {len(all_paragraph_docs)}"
#     )

#     vector_db = FAISS.from_documents(all_paragraph_docs, embedding_model)
#     return vector_db, embedding_model, file_paths


# # ===============================================================
# # 5) CrossEncoder for re-ranking
# # ===============================================================
# cross_encoder_model = "BAAI/bge-reranker-v2-m3"
# cross_encoder = CrossEncoder(cross_encoder_model, device="cuda:0")


# # ===============================================================
# # 6) The main pipeline
# # ===============================================================
# def cooperate_plagiarism_check(user_text: str,
#                                vector_db: FAISS,
#                                embedding_model,
#                                exclude_file_path: str = None,
#                                initial_k=10,
#                                rerank_top_k=3):
#     """
#     1) Similarity Search in FAISS
#     2) Rerank with CrossEncoder
#     3) 'Main' LLM analysis
#     4) 'Expert' feedback x2
#     5) 'Judge' verdict => "ACCEPT" or "ABSTAIN"

#     Returns a dict including judge_output, etc.
#     """
#     # (1) Similarity search
#     retrieved_initial = vector_db.similarity_search(user_text, k=initial_k)

#     # Exclude same file (avoid self-match)
#     if exclude_file_path is not None:
#         retrieved_initial = [
#             d for d in retrieved_initial
#             if d.metadata.get("file_path", "") != exclude_file_path
#         ]

#     # (2) Rerank with CrossEncoder
#     pairs = [(user_text, d.page_content) for d in retrieved_initial]
#     cross_scores = cross_encoder.predict(pairs)
#     docs_scores = list(zip(retrieved_initial, cross_scores))
#     docs_scores.sort(key=lambda x: x[1], reverse=True)
#     top_docs = docs_scores[:rerank_top_k]

#     doc_info_list = []
#     for doc, sc in top_docs:
#         emb_sim = compute_embedding_similarity(embedding_model, user_text,
#                                                doc.page_content)
#         overlaps = highlight_matches(doc.page_content, user_text)
#         doc_info_list.append({
#             "content": doc.page_content,
#             "cross_score": sc,
#             "embedding_sim": emb_sim,
#             "overlaps": overlaps,
#             "file_path": doc.metadata.get("file_path", "")
#         })

#     top_texts = [info["content"] for info in doc_info_list]

#     # (3) Main model
#     prompt_main = ("你是主要分析模型 (Main Model)。以下是用户的段落，以及最相似的文献段落。\n"
#                    "检查是否有文字重叠，是否存在抄袭倾向。\n\n"
#                    f"[用户段落]:\n{user_text}\n\n"
#                    f"[最相似文献段落Top {rerank_top_k}]:\n" + "\n".join(top_texts) +
#                    "\n\n请用中文回答，并简要说明你的想法(1~3句话)。")
#     main_analysis = generate_with_openai_api(prompt_main)

#     # (4) Expert 1
#     prompt_fb1 = ("你是检查专家(Expert 1)，请审阅以下资料：\n"
#                   f"[用户段落]: {user_text}\n"
#                   f"[Main Model 分析]: {main_analysis}\n"
#                   f"[相似文献Top {rerank_top_k}]:\n" + "\n".join(top_texts) +
#                   "\n\n请简短指出Main Model的分析是否可信，是否遗漏了什么。\n")
#     fb1 = generate_with_openai_api(prompt_fb1)

#     # (5) Expert 2
#     prompt_fb2 = ("你是检查专家(Expert 2)，请审阅以下资料：\n"
#                   f"[用户段落]: {user_text}\n"
#                   f"[Main Model 分析]: {main_analysis}\n"
#                   f"[相似文献Top {rerank_top_k}]:\n" + "\n".join(top_texts) +
#                   "\n\n请简短指出Main Model的分析是否可信，是否遗漏了什么。\n")
#     fb2 = generate_with_openai_api(prompt_fb2)

#     # Combine feedbacks
#     feedbacks = [f"Expert 1 Feedback: {fb1}", f"Expert 2 Feedback: {fb2}"]

#     # (6) Judge
#     judge_prompt = f"""
# 你是评审模型(Judge)。以下是资料：
# [用户段落]: {user_text}
# [Main Model 分析]: {main_analysis}
# {feedbacks[0]}
# {feedbacks[1]}

# 请你判断，是否应该"ABSTAIN"(因为怀疑有抄袭或不确定)或"ACCEPT"。
# 请用JSON格式回答:
# {{
#   "verdict": "...",
#   "reason": "..."
# }}

# "verdict"只能是"ACCEPT"或"ABSTAIN"。
# "reason"用中文简述根据。
#     """
#     judge_output = generate_with_openai_api(judge_prompt)

#     # (7) Average cross-score
#     if len(top_docs) > 0:
#         avg_confidence = sum([sc for _, sc in top_docs]) / len(top_docs)
#     else:
#         avg_confidence = 0.0

#     # Extract plagiarism percentage based on verdict and scores
#     verdict_match = re.search(r'"verdict"\s*:\s*"(\w+)"', judge_output)
#     verdict = verdict_match.group(1) if verdict_match else "UNKNOWN"

#     # Calculate plagiarism percentage
#     # Higher if ABSTAIN, lower if ACCEPT
#     if verdict == "ABSTAIN":
#         # Higher plagiarism score for ABSTAIN
#         plagiarism_percentage = 50 + (avg_confidence * 50)  # Scale to 50-100%
#     else:
#         # Lower plagiarism score for ACCEPT
#         plagiarism_percentage = avg_confidence * 40  # Scale to 0-40%

#     # Ensure within 0-100 range
#     plagiarism_percentage = max(0, min(100, plagiarism_percentage))

#     # === Build snippet(s) from EVERY overlap =========================
#     all_overlaps = []
#     for info in doc_info_list:  # gather *all* matches
#         all_overlaps.extend(info["overlaps"])

#     all_overlaps = list(dict.fromkeys(all_overlaps))  # deduplicate, keep order

#     # (A) complete concatenation for the UI
#     # plagiarism_snippet_full = " ".join(all_overlaps)
#     plagiarism_snippet_full = all_overlaps

#     # (B) longest chunk – keeps old behaviour for backward‑compat
#     plagiarism_snippet = max(all_overlaps, key=len) if all_overlaps else ""

#     # fallback if we somehow had no overlaps at all
#     # if not plagiarism_snippet_full and doc_info_list:
#     #     snippet_len = len(doc_info_list[0]["content"])
#     #     plagiarism_snippet_full = doc_info_list[0][
#     #         "content"][:snippet_len] + "…"
#     original_snippet = plagiarism_snippet_full

#     # ----- GPT-4.1 Sentence-level Highlighting -----
#     user_sentences = split_into_sentences_chinese(user_text)
#     plag_highlight_prompt = (
#         "你是学术剽窃检测AI。下面是用户逐句分割的段落，以及最相似的参考文献段落。\n"
#         "请你判断哪些用户句子涉嫌剽窃，请只用[PLAG]...[/PLAG]包裹这些句子输出原文（不必解释，保持原句原格式，剩下句子不要输出）。\n"
#         "如果没有任何句子涉嫌剽窃，则只输出空字符串。\n\n"
#         "【用户分句】:\n" +
#         "\n".join([f"{i+1}. {s}" for i, s in enumerate(user_sentences)]) +
#         "\n\n【最相似文献段落】:\n" + "\n".join(top_texts))
#     gpt4_plag_sentence_mark = generate_with_openai_api(plag_highlight_prompt,
#                                                        max_tokens=256)
#     plagiarized_sents = re.findall(r'\[PLAG\](.+?)\[/PLAG\]',
#                                    gpt4_plag_sentence_mark,
#                                    flags=re.S)

#     def highlight_sentences(user_sentences, plag_sents):
#         result = []
#         for s in user_sentences:
#             if s.strip() in [ps.strip() for ps in plag_sents]:
#                 result.append(f'<mark style="background:orange">{s}</mark>')
#             else:
#                 result.append(s)
#         return ''.join(result)

#     gpt_sentence_highlight_html = highlight_sentences(user_sentences,
#                                                       plagiarized_sents)
#     # gpt4_1_snippet = " ".join(plagiarized_sents)
#     gpt4_1_snippet = plagiarized_sents

#     # updated on 04/23/2025
#     # ----- SMART: PICK CLOSER TO PERCENTAGE -----
#     def snippet_score_ratio(snippet, user_text):
#         if not snippet or not user_text:
#             return 0.0
#         return len(snippet) / max(1, len(user_text))

#     ratio_original = snippet_score_ratio(original_snippet, user_text)
#     ratio_gpt41 = snippet_score_ratio(gpt4_1_snippet, user_text)
#     target_ratio = plagiarism_percentage / 100

#     if abs(ratio_original - target_ratio) < abs(ratio_gpt41 - target_ratio):
#         plagiarism_snippet = original_snippet
#     else:
#         plagiarism_snippet = gpt4_1_snippet
#     # updated on 04/23/2025
#     # ✅ Forcefully return 0% and empty snippet if judged as human
#     if verdict == "ACCEPT":
#         plagiarism_percentage = 0.0
#         plagiarism_snippet = []

#     gc.collect()
#     torch.cuda.empty_cache()

#     def transfer_numpy_to_float(data):
#         if isinstance(data, dict):
#             return {k: transfer_numpy_to_float(v) for k, v in data.items()}
#         elif isinstance(data, list):
#             return [transfer_numpy_to_float(item) for item in data]
#         elif isinstance(data, (numpy.integer, )):
#             return int(data)
#         elif isinstance(data, (numpy.floating, )):
#             return float(data)
#         elif isinstance(data, numpy.ndarray):
#             return data.tolist()
#         else:
#             return data

#     return {
#         # list 中如果有非 float型態 (如 numpy)，轉換成json時會報錯，這邊使用 transfer_numpy_to_float 轉換
#         "top_docs_info": transfer_numpy_to_float(doc_info_list),
#         "main_analysis": main_analysis,
#         "feedbacks": transfer_numpy_to_float(feedbacks),
#         "judge_output": judge_output,
#         "plagiarism_confidence": round(float(round(avg_confidence * 100, 2))),
#         "plagiarism_percentage": round(float(round(plagiarism_percentage, 2))),
#         "plagiarism_snippet": plagiarism_snippet,
#         "verdict": verdict
#     }
