import json
import os

save_dir = "/home/undergrad/PlagiarismDetector/backend/uploaded_pdfs"

os.makedirs(save_dir, exist_ok=True)  # 確保資料夾存在
load_path = os.path.join(save_dir, "array_data.json")

# 讀取 JSON 並還原成二維陣列
with open(load_path, 'r', encoding='utf-8') as f:
    loaded_array = json.load(f)

paragraph_count = 0
for page_idx, page_paragraphs in enumerate(loaded_array):
    if page_paragraphs:
        for para_idx, paragraph in enumerate(page_paragraphs):
            paragraph_count += 1
print(f"總共有 {paragraph_count} 個段落")
