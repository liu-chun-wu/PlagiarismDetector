import json
import os

# 檔案路徑設定
input_path = "/mnt/Agents4Financial/PlagiarismDetector/backend/tool/input.txt"
output_path = "/mnt/Agents4Financial/PlagiarismDetector/backend/tool/output.txt"

# 讀取 input.txt 的內容
with open(input_path, "r", encoding="utf-8") as f:
    original_text = f.read()

# 轉換成 JSON 格式的字串（帶 \n 換行符號等）
json_ready_text = json.dumps(original_text, ensure_ascii=False, indent=2)

# 如果 output.txt 存在，先刪除
if os.path.exists(output_path):
    os.remove(output_path)

# 寫入新的內容到 output.txt
with open(output_path, "w", encoding="utf-8") as f:
    f.write(json_ready_text)

print("已成功從 input.txt 讀取並寫入 output.txt")
