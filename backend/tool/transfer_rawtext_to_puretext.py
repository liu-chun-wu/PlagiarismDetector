import json
import os

# 原始文字
original_text = """
以解構式行為理論所定義三個變數，分解出三個觀點內的兩個細項，三個前因支 援歸納出本研究的三個前因理論，再由分解出的變數去感受行為，最後再由感受到的 結果影響實際行為。 本研究主要以計畫行為理論(TPB)作為概念，將使用者在社群商務環境下對沉浸 因素分成三個觀點來探討：技術觀點之「感知互動性(Perceived Interactivity)」與「感 知安全性(Perceived Security)」、社會觀點之「評價評論(Recommendation & Referrals)」 與「推薦介紹(Rating & Reviews)」以及個人特質之「嚴謹性(Conscientiousness)」與「外 向性(Extraversion)」，探討消費者是否會有這三個因素影響「沉浸」，且是否能藉由沉 浸增加其「購買意圖」。因此，本研究提出以下的研究架構，如圖 3-1 所示。 探討社群商務沉浸對於購買意圖之影響：技術、社會化、個人特質之觀點       第三章  研究方法 圖 3 - 1 本研究之研究架構圖 技 術 感知互動性 感知安全性 社 會 化 評價評論 推薦介紹 個 人 特 質 嚴謹性 外向性 沉浸 購買意願 探討社群商務沉浸對於購買意圖之影響：技術、社會化、個人特質之觀點       第三章  研究方法

"""

# 轉換成 JSON 格式的字串（帶 \n 換行符號等）
json_ready_text = json.dumps(original_text, ensure_ascii=False, indent=2)

# 檔案路徑
file_path = "/home/undergrad/PlagiarismDetector/backend/tool/output.txt"

# 如果檔案存在，先刪除
if os.path.exists(file_path):
    os.remove(file_path)

# 寫入新的內容
with open(file_path, "w", encoding="utf-8") as f:
    f.write(json_ready_text)

print("已成功寫入 output.txt")
