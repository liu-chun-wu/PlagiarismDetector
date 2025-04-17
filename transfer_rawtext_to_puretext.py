import json
import os

# 原始文字
original_text = """
本研究分析市面上常見攝影廠牌(Hikvision、Geovision、Dahwa、AXIS、LILIN、VivoTek)輸出的H.264碼流，利用Wireshark和MP4 Reader觀察串流封包資料如何影響MP4封裝的Meta Data，找出無法播放錄影檔的封裝特徵，並與正常錄影檔的封裝結構比較，以利後續開發修復程式。

論文分五章：第二章探討影像串流協定堆疊架構、H.264視訊壓縮標準及MP4檔案格式。第三章闡述研究步驟、網路封包資料與MP4封裝資料的關聯性，以及正常和異常封裝的差異。第四章分析moov box解析方式、修正情境及修正程序。第五章總結研究結論、限制和貢獻。
"""

# 轉換成 JSON 格式的字串（帶 \n 換行符號等）
json_ready_text = json.dumps(original_text, ensure_ascii=False, indent=2)

# 檔案路徑
file_path = "/home/undergrad/PlagiarismDetector/output.txt"

# 如果檔案存在，先刪除
if os.path.exists(file_path):
    os.remove(file_path)

# 寫入新的內容
with open(file_path, "w", encoding="utf-8") as f:
    f.write(json_ready_text)

print("已成功寫入 output.txt")
