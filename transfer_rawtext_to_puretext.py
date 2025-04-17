import json
import os

# 原始文字
original_text = """

本研究以目前專案所使用到之市面上常見之攝影廠牌（Hikvision、Geovision、Dahwa、 AXIS、LILIN、VivoTek）所輸出之H.264 碼流搭配Wireshark 以及MP4 Reader，在觀察串 流封包資料如何影響後續MP4 封裝時產生之Meta Data 後，找出不可播放錄影檔的封裝 特徵並嘗試與比較正常可播放之錄影檔封裝結構進行比較，以利後續實作出修復程式。 本論文共分為五個章節，第二章為文獻探討，其中針對影像串流協定的堆疊架構、 H.264 視訊壓縮標準以及MP4 檔案格式進行說明。第三章為系統架構，主要為研究步驟、 網路封包資料與MP4 封裝資料之關聯性、正常與有問題之封裝差異比較。第四章為實 驗分析與討論，包含moov box 內容解析方式、修正情境分析以及歸納出修正程序。第 五章為結論，包含研究結論、研究限制與貢獻。論文的架構如下圖所示： 緒論 文獻探討 系統架構 實驗分析與討論 結論 第二章 文獻探討 本章分四個部分，分別是串流協定堆疊架構、H.264 標準、MP4（MPEG-4 Part 14） 檔案格式以及MP4 Box Atom 資料簡化方式，以下將分各小節逐一說明。

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
