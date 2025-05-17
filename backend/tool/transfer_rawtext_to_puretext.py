import json
import os

# 原始文字
original_text = """
過去深度學習應用於股票之研究已有很多，但大多應用僅止於預測股價，該類研 究衡量方式大多為統計量，如Chen、Zhou 及Dai (2015)以準確率(accuracy)衡量[4]、Di Persio 及Honchar (2016)以均方差(MSE;Mean square error)、Accuracy 衡量[7]、 Tsantekidis, A., et al.(2017)衡量股價變動方向的正確率，以Kohen’s kappa 衡量等 [20]，以上之研究多停留提升預測之精準度，較少人將研究拓展至將預測股價應用於實 際股票交易，並檢驗何種模型產生之投資績效較好。 Bai、Kolter 及Koltun (2018)曾對TCN 及LSTM、GRU(Gated recurrent units network)優劣做廣泛性的比較[20]，發現TCN 優於傳統LSTM、GRU，後來Pradhan 與 Longpre(2016)、Prakash et al.(2016)、Kim et al.(2017) 都提出了加上殘差連接(shortcut connection)的LSTM 網路[37] [38] [30]，但尚未有TCN 與加上殘差連接(residual connection)的LSTM 網路的比較。

"""

# 轉換成 JSON 格式的字串（帶 \n 換行符號等）
json_ready_text = json.dumps(original_text, ensure_ascii=False, indent=2)

# 檔案路徑
file_path = "/mnt/Agents4Financial/PlagiarismDetector/backend/tool/output.txt"

# 如果檔案存在，先刪除
if os.path.exists(file_path):
    os.remove(file_path)

# 寫入新的內容
with open(file_path, "w", encoding="utf-8") as f:
    f.write(json_ready_text)

print("已成功寫入 output.txt")
