# 檔案說明
- extract: 抓取每篇 paper 每個小章節前面一小段
- paraphrase_and_generate: 把 extract 抓取下來的小段落傳給 Gemini 進行改寫與生成
- unzip: 就只是把隔壁 paper_webcrawling 下載下來的都解壓縮

# 使用說明
- 填入 input、output 的 directory，paraphrase_and_generate 再加上 Gemini 的 API
- 在 paper_webcrawling 的環境再裝 PyMuPDF 就好
    ```pip install PyMuPDF```