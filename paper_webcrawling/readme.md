# Setting
1. 新增gemini API key
    - https://aistudio.google.com/app/apikey?hl=zh-tw 
2. 申請論文網站帳號
    - https://ndltd.ncl.edu.tw/ 
3. 在根目錄新增.env
    - 新增 USERNAME=論文網站帳號
    - 新增 PASSWORD=論文網站密碼
    - 新增 GEMINI_API_KEY=你的gemini API key
4. 建一個新個conda 環境    
5. 使用 environment.yml 安裝套件，在新的環境下指令 conda env create --name my_env -f environment.yml
    - my_env 用你的新環境名稱取代掉
6. 如果 chromedriver 和你當前的 chrome 的版本不一樣會出錯，要去以下網址找
    - https://googlechromelabs.github.io/chrome-for-testing/
7. 修改下載的路徑，windows需要絕對路徑
    - download_directory = "D:\\PlagiarismDetector\\paper_webcrawling\\paper" 
# Using
- web_crawl(mode: str, keywords: list) => 下載檔案到paper/
- 簡易查詢
    - 只有一個關鍵字
    ```python=
    keywords = ["囤房稅"]
    web_crawl(mode="basic", keywords=keywords)
    ```
- 進階查詢
    - 參數依照以下規則
    - 第一個關鍵字,第一個欄位名稱,邏輯,第二個關鍵字,第二個欄位名稱,學門(只有這個可以空，其他都不能空)
    ```python=
    keywords = ["中央大學", "校院名稱", "and", "2021", "論文出版年", ""]
    web_crawl(mode="advance", keywords=keywords)
    ```
