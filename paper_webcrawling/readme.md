# Setting
1. 新增gemini API key
    - https://aistudio.google.com/app/apikey?hl=zh-tw 
2. 申請論文網站帳號
    - https://ndltd.ncl.edu.tw/ 
3. 在根目錄新增.env
    - 新增 USERNAME=論文網站帳號
    - 新增 PASSWORD=論文網站密碼
    - 新增 USERNAME=你的gemini API key
4. 建一個新個conda 環境    
5. 使用 requirement.txt 安裝套件
6. 如果 chromedriver 和你當前的 chrome 的版本不一樣會出錯，要去以下網址找
    - https://googlechromelabs.github.io/chrome-for-testing/
# Using
- 呼叫 web_crawl(mode: str, keyword: str)
    1. mode: 
        - 'basic' : 簡易查詢
        - 'advance' : 進階查詢(還沒完成)
    2. keyword: 關鍵字
- 下載的 .zip 或是 .pdf 會在 paper/