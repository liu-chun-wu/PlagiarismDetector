# PDF Text Extractor

## 概述

PDF Text Extractor 是一個專門針對學術論文和文檔的文本提取工具。它使用 Google 的 Gemini API 來智能識別和提取 PDF 中的說明性文字段落，過濾掉公式、表格、圖表等非文本內容，並對段落進行智能處理，確保跨頁段落的連續性和正確性。

## 系統需求

- Python 3.7+
- PyMuPDF (fitz)
- Google Generative AI Python SDK
- Pillow (PIL)

## 安裝依賴

```bash
pip install PyMuPDF google-generativeai pillow python-dotenv
```

## 使用方法

### 基本使用

```python
from pdf_cutter import process_pdf

# 處理PDF文件
pdf_path = "your_document.pdf"
api_key = "your_gemini_api_key"
result = process_pdf(pdf_path, api_key)

# 查看結果
for page_idx, page_paragraphs in enumerate(result):
    if page_paragraphs:
        print(f"頁面 {page_idx}, 段落數: {len(page_paragraphs)}")
        for para_idx, paragraph in enumerate(page_paragraphs):
            print(f"  段落 {para_idx}: {paragraph[:100]}...")
```

### 進階選項

```python
# 使用文本模式分析 (不使用圖像)
result = process_pdf(pdf_path, api_key, use_image=False)

# 自定義短段落過濾長度
result = process_pdf(pdf_path, api_key, min_length=50)
```

## 返回結果

函數返回一個二維列表：
- 第一層索引對應 PDF 的頁碼 (從 0 開始)
- 第二層索引對應該頁上的段落編號
- 每個元素是提取的段落文本

```python
# 訪問第 3 頁的第 2 段文字
text = result[2][1]
```

## 工作原理

1. **頁面分析**：使用 Gemini API 分析每頁 PDF 內容，識別說明性文字段落
2. **初步提取**：使用正則表達式提取標記的段落
3. **段落過濾**：根據最小長度要求過濾段落
4. **後處理**：
   - 檢查跨頁段落連續性並合併
   - 處理以冒號結尾的段落及其後續列表項
   - 合併連續的列表項

## 注意事項

- 此工具需要 Google Gemini API 金鑰
- 處理大型 PDF 可能耗費較多時間和 API 配額
- 圖像模式分析提供更好的排版識別，但處理速度較慢

## 使用場景

- 學術論文文本提取
- 文本數據準備和預處理

## 限制

- 目前僅支持中英文文檔
- 表格內容無法保留原始結構
- 需要網絡連接以使用 Gemini API