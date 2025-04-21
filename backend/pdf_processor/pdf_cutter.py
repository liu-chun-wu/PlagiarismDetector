import fitz  # PyMuPDF
from google import genai
from google.genai import types
import re
from PIL import Image
import io
import time
import os
from dotenv import load_dotenv


def extract_page_as_text(pdf_path, page_number):
    """提取 PDF 特定頁面的文本，保留段落和基本排版"""
    doc = fitz.open(pdf_path)

    if page_number < -0 or page_number >= len(doc):
        raise ValueError(f"頁面編號無效。PDF 僅有 {len(doc)} 頁")

    page = doc[page_number]
    text = page.get_text()
    doc.close()

    return text


def extract_page_as_image(pdf_path, page_number):
    """將 PDF 頁面轉換為圖像"""
    doc = fitz.open(pdf_path)

    if page_number < 0 or page_number >= len(doc):
        raise ValueError(f"頁面編號無效。PDF 僅有 {len(doc)} 頁")

    page = doc[page_number]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x放大以獲得更好的圖像品質

    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))

    doc.close()
    return img


def analyze_content_with_gemini(pdf_path,
                                page_number,
                                use_image=True,
                                api_key=None):
    """使用 Gemini 分析 PDF 頁面內容並進行段落分割"""

    # 設置模型
    client = genai.Client(api_key=api_key)

    # 準備提示
    prompt = """請分析這個 PDF 頁面的內容，但僅關注說明型文章的部分。

    請遵循以下篩選規則：
    1. 僅提取連續完整的說明性文字段落（如背景介紹、方法說明、分析討論、條列式說明等）
    2. 忽略以下內容類型：
    - 數學公式和推導過程
    - 目錄、索引和參考文獻
    - 表格內容和表格說明
    - 圖表標籤和圖表說明
    - 代碼片段
    - 作者資訊和單純的標題
    - 文件截圖
    3. 有些段落會因為換頁被分割在下一頁的上方，請不要忽略他們

    對於符合條件的說明型文字段落：
    1. 將每個段落標記為 "[[段落N]]"（N 是段落編號）
    2. 確保同個段落的內容被分割在同個段落
    3. 如果段落中嵌入了短公式，但段落主體是說明文字，則保留整個段落
    4. 條列式的說明內容請與標題保留在同個段落

    如果頁面上沒有符合條件的說明型文字，請回覆 "[[無符合條件的內容]]"
    """

    if use_image:
        # 使用圖像方式（保留完整排版和視覺元素）
        img = extract_page_as_image(pdf_path, page_number)

        # 使用 Gemini 的多模態功能分析內容
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[img],
            config=types.GenerateContentConfig(max_output_tokens=10000,
                                               temperature=0.5,
                                               system_instruction=prompt))

    else:
        # 使用文本方式（可能丟失一些排版信息）
        text = extract_page_as_text(pdf_path, page_number)

        # 添加文本到提示中
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[text],
            config=types.GenerateContentConfig(max_output_tokens=10000,
                                               temperature=0.5,
                                               system_instruction=prompt))
    time.sleep(2)
    return response.text


def filter_short_paragraphs(paragraphs, min_length=10):
    """過濾掉長度過短的段落，但始終保留第一和最後一個段落
    
    Args:
        paragraphs: 段落列表
        min_length: 最小字符數 (預設為10個字符)
        
    Returns:
        過濾後的段落列表
    """
    filtered_paragraphs = []

    # 如果沒有段落，直接返回空列表
    if not paragraphs:
        return filtered_paragraphs

    # 遍歷所有段落
    for i, paragraph in enumerate(paragraphs):
        # 移除空白字符後計算長度
        trimmed = paragraph.strip()

        # 如果是第一段或最後一段，無論長度都保留
        if i == 0 or i == len(paragraphs) - 1:
            filtered_paragraphs.append(paragraph)
            if len(trimmed) < min_length:
                print(
                    f"保留第{'一' if i == 0 else '最後一'}段 (長度: {len(trimmed)}): '{trimmed}'"
                )
        # 其他段落依據長度決定是否保留
        elif len(trimmed) >= min_length:
            filtered_paragraphs.append(paragraph)
        else:
            print(f"已過濾段落 {i+1} (長度: {len(trimmed)}): '{trimmed}'")

    return filtered_paragraphs


def extract_filtered_paragraphs(analysis_result):
    """從分析結果中提取篩選後的段落"""
    # 檢查是否有符合條件的內容
    if "[[無符合條件的內容]]" in analysis_result:
        return []

    # 提取段落
    paragraphs = re.split(r'\[\[段落\d+\]\]', analysis_result)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def check_paragraph_continuity(paragraph1, paragraph2, api_key):
    """檢查兩個段落連接起來是否合理"""
    # 設置模型
    client = genai.Client(api_key=api_key)

    # 準備提示
    prompt = """請判斷以下兩個段落是否是連續的內容，應該合併為同一段落。

段落1:
{paragraph1}

段落2:
{paragraph2}

判斷標準:
1. 段落1是否未完成的句子，段落2是否是段落1的延續
2. 段落1和段落2是否討論同一主題，且邏輯上緊密相連
3. 兩段文字合併後是否更通順、更完整

請只回覆 "是" 或 "否"，不要解釋原因。
"""

    # 填入段落
    filled_prompt = prompt.format(paragraph1=paragraph1, paragraph2=paragraph2)

    # 使用Gemini分析
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[filled_prompt],
        config=types.GenerateContentConfig(max_output_tokens=10,
                                           temperature=0.1))

    # 等待一下以避免API限制
    time.sleep(2)

    # 獲取回答並判斷
    answer = response.text.strip().lower()
    return "是" in answer


def is_bullet_or_numbered_list(text):
    """檢查文本是否以•或數字+點開頭"""
    # 去除可能的空白
    text = text.strip()

    # 檢查是否以•開頭
    if text.startswith('•'):
        return True

    # 檢查是否以數字+點開頭，允許項目符號前有空格
    number_pattern = r'^\s*\d+\.'
    if re.match(number_pattern, text):
        return True

    return False


def get_next_number(text):
    """從文本中提取第一個數字（用於判斷序號）"""
    number_match = re.match(r'^\s*(\d+)\.', text)
    if number_match:
        return int(number_match.group(1))
    return None


def post_process_paragraphs_from_list(paragraphs_list, api_key):
    """對段落列表進行二次處理
    
    Args:
        paragraphs_list: 二維列表，第一層表示頁碼，第二層表示頁內段落
        api_key: Gemini API 金鑰
        
    Returns:
        處理後的二維段落列表
    """
    print("開始段落後處理...")

    # 創建結果列表的副本，避免直接修改原始數據
    processed_results = [
        page_paragraphs.copy() if page_paragraphs else []
        for page_paragraphs in paragraphs_list
    ]

    # 1. 檢查每頁的最後一個段落與下一頁的第一個段落是否應連接
    for current_page in range(len(paragraphs_list) - 1):
        next_page = current_page + 1

        # 確保當前頁和下一頁都有段落
        if not paragraphs_list[current_page] or not paragraphs_list[next_page]:
            continue

        # 獲取當前頁的最後一個段落和下一頁的第一個段落
        last_paragraph = paragraphs_list[current_page][-1]
        first_paragraph = paragraphs_list[next_page][0]

        # 檢查是否應該合併
        if check_paragraph_continuity(last_paragraph, first_paragraph,
                                      api_key):
            print(f"合併頁面 {current_page} 的最後段落與頁面 {next_page} 的第一段落")

            # 合併段落
            merged_paragraph = last_paragraph.strip(
            ) + " " + first_paragraph.strip()

            # 更新結果
            processed_results[current_page][-1] = merged_paragraph
            processed_results[next_page].pop(0)  # 移除已合併的段落

    # 2. 處理冒號結尾的段落和列表項
    for page in range(len(processed_results)):
        if not processed_results[page]:
            continue

        i = 0
        while i < len(processed_results[page]):
            # 獲取當前段落
            current_text = processed_results[page][i].strip()

            # 檢查是否以冒號結尾（包括全形和半形冒號）
            if current_text.rstrip().endswith(
                    ':') or current_text.rstrip().endswith('：'):
                next_text = None
                next_page_idx = None
                is_next_page = False

                # 檢查是否有當前頁的下一個段落
                if i + 1 < len(processed_results[page]):
                    next_text = processed_results[page][i + 1].strip()
                    next_idx = i + 1
                    next_page_idx = page
                # 如果是該頁的最後一個段落，檢查下一頁的第一個段落
                elif page + 1 < len(processed_results) and processed_results[
                        page + 1]:
                    next_text = processed_results[page + 1][0].strip()
                    next_idx = 0
                    next_page_idx = page + 1
                    is_next_page = True

                # 如果找到了下一個段落
                if next_text is not None:
                    # 檢查下一段是否為列表項
                    if is_bullet_or_numbered_list(next_text):
                        print(f"發現冒號結尾段落後跟隨列表項: 頁 {page} 段落 {i}" +
                              (" (下一頁)" if is_next_page else ""))

                        # 合併段落
                        merged_content = current_text + "\n" + next_text
                        processed_results[page][i] = merged_content

                        # 刪除已合併的段落
                        processed_results[next_page_idx].pop(next_idx)

                        # 如果是在同一頁上繼續檢查後續列表項
                        if not is_next_page:
                            # 檢查是否還有更多列表項需要合併
                            continue_merging = True
                            current_number = get_next_number(next_text)

                            while continue_merging and i + 1 < len(
                                    processed_results[page]):
                                next_text = processed_results[page][i +
                                                                    1].strip()

                                # 檢查是否是連續的列表項
                                is_bullet = next_text.startswith('•')
                                next_number = get_next_number(next_text)

                                if (is_bullet and '•' in merged_content) or \
                                   (current_number is not None and next_number is not None and next_number == current_number + 1):
                                    # 繼續合併
                                    merged_content = processed_results[page][
                                        i] + "\n" + next_text
                                    processed_results[page][i] = merged_content

                                    # 更新當前數字
                                    current_number = next_number

                                    # 刪除已合併的段落
                                    processed_results[page].pop(i + 1)
                                else:
                                    continue_merging = False
                    else:
                        # 冒號結尾但下一段不是列表項，刪除該段
                        print(f"刪除冒號結尾但後面不是列表項的段落: 頁 {page} 段落 {i}")
                        processed_results[page].pop(i)
                        continue  # 不增加索引，因為當前索引已指向下一個段落
                else:
                    # 冒號結尾但沒有下一段，刪除該段
                    print(f"刪除冒號結尾且找不到後續段落的段落: 頁 {page} 段落 {i}")
                    processed_results[page].pop(i)
                    continue  # 不增加索引，因為當前索引已指向下一個段落

            # 進入下一個段落
            i += 1

    print("段落後處理完成")
    return processed_results


def process_pdf_page(pdf_path,
                     page_number,
                     use_image=True,
                     min_length=30,
                     api_key=None):
    """處理 PDF 頁面並使用 Gemini 進行分析"""
    print(f"正在處理 PDF: {pdf_path}, 頁面: {page_number}")

    # 分析內容
    analysis = analyze_content_with_gemini(pdf_path, page_number, use_image,
                                           api_key)

    # 提取段落
    paragraphs = extract_filtered_paragraphs(analysis)

    # 過濾段落
    filtered_paragraphs = filter_short_paragraphs(paragraphs, min_length)

    print(f"分析完成，共識別出 {len(filtered_paragraphs)} 個段落")

    return filtered_paragraphs


def process_pdf(pdf_path, api_key, min_length=30, use_image=True):
    """
    處理整個PDF檔案，提取其中的說明型文字段落
    
    Args:
        pdf_path: PDF文件路徑
        api_key: Google Gemini API金鑰
        min_length: 段落最小長度
        use_image: 是否使用圖像模式分析

    Returns:
        二維列表，第一層索引表示頁碼，第二層索引表示段落編號
    """
    # 獲取PDF頁數
    pdf_length = len(fitz.open(pdf_path))

    # 存儲結果
    results = [[] for _ in range(pdf_length)]

    # 處理每一頁
    for page_number in range(pdf_length):
        # 執行分析
        print(f"\n開始處理第{page_number}頁...")
        paragraphs = process_pdf_page(pdf_path,
                                      page_number,
                                      use_image=use_image,
                                      min_length=min_length,
                                      api_key=api_key)

        # 將段落內容存入結果
        results[page_number] = paragraphs

        # 顯示分析結果預覽
        if paragraphs:
            print(f"\n第{page_number}頁分析結果預覽：")
            preview = paragraphs[0][:300] + "..." if len(
                paragraphs[0]) > 300 else paragraphs[0]
            print(preview)

    # 執行段落後處理 (使用新的基於列表的處理功能)
    print("\n開始進行段落的二次處理...")
    processed_results = post_process_paragraphs_from_list(results, api_key)

    print("PDF處理完成！所有段落處理和優化已完成。")

    return processed_results


# 使用範例
if __name__ == "__main__":
    # 參數設定
    pdf_path = "/home/undergrad/PlagiarismDetector/backend/uploaded_pdfs/1.pdf"  # 替換為你的PDF路徑
    load_dotenv()
    api_key = os.getenv("GEMINI_APIKEY")
    # 替換為你的API金鑰

    # 處理PDF
    result = process_pdf(pdf_path, api_key)

    # 輸出結果示例
    for page_idx, page_paragraphs in enumerate(result):
        if page_paragraphs:
            print(f"頁面 {page_idx}, 段落數: {len(page_paragraphs)}")
            for para_idx, paragraph in enumerate(page_paragraphs):
                preview = paragraph[:100] + "..." if len(
                    paragraph) > 100 else paragraph
                print(f"  段落 {para_idx}: {preview}")
