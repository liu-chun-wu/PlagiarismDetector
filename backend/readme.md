# 後端說明

## flask_api

- 使用 flask 作為後端框架 處理傳遞進來的 txt 或是 pdf ，並回傳處理結果

## check_rephrase_content

### 輸入

- txt 的 文本

### 輸出

- "top_docs_info": 長度為3一維陣列，包含相似度最高的三個來源文本
- "main_analysis":
- "feedbacks": transfer_numpy_to_float(feedbacks),
- "judge_output": judge_output,
- "avg_confidence": round(float(round(avg_confidence * 100, 2))),
- "plagiarism_percentage": round(float(round(plagiarism_percentage, 2))),
- "plagiarism_snippet": plagiarism_snippet,
- "verdict": verdict
