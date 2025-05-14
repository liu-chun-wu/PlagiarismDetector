import os
import zipfile
import shutil

# 設定來源資料夾和解壓縮目標資料夾
source_folder = ""  # 修改為你的資料夾路徑
destination_folder = ""  # 修改為解壓縮後的存放資料夾

# 確保目標資料夾存在
os.makedirs(destination_folder, exist_ok=True)

# 遍歷資料夾內的所有檔案
for file in os.listdir(source_folder):
    if file.endswith(".zip"):  # 確保是 ZIP 檔案
        zip_path = os.path.join(source_folder, file)
        extract_path = os.path.join(destination_folder, file[:-4])  # 以 ZIP 檔名為資料夾名稱

        # 確保解壓縮的資料夾存在
        os.makedirs(extract_path, exist_ok=True)

        # 解壓縮 ZIP 檔案
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print(f"解壓縮完成：{file} -> {extract_path}")

        # 移動解壓縮資料夾內的 PDF 檔案到 destination_folder
        for root, _, files in os.walk(extract_path):
            for pdf_file in files:
                if pdf_file.endswith(".pdf"):
                    pdf_source_path = os.path.join(root, pdf_file)
                    pdf_destination_path = os.path.join(destination_folder, pdf_file)

                    # 如果目標資料夾已存在同名 PDF，直接加數字（不加底線）
                    base, ext = os.path.splitext(pdf_file)
                    counter = 1
                    while os.path.exists(pdf_destination_path):
                        pdf_destination_path = os.path.join(destination_folder, f"{base}{counter}{ext}")
                        counter += 1

                    shutil.move(pdf_source_path, pdf_destination_path)
                    print(f"移動 PDF 檔案：{pdf_source_path} -> {pdf_destination_path}")

print("所有 ZIP 檔案解壓縮並處理 PDF 完成！")
