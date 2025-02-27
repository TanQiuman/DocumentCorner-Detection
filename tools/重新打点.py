import os

def delete_txt_files(folder_path):
    # 获取指定目录下的所有文件
    files = os.listdir(folder_path)

    # 遍历文件列表，删除所有 .txt 文件
    for file in files:
        if file.lower().endswith('.json'):  # 匹配所有 .txt 文件
            file_path = os.path.join(folder_path, file)
            try:
                os.remove(file_path)  # 删除文件
                print(f"已删除文件: {file_path}")
            except Exception as e:
                print(f"删除文件 {file_path} 时出错: {e}")


folder_path = r'E:\代码和相关论文\数据集\Tqm整理数据集\复杂背景\complex'  # 替换为你的文件夹路径
delete_txt_files(folder_path)
