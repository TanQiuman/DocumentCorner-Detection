import os


def find_unpaired_files(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 分离出.jpg文件和.json文件
    jpg_files = {f for f in files if f.endswith('.jpg')}
    json_files = {f for f in files if f.endswith('.json')}

    # 找出没有成对的.jpg文件
    unpaired_jpg = jpg_files - {f.replace('.json', '.jpg') for f in json_files}

    # 找出没有成对的.json文件
    unpaired_json = json_files - {f.replace('.jpg', '.json') for f in jpg_files}

    return unpaired_jpg, unpaired_json


# 示例用法
folder_path = r'E:\example\simplecover'  # 替换为你的文件夹路径
unpaired_jpg, unpaired_json = find_unpaired_files(folder_path)

print("没有成对的.jpg文件:", unpaired_jpg)
print("没有成对的.json文件:", unpaired_json)