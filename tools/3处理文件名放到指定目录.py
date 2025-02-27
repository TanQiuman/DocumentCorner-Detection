import os
import shutil
import json

def rename_and_move_files(src_folder, img_dest_folder, json_dest_folder, start_num=1):
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(img_dest_folder, exist_ok=True)
    os.makedirs(json_dest_folder, exist_ok=True)

    # 获取源文件夹中的所有jpg文件
    files = sorted([f for f in os.listdir(src_folder) if f.endswith('.jpg')])

    if not files:
        print("没有找到任何jpg文件!")
        return

    # 检查jpg和json文件是否一一对应
    for i, img in enumerate(files, start=start_num):
        json_file = img.replace('.jpg', '.json')

        img_path = os.path.join(src_folder, img)
        json_path = os.path.join(src_folder, json_file)

        if not os.path.exists(json_path):
            print(f"Warning: 找不到 {img} 对应的 {json_file} 文件")
            continue  # 如果没有对应的json文件，就跳过

        # 创建新的文件名
        new_name = f"{i}.jpg"
        new_json_name = f"{i}.json"

        # 新文件路径
        new_img_path = os.path.join(img_dest_folder, new_name)
        new_json_path = os.path.join(json_dest_folder, new_json_name)

        # 移动并重命名文件
        try:
            shutil.copy(img_path, new_img_path)
            shutil.copy(json_path, new_json_path)
            print(f"文件已成功移动: {img} -> {new_name}, {json_file} -> {new_json_name}")
            # 读取新的json文件内容
            with open(new_json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # 更新json中的imagePath字段
            json_data['imagePath'] = new_name

            # 将更新后的内容写回新的json文件
            with open(new_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"移动文件时发生错误: {e}")

# 使用示例
img_folder = r'E:\代码和相关论文\数据集\Tqm整理数据集\重叠复杂纸张\simplecover'  # 图片文件夹路径
img_dest_folder = r'E:\coco\images\val2017'   # 目标图片文件夹路径
json_dest_folder = r'E:\coco\images\val_annotations'  # 目标json文件夹路径

rename_and_move_files(img_folder, img_dest_folder, json_dest_folder, start_num=4226)
