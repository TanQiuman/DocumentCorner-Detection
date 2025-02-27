import os

def count_jpg_images_in_subfolders(root_folder):
    jpg_count_by_folder = {}
    total_jpg_count = 0  # 初始化总计数器

    # 遍历根目录及其子目录
    for subdir, dirs, files in os.walk(root_folder):
        # 统计当前子目录下的所有.jpg文件
        jpg_files = [f for f in files if f.lower().endswith('.jpg')]
        if jpg_files:
            jpg_count = len(jpg_files)
            jpg_count_by_folder[subdir] = jpg_count
            total_jpg_count += jpg_count  # 累加到总数

    return jpg_count_by_folder, total_jpg_count

# 示例：调用函数并打印结果
root_folder = 'E:\胡文胜-毕业资料整理\数据集\MDLDataSet\拍摄原片留存\第三版留存'  # 替换为你的文件夹路径
count_by_folder, total_count = count_jpg_images_in_subfolders(root_folder)

# 输出每个子文件夹的统计结果
for folder, count in count_by_folder.items():
    print(f"文件夹 {folder} 包含 {count} 张 .jpg 图片。")

# 输出所有子文件夹中的 .jpg 图片总数
print(f"所有子文件夹中 .jpg 图片的总数是 {total_count} 张。")
