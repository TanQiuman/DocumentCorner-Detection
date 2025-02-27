import os


def get_jpg_files(directory):
    """获取指定目录下所有 .jpg 文件名（不包括扩展名）。"""
    return {os.path.splitext(file)[0] for file in os.listdir(directory) if file.endswith('.jpg')}


def compare_jpg_filenames(dir1, dir2):
    """比较两个目录下的 .jpg 文件名是否一致。"""
    files_dir1 = get_jpg_files(dir1)
    files_dir2 = get_jpg_files(dir2)

    if files_dir1 == files_dir2:
        print("两个目录下的 .jpg 文件名一致。")
    else:
        print("两个目录下的 .jpg 文件名不一致。")
        print("仅存在于第一个目录的文件名:", files_dir1 - files_dir2)
        print("仅存在于第二个目录的文件名:", files_dir2 - files_dir1)


# 示例使用
dir1 = "D:\MyData\PythonProject\Higherhrnet-paper\data\coco\images\\val2017"  # 替换为第一个目录的路径
dir2 = "E:\XuyuanFiles-yuanshi\csqHelpProject\keypoint_rcnn_training_pytorch\Mpaper_data_my\Mpaper_data_my\Test\images"  # 替换为第二个目录的路径

compare_jpg_filenames(dir1, dir2)
