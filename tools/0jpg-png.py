from PIL import Image
import os

def convert_png_to_jpg(png_image_path):
    # 打开 PNG 图片
    img = Image.open(png_image_path)

    # 生成新的文件名，替换后缀为 .jpg
    jpg_image_path = os.path.splitext(png_image_path)[0] + '.jpg'

    # 将图片转换为 RGB 模式（PNG 是 RGBA 模式，而 JPG 不能包含透明度）
    img = img.convert('RGB')

    # 保存为 JPG 格式
    img.save(jpg_image_path, 'JPEG')

    # 删除原始的 PNG 文件
    os.remove(png_image_path)

    print(f"已将 {png_image_path} 转换为 {jpg_image_path} 并删除原文件。")
    return jpg_image_path

def convert_folder_png_to_jpg(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 筛选出所有的 .png 文件
    png_files = [f for f in files if f.lower().endswith('.png')]

    # 对每个 .png 文件进行转换
    for png_file in png_files:
        png_image_path = os.path.join(folder_path, png_file)
        convert_png_to_jpg(png_image_path)

# 示例用法
folder_path = r'E:\代码和相关论文\数据集\Tqm整理数据集\简单纸张干扰\complexfuse'  # 替换为你的文件夹路径
convert_folder_png_to_jpg(folder_path)
