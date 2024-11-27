import os
import shutil

def copy_first_200_files(source_image_folder, source_text_folder, target_image_folder, target_text_folder, max_files=300):
    # 检查目标文件夹是否存在，不存在则创建
    if not os.path.exists(target_image_folder):
        os.makedirs(target_image_folder)
    if not os.path.exists(target_text_folder):
        os.makedirs(target_text_folder)
    
    # 获取源图片文件夹中所有图片文件的文件名
    all_image_files = os.listdir(source_image_folder)
    image_files = [f for f in all_image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # 按 "frame数字_" 的数字部分排序
    def extract_number(file_name):
        try:
            # 从文件名中提取数字部分，比如 "frame123_.jpg" -> 123
            return int(file_name.split('frame')[1].split('_')[0])
        except (IndexError, ValueError):
            return float('inf')  # 如果文件名不符合规则，放在最后

    image_files.sort(key=extract_number)
    
    # 复制前 max_files 张图片和对应的标注文件
    for i, image_file in enumerate(image_files[:max_files]):
        # 复制图片文件
        image_source_path = os.path.join(source_image_folder, image_file)
        image_target_path = os.path.join(target_image_folder, image_file)
        shutil.copy(image_source_path, image_target_path)

        # 构造对应的标注文件名
        text_file = os.path.splitext(image_file)[0] + ".txt"
        text_source_path = os.path.join(source_text_folder, text_file)
        text_target_path = os.path.join(target_text_folder, text_file)

        # 检查标注文件是否存在，再复制
        if os.path.exists(text_source_path):
            shutil.copy(text_source_path, text_target_path)
            print(f"[{i+1}/{max_files}] Copied: {image_file} and {text_file}")
        else:
            print(f"[{i+1}/{max_files}] WARNING: {text_file} not found!")

    print("Copy completed!")

# 设置源文件夹和目标文件夹路径
source_image_folder = "/home/zsl/label_everything/水面,循环水,清晰,gt,中密度/images"  # 替换为你的源图片文件夹路径
source_text_folder = "/home/zsl/label_everything/水面,循环水,清晰,gt,中密度/labels"   # 替换为你的源标注文件夹路径
target_image_folder = "/home/zsl/label_everything/dataset1/image" # 替换为你的目标图片文件夹路径
target_text_folder = "/home/zsl/label_everything/dataset1/label"   # 替换为你的目标标注文件夹路径

copy_first_200_files(source_image_folder, source_text_folder, target_image_folder, target_text_folder)
