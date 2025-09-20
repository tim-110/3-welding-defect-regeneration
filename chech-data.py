# check_data.py
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *


def check_data_structure():
    """检查数据目录结构"""
    print("=" * 60)
    print("数据目录结构检查")
    print("=" * 60)

    # 检查原始数据
    print("1. 检查原始数据:")
    if os.path.exists(ORIGINAL_DATA_PATH):
        for class_name in CLASS_NAMES:
            class_path = os.path.join(ORIGINAL_DATA_PATH, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                print(f"   {class_name}: {len(images)} 张图像")
            else:
                print(f"   ❌ {class_name}: 文件夹不存在")
    else:
        print(f"   ❌ 原始数据路径不存在: {ORIGINAL_DATA_PATH}")

    # 检查处理后数据
    print("\n2. 检查处理后数据:")
    if os.path.exists(PROCESSED_DATA_DIR):
        # 检查增强训练数据
        if os.path.exists(AUGMENTED_TRAIN_DIR):
            total_train_images = 0
            for class_name in CLASS_NAMES:
                class_path = os.path.join(AUGMENTED_TRAIN_DIR, class_name)
                if os.path.exists(class_path):
                    images = [f for f in os.listdir(class_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    print(f"   训练/{class_name}: {len(images)} 张图像")
                    total_train_images += len(images)
            print(f"   训练数据总计: {total_train_images} 张图像")
        else:
            print(f"   ❌ 增强训练数据路径不存在: {AUGMENTED_TRAIN_DIR}")
    else:
        print(f"   ❌ 处理后数据路径不存在: {PROCESSED_DATA_DIR}")

    print("=" * 60)


def create_sample_data():
    """创建示例数据（如果原始数据为空）"""
    print("创建示例数据...")

    # 确保原始数据目录存在
    os.makedirs(ORIGINAL_DATA_PATH, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_path = os.path.join(ORIGINAL_DATA_PATH, class_name)
        os.makedirs(class_path, exist_ok=True)

        # 创建一些空的占位文件（您需要替换为真实图像）
        for i in range(3):  # 每个类别创建3个占位文件
            placeholder_path = os.path.join(class_path, f"placeholder_{i}.txt")
            with open(placeholder_path, 'w') as f:
                f.write(f"这是 {class_name} 类别的占位文件\n")
                f.write("请用真实的焊接缺陷图像替换这些文件\n")
                f.write("支持的格式: .jpg, .png, .jpeg\n")

        print(f"创建了 {class_name} 类别的占位文件")


if __name__ == "__main__":
    check_data_structure()

    # 如果原始数据为空，创建示例文件结构
    if not os.path.exists(ORIGINAL_DATA_PATH) or len(os.listdir(ORIGINAL_DATA_PATH)) == 0:
        print("\n原始数据为空，创建示例文件结构...")
        create_sample_data()