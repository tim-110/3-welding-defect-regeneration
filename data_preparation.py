# data_preparation_simple.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# 导入配置
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *


def prepare_simple_dataset():
    """简化版数据准备"""
    print("=" * 60)
    print("开始准备数据集（简化版）")
    print("=" * 60)

    # 检查原始数据
    if not os.path.exists(ORIGINAL_DATA_PATH):
        print(f"❌ 原始数据路径不存在: {ORIGINAL_DATA_PATH}")
        return None

    # 创建目录结构
    directories = [
        os.path.join(PROCESSED_DATA_DIR, "augmented", "train"),
        os.path.join(PROCESSED_DATA_DIR, "augmented", "test"),
        os.path.join(PROCESSED_DATA_DIR, "original", "train"),
        os.path.join(PROCESSED_DATA_DIR, "original", "test")
    ]

    for dir_path in directories:
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(dir_path, class_name)
            os.makedirs(class_dir, exist_ok=True)

    print("✅ 目录结构创建完成")

    # 处理每个类别
    for class_name in CLASS_NAMES:
        original_class_path = os.path.join(ORIGINAL_DATA_PATH, class_name)

        if not os.path.exists(original_class_path):
            print(f"⚠️  跳过不存在的类别: {class_name}")
            continue

        # 获取图像文件
        image_files = [f for f in os.listdir(original_class_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not image_files:
            print(f"⚠️  {class_name}: 没有图像文件")
            continue

        print(f"处理 {class_name}: {len(image_files)} 张图像")

        # 划分训练测试集
        train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

        # 复制文件
        for file in train_files:
            src = os.path.join(original_class_path, file)
            dst = os.path.join(PROCESSED_DATA_DIR, "augmented", "train", class_name, file)
            shutil.copy2(src, dst)

        for file in test_files:
            src = os.path.join(original_class_path, file)
            dst = os.path.join(PROCESSED_DATA_DIR, "augmented", "test", class_name, file)
            shutil.copy2(src, dst)

    print("=" * 60)
    print("✅ 数据准备完成！")
    print(f"训练数据: {AUGMENTED_TRAIN_DIR}")
    print(f"测试数据: {AUGMENTED_TEST_DIR}")
    print("=" * 60)

    return AUGMENTED_TRAIN_DIR


if __name__ == "__main__":
    prepare_simple_dataset()