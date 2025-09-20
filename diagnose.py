# diagnose.py
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *


def diagnose():
    print("=" * 60)
    print("系统诊断")
    print("=" * 60)

    # 检查目录权限
    directories = [BASE_DIR, PROCESSED_DATA_DIR, MODELS_DIR]
    for dir_path in directories:
        if os.path.exists(dir_path):
            try:
                # 尝试创建测试文件
                test_file = os.path.join(dir_path, "test_write.txt")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"✅ {dir_path} - 可写")
            except Exception as e:
                print(f"❌ {dir_path} - 不可写: {e}")
        else:
            print(f"❌ {dir_path} - 不存在")

    # 检查原始数据
    print("\n检查原始数据:")
    if os.path.exists(ORIGINAL_DATA_PATH):
        for class_name in CLASS_NAMES:
            class_path = os.path.join(ORIGINAL_DATA_PATH, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                print(f"📁 {class_name}: {len(images)} 张图像")
            else:
                print(f"❌ {class_name}: 文件夹不存在")
    else:
        print(f"❌ 原始数据路径不存在: {ORIGINAL_DATA_PATH}")

    print("=" * 60)


if __name__ == "__main__":
    diagnose()