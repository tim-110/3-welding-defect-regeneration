# dataset_inspector.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def inspect_dataset(dataset_path):
    """检查数据集结构和内容"""
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return

    print(f"🔍 检查数据集: {dataset_path}")
    print("=" * 60)

    # 检查目录结构
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}📁 {os.path.basename(root)}/")

        sub_indent = ' ' * 2 * (level + 1)
        for dir_name in dirs:
            print(f"{sub_indent}📁 {dir_name}/")

    print("\n📊 数据集统计:")
    print("=" * 40)

    total_images = 0
    class_stats = {}

    for split in ['train', 'test', 'val']:  # 检查训练集、测试集、验证集
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            print(f"\n{split.upper()} 集:")
            print("-" * 20)

            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    images = [f for f in os.listdir(class_path)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

                    if images:
                        # 检查第一张图像的属性
                        sample_image_path = os.path.join(class_path, images[0])
                        sample_image = cv2.imread(sample_image_path)

                        if sample_image is not None:
                            if len(sample_image.shape) == 3:
                                height, width, channels = sample_image.shape
                            else:
                                height, width = sample_image.shape
                                channels = 1

                            print(f"  {class_name}: {len(images)} 张图像")
                            print(f"    样本尺寸: {width}x{height}, 通道数: {channels}")

                            total_images += len(images)
                            if class_name not in class_stats:
                                class_stats[class_name] = 0
                            class_stats[class_name] += len(images)

    print(f"\n📈 总计: {total_images} 张图像")
    for class_name, count in class_stats.items():
        print(f"  {class_name}: {count} 张")


def visualize_dataset_samples(dataset_path, num_samples=5):
    """可视化数据集的样本"""
    print(f"\n🎨 可视化样本 (每个类别显示 {num_samples} 张)")
    print("=" * 50)

    train_path = os.path.join(dataset_path, 'train')
    if not os.path.exists(train_path):
        print("❌ 训练集不存在")
        return

    classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]

    fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 3 * len(classes)))
    if len(classes) == 1:
        axes = axes.reshape(1, -1)

    for i, class_name in enumerate(classes):
        class_path = os.path.join(train_path, class_name)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:num_samples]

        for j, img_name in enumerate(images):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)

            if image is not None:
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                axes[i, j].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
                axes[i, j].set_title(f'{class_name}\n{img_name}')
                axes[i, j].axis('off')
            else:
                axes[i, j].text(0.5, 0.5, '无法读取', ha='center', va='center')
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


def check_image_quality(dataset_path):
    """检查图像质量"""
    print("\n🔍 检查图像质量")
    print("=" * 40)

    problematic_images = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(root, file)

                # 检查文件大小
                file_size = os.path.getsize(img_path)
                if file_size == 0:
                    problematic_images.append((img_path, "空文件"))
                    continue

                # 检查图像可读性
                image = cv2.imread(img_path)
                if image is None:
                    problematic_images.append((img_path, "无法读取"))
                    continue

                # 检查图像尺寸
                if min(image.shape[:2]) < 50:  # 太小
                    problematic_images.append((img_path, f"尺寸过小: {image.shape}"))

    if problematic_images:
        print("❌ 发现有问题图像:")
        for img_path, issue in problematic_images[:10]:  # 显示前10个问题
            print(f"  {os.path.basename(img_path)}: {issue}")
        print(f"共发现 {len(problematic_images)} 个问题图像")
    else:
        print("✅ 所有图像质量良好")


def create_sample_dataset():
    """创建示例数据集结构"""
    sample_path = "sample_welding_dataset"
    os.makedirs(sample_path, exist_ok=True)

    # 创建目录结构
    splits = ['train', 'test', 'val']
    classes = ['porosity', 'crack', 'spatter']

    for split in splits:
        for class_name in classes:
            class_path = os.path.join(sample_path, split, class_name)
            os.makedirs(class_path, exist_ok=True)

    print(f"✅ 示例数据集结构已创建: {sample_path}")
    print("目录结构:")
    print("sample_welding_dataset/")
    print("├── train/")
    print("│   ├── porosity/")
    print("│   ├── crack/")
    print("│   └── spatter/")
    print("├── test/")
    print("│   ├── porosity/")
    print("│   ├── crack/")
    print("│   └── spatter/")
    print("└── val/")
    print("    ├── porosity/")
    print("    ├── crack/")
    print("    └── spatter/")

    return sample_path


def main():
    """主函数"""
    print("🔍 焊接缺陷数据集检查工具")
    print("=" * 50)

    while True:
        print("\n请选择操作:")
        print("1. 检查现有数据集")
        print("2. 可视化数据集样本")
        print("3. 检查图像质量")
        print("4. 创建示例数据集结构")
        print("5. 退出")

        choice = input("请输入选择 (1/2/3/4/5): ").strip()

        if choice == "1":
            dataset_path = input("请输入数据集路径: ").strip().strip('"')
            inspect_dataset(dataset_path)

        elif choice == "2":
            dataset_path = input("请输入数据集路径: ").strip().strip('"')
            visualize_dataset_samples(dataset_path)

        elif choice == "3":
            dataset_path = input("请输入数据集路径: ").strip().strip('"')
            check_image_quality(dataset_path)

        elif choice == "4":
            create_sample_dataset()

        elif choice == "5":
            print("再见!")
            break

        else:
            print("❌ 无效选择")


if __name__ == "__main__":
    main()