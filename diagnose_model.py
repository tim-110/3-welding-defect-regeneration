# diagnose_model.py
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *


def diagnose_model():
    """诊断模型状态"""
    print("=" * 60)
    print("模型诊断工具")
    print("=" * 60)

    # 检查文件是否存在
    model_exists = os.path.exists(MODEL_SAVE_PATH)
    scaler_exists = os.path.exists(SCALER_SAVE_PATH)

    print(f"模型文件: {MODEL_SAVE_PATH} {'✅' if model_exists else '❌'}")
    print(f"标准化器文件: {SCALER_SAVE_PATH} {'✅' if scaler_exists else '❌'}")

    if not model_exists or not scaler_exists:
        print("\n❌ 请先训练模型:")
        print("python welding_detection_system_fixed.py")
        return False

    # 检查文件大小
    model_size = os.path.getsize(MODEL_SAVE_PATH) if model_exists else 0
    scaler_size = os.path.getsize(SCALER_SAVE_PATH) if scaler_exists else 0

    print(f"模型文件大小: {model_size / 1024:.1f} KB")
    print(f"标准化器文件大小: {scaler_size / 1024:.1f} KB")

    if model_size < 1024:  # 小于1KB可能是损坏的文件
        print("❌ 模型文件可能损坏")
        return False

    # 尝试加载模型
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        import joblib

        print("\n尝试加载模型...")
        model = load_model(MODEL_SAVE_PATH)
        scaler = joblib.load(SCALER_SAVE_PATH)

        print("✅ 模型加载测试成功")
        print(f"模型输入形状: {model.input_shape}")
        print(f"模型输出形状: {model.output_shape}")

        return True

    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_data():
    """检查训练数据"""
    print("\n" + "=" * 60)
    print("训练数据检查")
    print("=" * 60)

    if not os.path.exists(AUGMENTED_TRAIN_DIR):
        print(f"❌ 训练数据路径不存在: {AUGMENTED_TRAIN_DIR}")
        return False

    total_images = 0
    for class_name in CLASS_NAMES:
        class_path = os.path.join(AUGMENTED_TRAIN_DIR, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f"{class_name}: {len(images)} 张图像")
            total_images += len(images)
        else:
            print(f"{class_name}: ❌ 文件夹不存在")

    print(f"总计: {total_images} 张训练图像")
    return total_images > 0


if __name__ == "__main__":
    # 检查模型
    model_ok = diagnose_model()

    # 检查训练数据
    data_ok = check_training_data()

    print("\n" + "=" * 60)
    if model_ok and data_ok:
        print("✅ 系统状态正常，可以开始预测")
        print("运行: python predict_fixed.py")
    else:
        print("❌ 系统存在问题，请检查:")
        if not data_ok:
            print("  - 训练数据不存在或为空")
            print("  - 运行: python data_preparation_simple.py")
        if not model_ok:
            print("  - 模型文件问题")
            print("  - 运行: python welding_detection_system_fixed.py")



