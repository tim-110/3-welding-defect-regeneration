import os
import tensorflow as tf
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             average_precision_score, accuracy_score)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, Conv2D, MaxPooling2D, \
    UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy import ndimage as ndi
import random
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import json
import glob
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import moment
from skimage.measure import regionprops
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# ========== 强制GPU内存优化 ==========
print("🚀 配置GPU内存使用策略...")

# 禁用内存增长，让TensorFlow直接分配大块内存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)

        # 设置GPU内存限制为14GB（给系统留4GB）
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=14 * 1024)]
        )
        print(f"✅ GPU内存配置: 分配14GB内存，禁用内存增长")
    except RuntimeError as e:
        print(f"❌ GPU配置错误: {e}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ 回退到内存增长模式")
else:
    print("⚠️ 未检测到GPU设备，使用CPU训练")

# 设置环境变量优化内存分配
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

# 验证GPU状态
print(f"🔍 GPU设备检测: {len(gpus)} 个物理GPU")

# 清除之前的会话
tf.keras.backend.clear_session()

print("🎯 GPU内存配置完成")

# 字体配置
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 8)

# 配置参数（优化后的参数）
LABELME_SOURCE_DIR = r"C:\Users\huanglei\PycharmProjects\PythonProject1\semi_auto_labeling\gpu_auto_labels"
LABELME_TRAIN_DIR = r"C:\Users\huanglei\PycharmProjects\PythonProject1\labelme_train_annotations"
LABELME_VAL_DIR = r"C:\Users\huanglei\PycharmProjects\PythonProject1\labelme_val_annotations"
LABELME_TRAIN_OUTPUT = r"C:\Users\huanglei\PycharmProjects\PythonProject1\labelme_train_data"
LABELME_VAL_OUTPUT = r"C:\Users\huanglei\PycharmProjects\PythonProject1\labelme_val_data"
SEGMENTATION_MODEL_PATH = r"C:\Users\huanglei\PycharmProjects\PythonProject1\models\segmentation_model.h5"
SEGMENTED_OUTPUT_PATH = r"C:\Users\huanglei\PycharmProjects\PythonProject1\segmented_data"

AUGMENTED_TRAIN_PATH = r"C:\Users\huanglei\PycharmProjects\PythonProject1\final_augmented_train"
AUGMENTED_VAL_PATH = r"C:\Users\huanglei\PycharmProjects\PythonProject1\final_augmented_val"
MODEL_SAVE_PATH = r"C:\Users\huanglei\PycharmProjects\PythonProject1\models\welding_defect_model.h5"
SCALER_SAVE_PATH = r"C:\Users\huanglei\PycharmProjects\PythonProject1\models\feature_scaler.pkl"
ENSEMBLE_SAVE_PATH = r"C:\Users\huanglei\PycharmProjects\PythonProject1\models\ensemble_model.pkl"
PLOT_SAVE_PATH = r"C:\Users\huanglei\PycharmProjects\PythonProject1\plots"

# 优化后的参数
IMG_SIZE = (128, 128)  # 调整为适中的尺寸
SEG_INPUT_SIZE = (256, 256, 3)
CLASS_NAMES = ['crack', 'porosity', 'spatter']
CLASS_NAMES_CN = ['裂纹', '气孔', '飞溅']
NUM_CLASSES = len(CLASS_NAMES)
EPOCHS = 100  # 减少训练轮次
BATCH_SIZE = 8  # 适中的批量大小
LEARNING_RATE = 0.001
SEG_EPOCHS = 50
SEG_BATCH_SIZE = 4


# ========================== 基础工具函数 ==========================
def create_necessary_directories():
    """创建必要目录"""
    directories = [
        AUGMENTED_TRAIN_PATH, AUGMENTED_VAL_PATH,
        os.path.dirname(MODEL_SAVE_PATH), PLOT_SAVE_PATH,
        LABELME_TRAIN_OUTPUT, LABELME_VAL_OUTPUT,
        SEGMENTED_OUTPUT_PATH, os.path.dirname(SCALER_SAVE_PATH),
        LABELME_TRAIN_DIR, LABELME_VAL_DIR
    ]

    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")

    return True


def auto_distribute_labelme_data():
    """自动将标注文件分配到训练集和验证集"""
    print("🔄 自动分配LabelMe标注数据...")

    # 确保目标目录存在
    os.makedirs(LABELME_TRAIN_DIR, exist_ok=True)
    os.makedirs(LABELME_VAL_DIR, exist_ok=True)

    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(LABELME_SOURCE_DIR, "*.json"))

    if not json_files:
        print(f"错误: 在 {LABELME_SOURCE_DIR} 中未找到JSON文件")
        return False

    print(f"找到 {len(json_files)} 个标注文件")

    # 按类别组织文件
    class_files = {class_name: [] for class_name in CLASS_NAMES}
    other_files = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取图像文件名
            image_filename = data['imagePath']
            image_path = os.path.join(LABELME_SOURCE_DIR, image_filename)

            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在 {image_path}")
                continue

            # 检查标注类别
            shapes = data.get('shapes', [])
            valid_classes = []

            for shape in shapes:
                label = shape['label']
                if label in CLASS_NAMES:
                    valid_classes.append(label)

            if valid_classes:
                # 使用第一个有效类别
                main_class = valid_classes[0]
                class_files[main_class].append((json_file, image_path))
            else:
                other_files.append((json_file, image_path))

        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
            continue

    # 统计类别分布
    print("\n📊 类别分布统计:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {len(class_files[class_name])} 个样本")

    # 按类别分配训练集和验证集
    train_files = []
    val_files = []

    for class_name in CLASS_NAMES:
        files = class_files[class_name]
        if len(files) < 2:
            print(f"警告: 类别 {class_name} 样本过少，全部用于训练")
            train_files.extend(files)
            continue

        # 按8:2分割
        train_class, val_class = train_test_split(
            files, test_size=0.2, random_state=42
        )
        train_files.extend(train_class)
        val_files.extend(val_class)

    print(f"\n📁 数据分配结果:")
    print(f"  训练集: {len(train_files)} 个样本")
    print(f"  验证集: {len(val_files)} 个样本")

    # 复制文件到对应目录
    def copy_files(file_list, target_dir):
        """复制文件到目标目录"""
        for json_file, image_path in file_list:
            try:
                # 复制JSON文件
                json_filename = os.path.basename(json_file)
                target_json = os.path.join(target_dir, json_filename)
                shutil.copy2(json_file, target_json)

                # 复制图像文件
                image_filename = os.path.basename(image_path)
                target_image = os.path.join(target_dir, image_filename)
                shutil.copy2(image_path, target_image)

            except Exception as e:
                print(f"复制文件时出错: {e}")

    # 复制训练集文件
    print("复制训练集文件...")
    copy_files(train_files, LABELME_TRAIN_DIR)

    # 复制验证集文件
    print("复制验证集文件...")
    copy_files(val_files, LABELME_VAL_DIR)

    print("✅ LabelMe数据自动分配完成")
    return True


def load_labelme_data(labelme_dir, output_img_dir):
    """将LabelMe JSON标注文件转换为模型可用的图像和标签"""
    # 创建输出目录
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(output_img_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(labelme_dir, "*.json"))

    if not json_files:
        print(f"警告: 在 {labelme_dir} 中未找到JSON文件")
        return output_img_dir

    processed_count = 0
    error_count = 0

    print(f"正在处理 {len(json_files)} 个LabelMe JSON文件...")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取图像路径
            image_path = data['imagePath']
            image_dir = os.path.dirname(json_file)
            full_image_path = os.path.join(image_dir, image_path)

            # 读取图像
            if not os.path.exists(full_image_path):
                print(f"警告: 图像文件不存在 {full_image_path}")
                continue

            image = cv2.imdecode(np.fromfile(full_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print(f"警告: 无法读取图像 {full_image_path}")
                continue

            # 获取标注信息
            shapes = data.get('shapes', [])
            if not shapes:
                print(f"警告: 文件 {json_file} 中没有标注")
                continue

            # 处理每个标注
            for i, shape in enumerate(shapes):
                label = shape['label']
                points = shape['points']

                # 检查标签是否在目标类别中
                if label not in CLASS_NAMES:
                    print(f"警告: 未知标签 '{label}'，跳过")
                    continue

                # 创建边界框或裁剪区域
                points_array = np.array(points, dtype=np.float32)
                x_min, y_min = np.min(points_array, axis=0)
                x_max, y_max = np.max(points_array, axis=0)

                # 确保坐标在图像范围内
                x_min = max(0, int(x_min))
                y_min = max(0, int(y_min))
                x_max = min(image.shape[1], int(x_max))
                y_max = min(image.shape[0], int(y_max))

                # 裁剪缺陷区域
                if x_max > x_min and y_max > y_min:
                    defect_region = image[y_min:y_max, x_min:x_max]

                    # 调整图像大小
                    if defect_region.size > 0:
                        defect_region = cv2.resize(defect_region, IMG_SIZE)

                        # 保存裁剪的图像
                        base_name = os.path.splitext(os.path.basename(json_file))[0]
                        output_filename = f"{base_name}_{i}.jpg"
                        output_path = os.path.join(output_img_dir, label, output_filename)

                        # 使用OpenCV保存，确保中文路径兼容
                        cv2.imencode('.jpg', defect_region)[1].tofile(output_path)
                        processed_count += 1

        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
            error_count += 1
            continue

    print(f"LabelMe数据转换完成: 成功处理 {processed_count} 个缺陷，错误 {error_count} 个")
    return output_img_dir


def create_dataset_from_labelme():
    """从LabelMe标注创建训练和验证数据集"""
    print("🔄 开始转换LabelMe标注数据...")

    # 首先自动分配数据
    if not auto_distribute_labelme_data():
        print("❌ LabelMe数据分配失败")
        return None, None

    # 转换训练数据
    print("转换训练数据...")
    train_output_dir = LABELME_TRAIN_OUTPUT
    load_labelme_data(LABELME_TRAIN_DIR, train_output_dir)

    # 转换验证数据
    print("转换验证数据...")
    val_output_dir = LABELME_VAL_OUTPUT
    load_labelme_data(LABELME_VAL_DIR, val_output_dir)

    return train_output_dir, val_output_dir


# ========================== 高级训练监控回调 ==========================
class AdvancedTrainingMonitor(Callback):
    """高级训练监控回调，实时显示关键指标"""

    def __init__(self, validation_data, class_names_cn):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.class_names_cn = class_names_cn
        self.best_map50 = 0
        self.best_accuracy = 0
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # 计算epoch时间
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        avg_time = np.mean(self.epoch_times[-5:]) if len(self.epoch_times) >= 5 else epoch_time

        # 验证集预测
        y_pred_proba = self.model.predict(self.X_val, verbose=0, batch_size=BATCH_SIZE)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # 计算关键指标
        accuracy = accuracy_score(self.y_val, y_pred)
        map50 = self.calculate_map50(self.y_val, y_pred_proba)

        # 各类别精度和召回率
        class_precision = precision_score(self.y_val, y_pred, average=None, zero_division=0)
        class_recall = recall_score(self.y_val, y_pred, average=None, zero_division=0)
        class_f1 = f1_score(self.y_val, y_pred, average=None, zero_division=0)

        # 更新最佳指标
        if map50 > self.best_map50:
            self.best_map50 = map50
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy

        # 打印详细指标
        print(f"\n🎯 Epoch {epoch + 1:03d} 详细指标:")
        print(f"   ⏱️  时间: {epoch_time:.1f}s (平均: {avg_time:.1f}s)")
        print(f"   📊 准确率: {accuracy:.4f} | 最佳: {self.best_accuracy:.4f}")
        print(f"   🎯 mAP@50: {map50:.4f} | 最佳: {self.best_map50:.4f}")
        print(f"   📈 损失: {logs.get('loss', 0):.4f} | 验证损失: {logs.get('val_loss', 0):.4f}")

        # 打印各类别指标
        print(f"   🎪 各类别性能:")
        for i, class_name in enumerate(self.class_names_cn):
            print(f"     {class_name}: 精度={class_precision[i]:.3f}, 召回={class_recall[i]:.3f}, F1={class_f1[i]:.3f}")

        # 学习率信息
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"   📚 学习率: {current_lr:.6f}")

    def calculate_map50(self, y_true, y_pred_proba):
        """计算mAP@50"""
        aps = []
        for class_idx in range(NUM_CLASSES):
            binary_labels = (y_true == class_idx).astype(int)
            class_probs = y_pred_proba[:, class_idx]
            ap = average_precision_score(binary_labels, class_probs)
            aps.append(ap)
        return np.mean(aps) if aps else 0


class ProgressLogger(Callback):
    """训练进度记录器"""

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n🚀 开始训练: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"\n✅ 训练完成! 总时间: {total_time / 60:.1f} 分钟")
        print("=" * 80)


# ========================== 改进的特征提取 ==========================
def improved_preprocess_image(image):
    """改进的图像预处理"""
    # 调整大小
    image = cv2.resize(image, IMG_SIZE)

    # 对比度增强
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 噪声去除
    image = cv2.medianBlur(image, 3)

    return image


def extract_moment_features(image):
    """提取矩特征"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 计算Hu矩
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()

        # 对数变换
        for i in range(len(hu_moments)):
            if hu_moments[i] != 0:
                hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))

        # 计算像素矩
        pixel_moments = [
            np.mean(gray), np.std(gray), np.var(gray),
            moment(gray.flatten(), moment=3), moment(gray.flatten(), moment=4)
        ]

        return np.concatenate([hu_moments, pixel_moments])
    except:
        return np.zeros(12)  # 7个Hu矩 + 5个像素矩


def extract_enhanced_features(image):
    """改进的特征提取 - 提高特征质量"""
    try:
        # 使用改进的预处理
        image = improved_preprocess_image(image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        features = []

        # 1. 核心统计特征
        features.extend([
            np.mean(gray), np.std(gray), np.median(gray),
            np.percentile(gray, 25), np.percentile(gray, 75),
            np.percentile(gray, 90), moment(gray.flatten(), moment=3),
            moment(gray.flatten(), moment=4)
        ])

        # 2. LBP特征
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
        lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
        features.extend(lbp_hist)

        # 3. GLCM特征
        glcm = graycomatrix(gray, distances=[1], angles=[0],
                            levels=256, symmetric=True, normed=True)
        glcm_props = ['contrast', 'correlation', 'energy', 'homogeneity']
        for prop in glcm_props:
            features.append(graycoprops(glcm, prop)[0, 0])

        # 4. 形状特征
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                features.extend([area, perimeter, circularity])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])

        # 5. 矩特征
        moment_features = extract_moment_features(image)
        features.extend(moment_features[:10])  # 取前10个最重要的矩特征

        return np.array(features)

    except Exception as e:
        print(f"特征提取错误: {str(e)}")
        return None


# ========================== 数据增强函数 ==========================
def advanced_augment_image(image):
    """高级数据增强"""
    augmented = []

    # 基础增强
    # 随机旋转
    angle = random.uniform(-15, 15)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    augmented.append(rotated)

    # 随机水平翻转
    if random.random() > 0.5:
        flipped = cv2.flip(image, 1)
        augmented.append(flipped)

    # 随机亮度调整
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.7, 1.3)
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
            brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            brightened = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        augmented.append(brightened)

    return augmented


def augment_dataset(original_path, output_path, target_multiplier=2):
    """增强数据集"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    total_original = 0
    total_augmented = 0

    for class_name in CLASS_NAMES:
        class_original_path = os.path.join(original_path, class_name)
        class_output_path = os.path.join(output_path, class_name)

        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)

        image_files = [f for f in os.listdir(class_original_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not image_files:
            print(f"警告: 类别 {class_name} 中未找到图片文件")
            continue

        original_count = len(image_files)
        target_count = original_count * target_multiplier
        need_to_add = target_count - original_count

        if need_to_add <= 0:
            print(f"类别 {class_name} 已有足够图片，无需增强")
            continue

        print(f"增强类别 {class_name}: 原始{original_count}张，需要增加{need_to_add}张以达到{target_count}张...")

        # 复制原始图片
        for img_file in image_files:
            src = os.path.join(class_original_path, img_file)
            dst = os.path.join(class_output_path, img_file)
            image = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imencode(os.path.splitext(img_file)[1], image)[1].tofile(dst)

        # 生成增强图片
        added = 0
        while added < need_to_add:
            img_file = random.choice(image_files)
            img_path = os.path.join(class_original_path, img_file)

            try:
                image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue

                augmented_images = advanced_augment_image(image)

                for aug_img in augmented_images:
                    if added >= need_to_add:
                        break

                    base_name, ext = os.path.splitext(img_file)
                    aug_filename = f"{base_name}_aug_{added}{ext}"
                    aug_path = os.path.join(class_output_path, aug_filename)

                    cv2.imencode(ext, aug_img)[1].tofile(aug_path)
                    added += 1
                    total_augmented += 1

            except Exception as e:
                print(f"处理图片 {img_file} 时出错: {str(e)}")
                continue

        total_original += original_count
        print(f"类别 {class_name} 增强完成: 共{target_count}张图片")

    print(f"\n数据集增强完成: 原始{total_original}张，新增{total_augmented}张，总计{total_original + total_augmented}张")
    return output_path


def load_specific_dataset(dataset_path, sample_ratio=1.0):
    """加载数据集"""
    features = []
    labels = []

    # 多线程处理函数
    def process_image(img_path, class_idx):
        try:
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return None
            feat = extract_enhanced_features(image)
            if feat is not None:
                return (feat, class_idx)
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {str(e)}")
        return None

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"警告: 类别文件夹不存在 - {class_path}")
            continue

        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # 对每个类别单独抽样
        if sample_ratio < 1.0 and len(image_files) > 0:
            sample_size = max(1, int(len(image_files) * sample_ratio))
            image_files = random.sample(image_files, sample_size)
            print(f"⚠️ 快速验证：{class_name} 类别随机抽样 {sample_ratio * 100}%，保留 {len(image_files)} 张")

        if not image_files:
            print(f"警告: 类别 {class_name} 的文件夹中未找到任何图片文件")
            continue

        print(f"正在加载 {os.path.basename(dataset_path)} 中的 {class_name} 类别，共 {len(image_files)} 张图片...")
        img_paths = [os.path.join(class_path, f) for f in image_files]

        # 多线程并行处理图片
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
            futures = [executor.submit(process_image, path, class_idx) for path in img_paths]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    features.append(result[0])
                    labels.append(result[1])

    if not features:
        raise ValueError(f"未从 {dataset_path} 加载到任何有效数据")

    print(f"从 {os.path.basename(dataset_path)} 成功加载 {len(features)} 张图片数据")
    return np.array(features), np.array(labels)


# ========================== 改进的模型架构 ==========================
def build_advanced_model(input_dim):
    """构建改进的神经网络模型"""
    model = Sequential([
        Dense(512, input_dim=input_dim, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),

        Dense(256, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),

        Dense(128, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),

        Dense(64, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.1),

        Dense(NUM_CLASSES, activation='softmax')
    ])

    # 使用Adam优化器
    optimizer = Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


# ========================== 改进的回调函数 ==========================
def get_enhanced_callbacks(X_val, y_val):
    """获取增强的回调函数"""
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),

        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),

        AdvancedTrainingMonitor((X_val, y_val), CLASS_NAMES_CN),
        ProgressLogger()
    ]


# ========================== 详细的性能分析 ==========================
def detailed_performance_analysis(model, X_val, y_val, model_type='neural'):
    """详细的性能分析"""
    if model_type == 'neural':
        y_pred_proba = model.predict(X_val, verbose=0, batch_size=BATCH_SIZE)
    else:
        y_pred_proba = model.predict_proba(X_val)

    y_pred = np.argmax(y_pred_proba, axis=1)
    confidences = np.max(y_pred_proba, axis=1)

    print("\n" + "=" * 80)
    print("📊 详细性能分析报告")
    print("=" * 80)

    # 基础指标
    accuracy = accuracy_score(y_val, y_pred)
    precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
    map50 = calculate_map50(y_val, y_pred_proba)

    print(f"🎯 整体性能:")
    print(f"   准确率: {accuracy:.4f}")
    print(f"   精确率 (macro): {precision_macro:.4f}")
    print(f"   召回率 (macro): {recall_macro:.4f}")
    print(f"   F1分数 (macro): {f1_macro:.4f}")
    print(f"   mAP@50: {map50:.4f}")
    print(f"   平均置信度: {np.mean(confidences):.4f}")

    # 各类别详细指标
    print(f"\n🎪 各类别性能:")
    class_precision = precision_score(y_val, y_pred, average=None, zero_division=0)
    class_recall = recall_score(y_val, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_val, y_pred, average=None, zero_division=0)

    for i in range(NUM_CLASSES):
        class_mask = (y_val == i)
        class_accuracy = accuracy_score(y_val[class_mask], y_pred[class_mask]) if np.sum(class_mask) > 0 else 0
        class_avg_confidence = np.mean(confidences[class_mask]) if np.sum(class_mask) > 0 else 0

        print(f"   {CLASS_NAMES_CN[i]}:")
        print(f"     准确率: {class_accuracy:.4f}")
        print(f"     精确率: {class_precision[i]:.4f}")
        print(f"     召回率: {class_recall[i]:.4f}")
        print(f"     F1分数: {class_f1[i]:.4f}")
        print(f"     平均置信度: {class_avg_confidence:.4f}")

    # 置信度分析
    print(f"\n📈 置信度分布分析:")
    confidence_bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    confidence_labels = ['低(<0.5)', '中(0.5-0.7)', '高(0.7-0.9)', '很高(>0.9)']

    for (low, high), label in zip(confidence_bins, confidence_labels):
        count = np.sum((confidences >= low) & (confidences < high))
        percentage = count / len(confidences) * 100
        correct_count = np.sum(((confidences >= low) & (confidences < high)) & (y_pred == y_val))
        correct_percentage = correct_count / count * 100 if count > 0 else 0
        print(f"   {label}: {count}个 ({percentage:.1f}%), 正确率: {correct_percentage:.1f}%")

    return y_pred, confidences, y_pred_proba


def calculate_map50(y_true, y_pred_proba):
    """计算mAP@50"""
    aps = []
    for class_idx in range(NUM_CLASSES):
        binary_labels = (y_true == class_idx).astype(int)
        class_probs = y_pred_proba[:, class_idx]
        ap = average_precision_score(binary_labels, class_probs)
        aps.append(ap)
    return np.mean(aps) if aps else 0


# ========================== 改进的训练流程 ==========================
def check_data_balance(y_train, y_val):
    """检查数据平衡性"""
    print("📊 数据分布检查:")
    class_weights = {}

    for i, class_name in enumerate(CLASS_NAMES_CN):
        train_count = np.sum(y_train == i)
        val_count = np.sum(y_val == i)
        total_count = train_count + val_count

        # 计算类别权重（处理不平衡数据）
        if train_count > 0:
            weight = len(y_train) / (NUM_CLASSES * train_count)
            class_weights[i] = weight
        else:
            class_weights[i] = 1.0

        print(f"  {class_name}: 训练集 {train_count} 个, 验证集 {val_count} 个, 权重: {class_weights[i]:.2f}")

    return class_weights


def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 损失曲线
    ax1.plot(history.history['loss'], label='训练损失', linewidth=2)
    ax1.plot(history.history['val_loss'], label='验证损失', linewidth=2)
    ax1.set_title('训练与验证损失对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 准确率曲线
    ax2.plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    ax2.set_title('训练与验证准确率对比', fontsize=14, fontweight='bold')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(PLOT_SAVE_PATH, 'improved_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()


def improved_training_workflow():
    """改进的训练流程"""
    print("🚀 开始高级焊接缺陷检测模型训练...")

    # 创建必要目录
    create_necessary_directories()

    # 使用LabelMe数据
    print("\n" + "=" * 80)
    print("📝 处理LabelMe标注数据...")

    # 转换LabelMe数据
    labelme_train_path, labelme_val_path = create_dataset_from_labelme()

    if labelme_train_path is None:
        print("❌ LabelMe数据处理失败，退出训练")
        return

    # 数据增强
    print("\n" + "=" * 80)
    print("🔄 数据增强处理...")
    augmented_train_path = augment_dataset(labelme_train_path, AUGMENTED_TRAIN_PATH, target_multiplier=2)
    augmented_val_path = augment_dataset(labelme_val_path, AUGMENTED_VAL_PATH, target_multiplier=2)

    # 加载数据
    print("\n" + "=" * 80)
    print("📂 加载训练数据...")
    try:
        X_train, y_train = load_specific_dataset(augmented_train_path, sample_ratio=0.05)
        X_val, y_val = load_specific_dataset(augmented_val_path, sample_ratio=0.05)
    except ValueError as e:
        print(f"❌ 错误: {e}")
        return

    # 数据平衡检查
    print("\n📊 数据分布检查:")
    class_weight_dict = check_data_balance(y_train, y_val)

    # 特征标准化
    print("\n🔧 特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"特征维度: {X_train_scaled.shape[1]}")

    # 训练主模型
    print("\n" + "=" * 80)
    print("🧠 训练高级神经网络模型...")
    model = build_advanced_model(X_train_scaled.shape[1])

    # 模型概要
    model.summary()

    # 训练模型
    history = model.fit(
        X_train_scaled, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_scaled, y_val),
        callbacks=get_enhanced_callbacks(X_val_scaled, y_val),
        class_weight=class_weight_dict,
        verbose=1
    )

    # 详细性能分析
    print("\n" + "=" * 80)
    print("📊 最终性能分析...")
    y_pred, confidences, y_pred_proba = detailed_performance_analysis(model, X_val_scaled, y_val, 'neural')

    # 保存模型
    model.save(MODEL_SAVE_PATH)
    print(f"\n💾 模型已保存: {MODEL_SAVE_PATH}")

    # 绘制训练历史
    plot_training_history(history)

    print("\n✅ 训练完成！所有优化已实施。")


def main():
    improved_training_workflow()


if __name__ == "__main__":
    main()
