
import os
# 导入配置
import sys

import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *


class WeldingDefectDetector:
    def __init__(self):
        self.model = None
        self.class_names = CLASS_NAMES
        self.scaler = StandardScaler()
        self.is_trained = False

    def preprocess_image(self, image):
        """简化预处理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 调整尺寸
        gray = cv2.resize(gray, IMG_SIZE)

        # 简单预处理
        processed = cv2.GaussianBlur(gray, (5, 5), 0)
        return processed

    def extract_simple_features(self, image):
        """提取简化特征"""
        # 基本统计特征
        mean_val = np.mean(image)
        std_val = np.std(image)
        min_val = np.min(image)
        max_val = np.max(image)

        # 边缘特征
        edges = cv2.Canny(image, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        # 纹理特征（简化）
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        gradient_mean = np.mean(gradient_magnitude)

        features = [mean_val, std_val, min_val, max_val, edge_density, gradient_mean]
        return np.array(features)

    def prepare_dataset(self, data_dir):
        """准备数据集"""
        print(f"从 {data_dir} 加载数据...")
        X = []
        y = []

        for i, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"⚠️  类别文件夹不存在: {class_dir}")
                continue

            print(f"处理类别: {class_name}")

            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            if not image_files:
                print(f"⚠️  没有找到图像文件: {class_dir}")
                continue

            for img_name in image_files:
                img_path = os.path.join(class_dir, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                # 提取特征
                features = self.extract_simple_features(image)
                X.append(features)
                y.append(i)

        if len(X) == 0:
            print("❌ 没有提取到任何特征")
            return np.array([]), np.array([])

        print(f"✅ 成功提取 {len(X)} 个样本的特征")
        return np.array(X), np.array(y)

    def build_model(self, input_shape, num_classes):
        """构建简化模型"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, data_dir, EARLY_STOPPING_PATIENCE=None, REDUCE_LR_PATIENCE=None, MIN_LEARNING_RATE=None,
              VALIDATION_SPLIT=None):
        """训练模型"""
        print("开始提取特征...")
        X, y = self.prepare_dataset(data_dir)

        if len(X) == 0:
            print("❌ 无法训练：没有数据")
            return None

        print(f"数据集形状: X={X.shape}, y={y.shape}")

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

        # 标准化特征
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # 保存scaler
        joblib.dump(self.scaler, SCALER_SAVE_PATH)
        print(f"✅ 特征标准化器已保存: {SCALER_SAVE_PATH}")

        # 构建模型
        print("构建模型...")
        self.model = self.build_model(X_train.shape[1], len(self.class_names))

        # 设置回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LEARNING_RATE,
                verbose=1
            )
        ]

        # 训练模型
        print("开始训练模型...")
        history = self.model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )

        # 评估模型
        print("评估模型...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"📊 测试集准确率: {test_accuracy:.4f}")

        # 保存模型
        self.model.save(MODEL_SAVE_PATH)
        self.is_trained = True
        print(f"✅ 模型已保存: {MODEL_SAVE_PATH}")

        return history


if __name__ == "__main__":
    print("=" * 60)
    print("开始训练焊接缺陷检测模型")
    print("=" * 60)

    # 检查训练数据是否存在
    if not os.path.exists(AUGMENTED_TRAIN_DIR):
        print(f"❌ 训练数据不存在: {AUGMENTED_TRAIN_DIR}")
        print("请先运行 data_preparation.py 准备数据")
    else:
        detector = WeldingDefectDetector()
        try:
            history = detector.train(AUGMENTED_TRAIN_DIR)
            if history:
                print("🎉 模型训练完成！")
            else:

                print("❌ 模型训练失败")
        except Exception as e:
            print(f"❌ 训练过程中出错: {e}")
            import traceback

            traceback.print_exc()