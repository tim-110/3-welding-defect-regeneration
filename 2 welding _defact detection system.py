import os
import sys
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import albumentations as A  # 用于数据增强

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置参数
IMG_SIZE = (128, 128)
CLASS_NAMES = ['crack', 'porosity', 'spatter', 'good']
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_PATH = "welding_defect_model.h5"
SCALER_SAVE_PATH = "feature_scaler.pkl"
AUGMENTED_TRAIN_DIR = "augmented_train_data"


class WeldingDefectDetector:
    def __init__(self):
        self.model = None
        self.class_names = CLASS_NAMES
        self.scaler = StandardScaler()
        self.is_trained = False

    def augment_image(self, image):
        """数据增强 - 创建10倍的增强图像（优化后：消除警告+整合逻辑）"""
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.VerticalFlip(p=0.5),  # 垂直翻转
            # 用 Affine 整合“旋转+平移+缩放”，替代 Rotate + ShiftScaleRotate
            A.Affine(
                rotate_limit=30,  # 旋转范围±30°（覆盖原 Rotate 的 30°，比原 ShiftScaleRotate 的15°更灵活）
                shift_limit=0.1,  # 平移范围±10%（保留原 ShiftScaleRotate 的平移功能）
                scale_limit=0.1,  # 缩放范围±10%（保留原 ShiftScaleRotate 的缩放功能）
                p=0.8  # 80%概率触发（与原 Rotate 一致）
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # 亮度/对比度调整
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # 高斯模糊
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # 弹性形变
        ])
        # 后续增强逻辑（生成10倍图像）不变...

        augmented_images = []
        for _ in range(10):  # 创建10个增强版本
            augmented = transform(image=image)['image']
            augmented_images.append(augmented)

        return augmented_images

    def augment_dataset(self, original_data_dir, output_dir):
        """增强整个数据集"""
        print(f"开始数据增强: {original_data_dir} -> {output_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for class_name in self.class_names:
            class_input_dir = os.path.join(original_data_dir, class_name)
            class_output_dir = os.path.join(output_dir, class_name)

            if not os.path.exists(class_output_dir):
                os.makedirs(class_output_dir)

            if not os.path.exists(class_input_dir):
                print(f"⚠️ 原始类别文件夹不存在: {class_input_dir}")
                continue

            image_files = [f for f in os.listdir(class_input_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            print(f"增强类别 {class_name}: {len(image_files)} 张原始图像")

            for img_name in image_files:
                img_path = os.path.join(class_input_dir, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                # 保存原始图像
                original_output_path = os.path.join(class_output_dir, f"original_{img_name}")
                cv2.imwrite(original_output_path, image)

                # 生成增强图像
                augmented_images = self.augment_image(image)

                for i, aug_img in enumerate(augmented_images):
                    aug_output_path = os.path.join(class_output_dir, f"aug_{i}_{img_name}")
                    cv2.imwrite(aug_output_path, aug_img)

        print(f"✅ 数据增强完成! 输出目录: {output_dir}")

    def preprocess_image(self, image):
        """预处理图像"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 调整尺寸
        gray = cv2.resize(gray, IMG_SIZE)

        # 预处理
        processed = cv2.GaussianBlur(gray, (5, 5), 0)
        processed = cv2.equalizeHist(processed)  # 直方图均衡化

        return processed

    def extract_advanced_features(self, image):
        """提取更丰富的特征"""
        # 基本统计特征
        mean_val = np.mean(image)
        std_val = np.std(image)
        min_val = np.min(image)
        max_val = np.max(image)

        # 直方图特征
        hist = cv2.calcHist([image], [0], None, [16], [0, 256])
        hist = hist.flatten() / hist.sum() if hist.sum() > 0 else hist.flatten()

        # 边缘特征
        edges = cv2.Canny(image, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        # 纹理特征 - Sobel
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        gradient_mean = np.mean(gradient_magnitude)
        gradient_std = np.std(gradient_magnitude)

        # 纹理特征 - LBP (简化版)
        lbp_features = self.calculate_simple_lbp(image)

        # 组合所有特征
        features = [
            mean_val, std_val, min_val, max_val,
            edge_density, gradient_mean, gradient_std
        ]
        features.extend(hist[:8])  # 取前8个直方图特征
        features.extend(lbp_features[:5])  # 取前5个LBP特征

        return np.array(features)

    def calculate_simple_lbp(self, image):
        """计算简化的LBP特征"""
        lbp = np.zeros_like(image, dtype=np.uint8)
        height, width = image.shape

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center = image[i, j]
                code = 0
                code |= (image[i - 1, j - 1] > center) << 7
                code |= (image[i - 1, j] > center) << 6
                code |= (image[i - 1, j + 1] > center) << 5
                code |= (image[i, j + 1] > center) << 4
                code |= (image[i + 1, j + 1] > center) << 3
                code |= (image[i + 1, j] > center) << 2
                code |= (image[i + 1, j - 1] > center) << 1
                code |= (image[i, j - 1] > center) << 0
                lbp[i, j] = code

        # 计算LBP直方图
        lbp_hist = cv2.calcHist([lbp], [0], None, [16], [0, 256])
        lbp_hist = lbp_hist.flatten() / lbp_hist.sum() if lbp_hist.sum() > 0 else lbp_hist.flatten()

        return lbp_hist

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

                # 预处理
                processed = self.preprocess_image(image)

                # 提取特征
                features = self.extract_advanced_features(processed)
                X.append(features)
                y.append(i)

        if len(X) == 0:
            print("❌ 没有提取到任何特征")
            return np.array([]), np.array([])

        print(f"✅ 成功提取 {len(X)} 个样本的特征")
        return np.array(X), np.array(y)

    def build_advanced_model(self, input_shape, num_classes):
        """构建更强大的模型（修正指标错误）"""
        # 导入多分类专用的精确率和召回率指标
        from tensorflow.keras.metrics import (
            SparseCategoricalAccuracy,
            SparseCategoricalPrecision,
            SparseCategoricalRecall
        )

        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            # 使用多分类专用指标函数，而非字符串
            metrics=[
                SparseCategoricalAccuracy(name='accuracy'),  # 准确率
                SparseCategoricalPrecision(name='precision'),  # 精确率
                SparseCategoricalRecall(name='recall')  # 召回率
            ]
        )

        return model

        return model

    def train(self, data_dir, validation_split=0.2):
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
        self.model = self.build_advanced_model(X_train.shape[1], len(self.class_names))
        self.model.summary()

        # 设置回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # 训练模型
        print("开始训练模型...model.compile")
        history = self.model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # 评估模型
        print("评估模型...")
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"📊 测试集准确率: {test_accuracy:.4f}")
        print(f"📊 测试集精确率: {test_precision:.4f}")
        print(f"📊 测试集召回率: {test_recall:.4f}")

        # 预测并生成分类报告
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        print("\n📋 分类报告:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))

        print("\n📊 混淆矩阵:")
        print(confusion_matrix(y_test, y_pred_classes))

        # 保存模型
        self.model.save(MODEL_SAVE_PATH)
        self.is_trained = True
        print(f"✅ 模型已保存: {MODEL_SAVE_PATH}")

        return history

    def predict_image(self, image_path):
        """预测单张图像的缺陷类型"""
        if not self.is_trained:
            # 尝试加载已训练的模型
            if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
                try:
                    self.model = models.load_model(MODEL_SAVE_PATH)
                    self.scaler = joblib.load(SCALER_SAVE_PATH)
                    self.is_trained = True
                    print("✅ 已加载预训练模型")
                except:
                    print("❌ 无法加载预训练模型，请先训练模型")
                    return None
            else:
                print("❌ 模型未训练，请先训练模型")
                return None

        # 加载和预处理图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None

        processed = self.preprocess_image(image)
        features = self.extract_advanced_features(processed)

        # 标准化特征
        features_scaled = self.scaler.transform([features])

        # 预测
        prediction = self.model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        # 显示结果
        result = {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                cls: float(prob) for cls, prob in zip(self.class_names, prediction[0])
            }
        }

        print(f"\n🔍 预测结果:")
        print(f"图像: {os.path.basename(image_path)}")
        print(f"预测缺陷类型: {result['class']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("所有类别概率:")
        for cls, prob in result['all_probabilities'].items():
            print(f"  {cls}: {prob:.4f}")

        # 可视化结果
        self.visualize_prediction(image, result)

        return result

    def visualize_prediction(self, image, result):
        """可视化预测结果"""
        plt.figure(figsize=(12, 5))

        # 显示原始图像
        plt.subplot(1, 2, 1)
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.title(f'Original Image\nPredicted: {result["class"]}\nConfidence: {result["confidence"]:.3f}')
        plt.axis('off')

        # 显示概率分布
        plt.subplot(1, 2, 2)
        classes = list(result['all_probabilities'].keys())
        probabilities = list(result['all_probabilities'].values())

        colors = ['red' if prob == max(probabilities) else 'blue' for prob in probabilities]
        bars = plt.bar(classes, probabilities, color=colors, alpha=0.7)

        plt.title('Prediction Probabilities')
        plt.xlabel('Defect Classes')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

        # 在柱状图上添加数值标签
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{prob:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("焊接缺陷检测系统")
    print("=" * 60)

    detector = WeldingDefectDetector()

    while True:
        print("\n请选择操作:")
        print("1. 数据增强")
        print("2. 训练模型")
        print("3. 预测图像缺陷")
        print("4. 退出")

        choice = input("请输入选择 (1/2/3/4): ").strip()

        if choice == "1":
            original_dir = input("请输入原始数据目录路径: ").strip()
            if os.path.exists(original_dir):
                detector.augment_dataset(original_dir, AUGMENTED_TRAIN_DIR)
            else:
                print("❌ 目录不存在")

        elif choice == "2":
            data_dir = input("请输入训练数据目录路径 (直接回车使用增强数据): ").strip()
            if not data_dir:
                data_dir = AUGMENTED_TRAIN_DIR

            if os.path.exists(data_dir):
                history = detector.train(data_dir)
                if history:
                    print("🎉 模型训练完成！")
            else:
                print("❌ 训练数据目录不存在")

        elif choice == "3":
            image_path = input("请输入要预测的图像路径: ").strip()
            if os.path.exists(image_path):
                result = detector.predict_image(image_path)
            else:
                print("❌ 图像文件不存在")

        elif choice == "4":
            print("再见!")
            break

        else:
            print("❌ 无效选择")





if __name__ == "__main__":
    main()