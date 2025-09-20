import os
import cv2
import numpy as np
import joblib
import traceback  # 补充导入traceback，解决之前的未定义问题
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf

# 导入配置和特征提取类
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *  # 确保config.py中定义了IMG_SIZE、CLASS_NAMES、MODEL_SAVE_PATH、SCALER_SAVE_PATH
from feature_extractor import WeldingFeatureExtractor  # 确保该类有extract_all_features方法


# -------------------------- 关键：删除重复的WeldingDefectTrainer类，保留完整的第一个类 --------------------------
class WeldingDefectTrainer:
    def __init__(self):
        # 初始化特征提取器（与预测时保持一致）
        self.feature_extractor = WeldingFeatureExtractor(img_size=IMG_SIZE)
        self.scaler = StandardScaler()  # 用于特征标准化
        self.model = self.build_model()  # 构建接收28个特征的模型

        # -------------------------- 中文显示配置：放在类初始化中，确保绘图前生效 --------------------------
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 适配不同系统的中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    def build_model(self):
        """构建适配28个特征输入的神经网络模型"""
        model = Sequential([
            # 输入层：明确接收28个特征
            Dense(128, activation='relu', input_shape=(28,)),
            BatchNormalization(),  # 加速训练并防止过拟合
            Dropout(0.3),  # 随机丢弃30%神经元，防止过拟合

            # 隐藏层
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            Dense(32, activation='relu'),
            BatchNormalization(),

            # 输出层：根据缺陷类别数量设置神经元数量（CLASS_NAMES从config.py导入）
            Dense(len(CLASS_NAMES), activation='softmax')
        ])

        # 编译模型（使用sparse_categorical_crossentropy适配整数标签）
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_dataset(self, data_dir):
        """从文件夹加载数据集并提取28个特征"""
        X = []  # 特征列表
        y = []  # 标签列表

        # 遍历每个缺陷类别文件夹（CLASS_NAMES从config.py导入，如['porosity', 'crack', 'spatter']）
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ 类别文件夹不存在: {class_dir}")
                continue

            # 遍历文件夹中的图像
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                # 过滤非图像文件
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue

                try:
                    # 读取图像并提取28个特征
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"⚠️ 无法读取图像: {img_path}")
                        continue

                    # 调用特征提取器提取28个特征（确保extract_all_features返回长度为28的列表/数组）
                    features = self.feature_extractor.extract_all_features(image)

                    # 验证特征数量是否为28，避免异常特征影响训练
                    if len(features) != 28:
                        print(f"⚠️ 特征数量异常（预期28，实际{len(features)}）: {img_path}")
                        continue

                    X.append(features)
                    y.append(class_idx)  # 用整数表示标签（适配sparse_categorical_crossentropy）

                except Exception as e:
                    print(f"⚠️ 处理图像失败 {img_path}: {e}")
                    continue

        if not X:
            raise ValueError("❌ 未加载到任何有效数据，请检查数据集路径和格式")

        return np.array(X), np.array(y)

    def train(self, train_dir, val_dir=None):
        """训练模型"""
        # 1. 加载训练数据
        print("🔍 加载训练数据并提取28个特征...")
        X_train, y_train = self.load_dataset(train_dir)
        print(f"✅ 训练数据加载完成: {len(X_train)}个样本，每个样本28个特征")

        # 2. 划分验证集（如果没有单独的验证集文件夹）
        if val_dir is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,  # 20%数据作为验证集
                random_state=42,  # 固定随机种子，确保结果可复现
                stratify=y_train  # 保持类别比例一致，避免验证集类别失衡
            )
            print(f"✅ 自动划分验证集: {len(X_val)}个样本")
        else:
            # 使用单独的验证集文件夹
            print("🔍 加载验证数据...")
            X_val, y_val = self.load_dataset(val_dir)
            print(f"✅ 验证数据加载完成: {len(X_val)}个样本")

        # 3. 特征标准化（仅用训练集拟合scaler，避免数据泄露）
        print("🔄 标准化特征...")
        self.scaler.fit(X_train)  # 计算28个特征的均值和标准差（仅用训练集）
        X_train_scaled = self.scaler.transform(X_train)  # 标准化训练集
        X_val_scaled = self.scaler.transform(X_val)  # 标准化验证集（使用训练集的均值/标准差）

        # 4. 定义训练回调（提前停止+保存最佳模型）
        callbacks = [
            # 当验证集准确率连续5轮不提升时停止训练，恢复最佳权重
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
            # 保存验证集准确率最高的模型（路径从config.py导入）
            ModelCheckpoint(
                MODEL_SAVE_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # 5. 训练模型
        print("🚀 开始训练模型（输入特征数：28）...")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=50,  # 最大训练轮次（EarlyStopping会提前停止）
            batch_size=16,  # 批次大小（根据CPU/GPU性能调整）
            callbacks=callbacks,
            verbose=1  # 显示训练过程（1=显示进度条，2=仅显示轮次结果）
        )

        # 6. 保存标准化器（供预测时使用，路径从config.py导入）
        joblib.dump(self.scaler, SCALER_SAVE_PATH)
        print(f"✅ 特征标准化器已保存至: {SCALER_SAVE_PATH}")

        # 7. 绘制训练曲线（调用类内方法，中文已配置）
        self.plot_training_history(history)

        return history  # 返回训练历史，供后续分析

    def plot_training_history(self, history):
        """绘制训练过程中的准确率和损失曲线（中文标签）"""
        plt.figure(figsize=(12, 4))  # 设置画布大小（宽12，高4）

        # -------------------------- 子图1：准确率曲线（中文标签） --------------------------
        plt.subplot(1, 2, 1)  # 1行2列，第1个子图
        plt.plot(history.history['accuracy'], color='#1f77b4', linewidth=2, label='训练准确率')
        plt.plot(history.history['val_accuracy'], color='#ff7f0e', linewidth=2, label='验证准确率')
        plt.title('模型准确率变化', fontsize=12)  # 中文标题
        plt.xlabel('训练轮次（Epoch）', fontsize=10)  # 中文X轴标签
        plt.ylabel('准确率', fontsize=10)  # 中文Y轴标签
        plt.legend(fontsize=9)  # 显示图例
        plt.grid(alpha=0.3)  # 添加网格，便于查看数值

        # -------------------------- 子图2：损失曲线（中文标签） --------------------------
        plt.subplot(1, 2, 2)  # 1行2列，第2个子图
        plt.plot(history.history['loss'], color='#1f77b4', linewidth=2, label='训练损失')
        plt.plot(history.history['val_loss'], color='#ff7f0e', linewidth=2, label='验证损失')
        plt.title('模型损失变化', fontsize=12)  # 中文标题
        plt.xlabel('训练轮次（Epoch）', fontsize=10)  # 中文X轴标签
        plt.ylabel('损失值', fontsize=10)  # 中文Y轴标签
        plt.legend(fontsize=9)  # 显示图例
        plt.grid(alpha=0.3)  # 添加网格，便于查看数值

        # 调整子图间距，避免标签重叠
        plt.tight_layout()
        # 保存图像（路径可自定义，如需要指定路径可改为绝对路径）
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')  # dpi=300确保图像清晰
        print("📊 训练曲线已保存至: training_history.png")
        plt.show()  # 显示图像（如果不需要显示，可注释掉）


# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":
    # 初始化训练器（初始化时会自动配置中文显示）
    trainer = WeldingDefectTrainer()

    try:
        # 训练模型（指定训练集和验证集路径）
        history = trainer.train(
            train_dir=r"C:\Users\huanglei\PycharmProjects\PythonProject1\augmented_train_data",
            val_dir=r"C:\Users\huanglei\PycharmProjects\PythonProject1\augmented_val_data"
        )
        print("🎉 模型训练完成！")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        traceback.print_exc()  # 打印详细错误堆栈，便于排查问题  X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),