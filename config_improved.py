# config_improved.py
import os

# ==================== 基础路径配置 ====================
BASE_DIR = r"D:\identification of welding defects"
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 确保目录存在
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ==================== 模型文件路径 ====================
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "welding_defect_model.keras")  # 改用keras格式
SCALER_SAVE_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")

# ==================== 缺陷类别配置 ====================
CLASS_NAMES = ['porosity', 'crack', 'spatter']

# ==================== 增强的训练参数 ====================
IMG_SIZE = (224, 224)
AUGMENT_FACTOR = 15  # 增加增强因子
MAX_SAMPLES_PER_CLASS = 200

# 训练参数
BATCH_SIZE = 16
EPOCHS = 200  # 增加训练轮数
VALIDATION_SPLIT = 0.2

# 回调函数参数
EARLY_STOPPING_PATIENCE = 20
REDUCE_LR_PATIENCE = 10
MIN_LEARNING_RATE = 1e-6

# 学习率调度
INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.95

# ==================== 特征提取参数 ====================
# 图像预处理参数
BLUR_KERNEL_SIZE = (5, 5)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# 特征维度
FEATURE_DIM = 23  # 增加特征维度