# 焊接缺陷识别项目 - 整合版 config.py
import os

# ==================================================
# 1. 基础路径配置（统一管理所有核心目录，兼容原两版配置）
# ==================================================
# 项目根目录（所有子路径的基准，修改此处可全局调整）
BASE_DIR = r"D:\identification of welding defects"

# 数据相关路径（覆盖原两版的原始数据、处理后数据定义）
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, "data")  # 原始数据集路径（与原版一致）
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")  # 处理后数据路径（与原版一致）

# 增强数据路径（补充原版的增强验证集路径，完善数据划分）
AUGMENTED_TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "augmented", "train")  # 增强训练集（与原版一致）
AUGMENTED_VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "augmented", "val")  # 新增增强验证集（优化版补充）
AUGMENTED_TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "augmented", "test")  # 增强测试集（与原版一致）

# 模型与标准化器路径（兼容两版格式，优先用keras格式，保留h5格式备选）
MODELS_DIR = os.path.join(BASE_DIR, "models")  # 模型保存根目录（与原版一致）
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "welding_defect_model.h5") # 优化版的keras格式（推荐）
MODEL_SAVE_PATH_H5 = os.path.join(MODELS_DIR, "welding_defect_model.h5")  # 原版的h5格式（备选）
SCALER_SAVE_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")  # 特征标准化器路径（两版一致）

# 自动创建必要目录（避免手动创建，防止路径不存在报错）
required_dirs = [PROCESSED_DATA_DIR, MODELS_DIR, AUGMENTED_TRAIN_DIR, AUGMENTED_VAL_DIR, AUGMENTED_TEST_DIR]
for dir_path in required_dirs:
    os.makedirs(dir_path, exist_ok=True)

# ==================================================
# 2. 缺陷类别配置（两版完全一致，直接保留）
# ==================================================
CLASS_NAMES = ['porosity', 'crack', 'spatter']  # 缺陷类别：气孔、裂纹、飞溅
CLASS_TO_LABEL = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}  # 新增类别-标签映射（便于代码复用）
LABEL_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASS_NAMES)}  # 新增标签-类别映射（便于结果解析）
NUM_CLASSES = len(CLASS_NAMES)  # 类别总数（补充参数，用于模型输出层定义）

# ==================================================
# 3. 训练核心参数（融合两版配置，取优化版的增强参数，保留原版备选）
# ==================================================
# 图像基础参数（两版一致，直接保留）
IMG_SIZE = (224, 224)  # 图像预处理尺寸（宽×高）

# 数据增强参数（融合两版：优化版的增强因子更高，保留原版参数备选）
AUGMENT_FACTOR = 15  # 优化版参数（增加增强数量，提升泛化能力）
AUGMENT_FACTOR_ORIGINAL = 5  # 原版参数（备选，避免生成过多数据时使用）
MAX_SAMPLES_PER_CLASS = 200  # 优化版参数（增加单类最大样本数，适配更多数据）
MAX_SAMPLES_PER_CLASS_ORIGINAL = 50  # 原版参数（备选，样本量较少时使用）

# 训练基础参数（优化版参数更全面，补充原版缺失项）
BATCH_SIZE = 16  # 两版一致，直接保留
EPOCHS = 200  # 优化版参数（增加训练轮次，配合早停策略避免过拟合）
EPOCHS_ORIGINAL = 50  # 原版参数（备选，快速测试时使用）
VALIDATION_SPLIT = 0.2  # 优化版新增（训练集划分验证集的比例，无独立验证集时使用）

# 学习率配置（优化版新增，完善训练策略）
INITIAL_LEARNING_RATE = 0.001  # 初始学习率
LEARNING_RATE_DECAY = 0.95  # 学习率衰减因子（每轮衰减）
MIN_LEARNING_RATE = 1e-6  # 最小学习率（避免衰减过低）

# 回调函数参数（优化版新增，防止过拟合、动态调整学习率）
EARLY_STOPPING_PATIENCE = 20  # 早停耐心值（连续20轮验证指标无提升则停止）
REDUCE_LR_PATIENCE = 10  # 学习率衰减耐心值（连续10轮无提升则衰减）

# ==================================================
# 4. 图像预处理与特征提取参数（优化版新增，补充特征相关配置）
# ==================================================
# 图像预处理参数（优化版新增，用于图像去噪、增强细节）
BLUR_KERNEL_SIZE = (5, 5)  # 高斯模糊核大小（去噪）
BILATERAL_D = 9  # 双边滤波直径（保留边缘的同时去噪）
BILATERAL_SIGMA_COLOR = 75  # 双边滤波颜色标准差
BILATERAL_SIGMA_SPACE = 75  # 双边滤波空间标准差

# 特征维度配置（融合两版：保留优化版的23维特征，补充28维特征备选）
FEATURE_DIM = 23  # 优化版特征维度（按优化版提取逻辑使用）
FEATURE_DIM_ALTERNATIVE = 28  # 备选特征维度（按原28特征逻辑使用时切换）

# ==================================================
# 5. 调试与辅助配置（原版新增，保留路径检查功能）
# ==================================================
DEBUG_MODE = True  # 调试模式（True：打印详细日志；False：仅打印关键信息）


def print_config_summary():
    """整合版路径与关键参数检查：打印所有重要配置，便于启动时验证"""
    print("=" * 60)
    print("📋 焊接缺陷识别 - 配置摘要")
    print("=" * 60)

    # 1. 关键路径检查
    print("\n【1. 关键路径验证】")
    key_paths = [
        ("原始数据路径", ORIGINAL_DATA_PATH),
        ("增强训练集路径", AUGMENTED_TRAIN_DIR),
        ("增强验证集路径", AUGMENTED_VAL_DIR),
        ("增强测试集路径", AUGMENTED_TEST_DIR),
        ("模型保存目录", MODELS_DIR),
        ("模型文件（keras格式）", MODEL_SAVE_PATH),
        ("特征标准化器", SCALER_SAVE_PATH)
    ]
    for path_name, path in key_paths:
        exists = os.path.exists(path) if not path.endswith((".keras", ".h5", ".pkl")) else True
        status = "✅ 正常" if exists else "❌ 不存在（需检查）"
        print(f"   • {path_name}：{path} → {status}")

    # 2. 核心训练参数
    print("\n【2. 核心训练参数】")
    print(f"   • 图像尺寸：{IMG_SIZE[0]}×{IMG_SIZE[1]} | 缺陷类别：{CLASS_NAMES}（共{NUM_CLASSES}类）")
    print(f"   • 批次大小：{BATCH_SIZE} | 最大训练轮次：{EPOCHS}（备选：{EPOCHS_ORIGINAL}）")
    print(f"   • 数据增强因子：{AUGMENT_FACTOR}（备选：{AUGMENT_FACTOR_ORIGINAL}）")
    print(f"   • 当前特征维度：{FEATURE_DIM}（备选：{FEATURE_DIM_ALTERNATIVE}）")

    # 3. 调试模式
    print("\n【3. 调试配置】")
    print(f"   • 调试模式：{'开启' if DEBUG_MODE else '关闭'}")
    print(f"   • 早停策略：耐心值{EARLY_STOPPING_PATIENCE} | 学习率衰减：耐心值{REDUCE_LR_PATIENCE}")

    print("\n" + "=" * 60)
    print("⚠️  提示：若路径标记'不存在'，请检查数据集位置或手动创建对应目录！")
    print("=" * 60)



#添加新的配置和修改
CLASS_NAMES = ['正常', '裂纹', '气孔', '夹渣', '未焊透', '未熔合']

# 启动时自动打印配置摘要（便于快速排查配置问题）
if __name__ == "__main__":
    print_config_summary()