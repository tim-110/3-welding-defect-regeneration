# welding_feature_extractor.py
import cv2
import numpy as np
from scipy import ndimage
from skimage import feature, filters
import math


class WeldingFeatureExtractor:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self):
        """获取特征名称"""
        names = []

        # 矩特征名称
        moment_types = ['零阶矩', '一阶矩X', '一阶矩Y', '二阶矩XX', '二阶矩YY', '二阶矩XY',
                        '三阶矩XXX', '三阶矩YYY', '三阶矩XXY', '三阶矩XYY', 'Hu矩1', 'Hu矩2',
                        'Hu矩3', 'Hu矩4', 'Hu矩5', 'Hu矩6', 'Hu矩7']

        # 纹理特征名称
        texture_names = ['平均灰度值', '灰度值方差', '平均梯度值', '梯度值方差', '能量值',
                         '对比度', '相关性', '同质性', '熵值']

        names.extend(moment_types)
        names.extend(texture_names)

        return names

    def rgb_to_grayscale(self, image):
        """RGB转灰度图"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.resize(gray, self.img_size)

    def calculate_moments(self, gray_image):
        """计算各阶矩特征"""
        # 二值化处理以便更好地提取缺陷特征
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 计算图像矩
        moments = cv2.moments(binary)

        features = []

        # 1. 零阶矩 (区域面积/总亮度) - 描述缺陷大小
        m00 = moments['m00'] if moments['m00'] != 0 else 1e-10
        features.append(m00)

        # 2. 一阶矩 (重心坐标) - 描述缺陷位置
        m10 = moments['m10'] / m00
        m01 = moments['m01'] / m00
        features.extend([m10, m01])

        # 3. 二阶矩 (形状特征) - 描述缺陷形状轮廓
        mu20 = moments['mu20'] / m00  # X方向离散程度
        mu02 = moments['mu02'] / m00  # Y方向离散程度
        mu11 = moments['mu11'] / m00  # 相关性
        features.extend([mu20, mu02, mu11])

        # 4. 三阶矩 (不对称性) - 描述边缘非对称特征
        mu30 = moments['mu30'] / m00  # X方向偏度
        mu03 = moments['mu03'] / m00  # Y方向偏度
        mu21 = moments['mu21'] / m00  # 混合偏度
        mu12 = moments['mu12'] / m00  # 混合偏度
        features.extend([mu30, mu03, mu21, mu12])

        # 5. Hu矩 (不变矩) - 对旋转、缩放、平移不变的矩
        hu_moments = cv2.HuMoments(moments)
        features.extend([hu_moments[i][0] for i in range(7)])

        return np.array(features)

    def calculate_shape_features(self, gray_image):
        """基于矩的形状分析特征"""
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        moments = cv2.moments(binary)

        m00 = moments['m00'] if moments['m00'] != 0 else 1e-10

        # 形状特征
        mu20 = moments['mu20'] / m00
        mu02 = moments['mu02'] / m00
        mu11 = moments['mu11'] / m00

        # 偏心率和方向
        theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02) if (mu20 - mu02) != 0 else 0

        # 区分裂纹（细长）和气孔（圆形）
        eccentricity = (mu20 + mu02 + math.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)) / \
                       (mu20 + mu02 - math.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2) + 1e-10)

        return eccentricity, theta

    # 在 welding_feature_extractor.py 中，完善 AdvancedFeatureExtractor 类
    class AdvancedFeatureExtractor:
        def __init__(self, img_size=(224, 224)):
            # 可复用 WeldingFeatureExtractor 的初始化逻辑，或自定义高级参数
            self.img_size = img_size

        def extract_advanced_features(self, image):
            """补充实际的高级特征提取逻辑（示例：基于 WeldingFeatureExtractor 扩展）"""
            # 1. 先转灰度图（复用类似逻辑）
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            gray = cv2.resize(gray, self.img_size)

            # 2. 提取“高级特征”（示例：在原有矩/纹理特征基础上，增加边缘检测特征）
            # （1）原有基础特征（可复用 WeldingFeatureExtractor 的方法，或重新实现）
            moments = self._calculate_advanced_moments(gray)
            texture = self._calculate_advanced_texture(gray)

            # （2）新增高级特征（如 Canny 边缘密度）
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (self.img_size[0] * self.img_size[1])  # 边缘占比

            # 3. 合并所有高级特征
            advanced_features = np.concatenate([moments, texture, [edge_density]])
            return advanced_features

        # 补充高级特征所需的辅助方法
        def _calculate_advanced_moments(self, gray):
            # 实现更精细的矩特征（如归一化矩、高阶矩）
            moments = cv2.moments(gray)
            return np.array([moments['m00'], moments['m10'], moments['m01']])  # 示例简化

        def _calculate_advanced_texture(self, gray):
            # 实现更高级的纹理特征（如多尺度GLCM、LBP特征）
            glcm = feature.graycomatrix(gray.astype(np.uint8), [1, 3], [0, np.pi / 4], symmetric=True, normed=True)
            energy = np.mean(feature.graycoprops(glcm, 'energy'))
            entropy = -np.mean(glcm * np.log(glcm + 1e-10))
            return np.array([energy, entropy])  # 示例简化

    def calculate_texture_features(self, gray_image):
        """计算纹理特征"""
        features = []

        # 1. 平均灰度值 - 明暗程度
        mean_intensity = np.mean(gray_image)
        features.append(mean_intensity)

        # 2. 灰度值方差 - 不均匀性
        intensity_variance = np.var(gray_image)
        features.append(intensity_variance)

        # 3. 平均梯度值 - 边缘清晰程度
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        mean_gradient = np.mean(gradient_magnitude)
        features.append(mean_gradient)

        # 4. 梯度值方差 - 边缘稳定程度
        gradient_variance = np.var(gradient_magnitude)
        features.append(gradient_variance)

        # 5. 能量值 - 纹理规则程度
        glcm = feature.graycomatrix(gray_image.astype(np.uint8), [1], [0], symmetric=True, normed=True)
        energy = np.sum(glcm ** 2)
        features.append(energy)

        # 6. 对比度 - 纹理对比程度
        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        features.append(contrast)

        # 7. 相关性 - 纹理相关程度
        correlation = feature.graycoprops(glcm, 'correlation')[0, 0]
        features.append(correlation)

        # 8. 同质性 - 纹理均匀程度
        homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        features.append(homogeneity)

        # 9. 熵值 - 纹理复杂程度
        entropy = -np.sum(glcm * np.log(glcm + 1e-10))
        features.append(entropy)

        return np.array(features)

    def extract_all_features(self, image):
        """提取所有特征"""
        # 转换为灰度图
        gray = self.rgb_to_grayscale(image)

        # 提取矩特征 (17个)
        moment_features = self.calculate_moments(gray)

        # 提取纹理特征 (9个)
        texture_features = self.calculate_texture_features(gray)

        # 提取形状特征 (2个)
        eccentricity, orientation = self.calculate_shape_features(gray)

        # 合并所有特征
        all_features = np.concatenate([
            moment_features,
            texture_features,
            [eccentricity, orientation]  # 添加形状特征
        ])

        return all_features

    def analyze_defect_type(self, features):
        """根据特征分析缺陷类型"""
        # 特征索引（根据特征名称顺序）
        eccentricity_idx = 26  # 偏心率的索引
        area_idx = 0  # 面积的索引
        intensity_var_idx = 18  # 灰度方差的索引

        eccentricity = features[eccentricity_idx]
        area = features[area_idx]
        intensity_variance = features[intensity_var_idx]

        # 基于经验的简单分类规则
        if eccentricity > 5.0 and area < 5000:
            # 高偏心率 + 小面积 → 裂纹
            return "crack", 0.8
        elif eccentricity < 2.0 and intensity_variance > 1000:
            # 低偏心率 + 高方差 → 气孔
            return "porosity", 0.7
        elif area > 8000 and intensity_variance > 800:
            # 大面积 + 中等方差 → 飞溅
            return "spatter", 0.6
        else:
            return "unknown", 0.5

    def get_feature_descriptions(self):
        """获取特征描述"""
        descriptions = {
            # 矩特征描述
            '零阶矩': '缺陷区域面积/总亮度，反映缺陷大小',
            '一阶矩X': '缺陷重心X坐标，反映水平位置',
            '一阶矩Y': '缺陷重心Y坐标，反映垂直位置',
            '二阶矩XX': 'X方向离散程度，反映水平伸展',
            '二阶矩YY': 'Y方向离散程度，反映垂直伸展',
            '二阶矩XY': 'XY相关性，反映形状方向',
            '三阶矩XXX': 'X方向偏度，反映水平不对称性',
            '三阶矩YYY': 'Y方向偏度，反映垂直不对称性',
            'Hu矩1-7': 'Hu不变矩，对旋转缩放平移不变的形状特征',

            # 纹理特征描述
            '平均灰度值': '缺陷区域平均亮度，反映明暗程度',
            '灰度值方差': '灰度分布方差，反映不均匀性',
            '平均梯度值': '平均边缘强度，反映清晰程度',
            '梯度值方差': '梯度分布方差，反映边缘稳定程度',
            '能量值': '纹理能量，反映规则程度',
            '对比度': '纹理对比程度',
            '相关性': '纹理相关程度',
            '同质性': '纹理均匀程度',
            '熵值': '纹理复杂程度',

            # 形状特征
            '偏心率': '形状偏心率，区分裂纹(高)和气孔(低)',
            '方向角': '缺陷主要方向'
        }
        return descriptions


# 使用示例和测试函数
def test_feature_extraction():
    """测试特征提取"""
    extractor = WeldingFeatureExtractor()

    # 创建测试图像（模拟不同缺陷）
    test_images = {
        'crack': np.random.randint(50, 150, (224, 224), dtype=np.uint8),  # 细长裂纹
        'porosity': np.random.randint(100, 200, (224, 224), dtype=np.uint8),  # 圆形气孔
        'spatter': np.random.randint(80, 180, (224, 224), dtype=np.uint8)  # 不规则飞溅
    }

    # 为每种缺陷添加特征形状
    # 裂纹：细长形状
    cv2.line(test_images['crack'], (50, 100), (150, 100), 200, 5)

    # 气孔：圆形
    cv2.circle(test_images['porosity'], (112, 112), 30, 220, -1)

    # 飞溅：不规则形状
    points = np.array([[80, 80], [120, 70], [150, 120], [100, 150], [60, 120]], np.int32)
    cv2.fillPoly(test_images['spatter'], [points], 210)

    print("特征提取测试:")
    print("=" * 60)

    for defect_type, image in test_images.items():
        features = extractor.extract_all_features(image)
        predicted_type, confidence = extractor.analyze_defect_type(features)

        print(f"{defect_type}:")
        print(f"  提取特征数: {len(features)}")
        print(f"  预测类型: {predicted_type} (置信度: {confidence:.2f})")
        print(f"  主要特征: 面积={features[0]:.1f}, 偏心率={features[26]:.2f}")
        print("-" * 40)


def extract_features_from_image(image_path):
    """从图像文件提取特征"""
    extractor = WeldingFeatureExtractor()

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    features = extractor.extract_all_features(image)
    predicted_type, confidence = extractor.analyze_defect_type(features)

    print(f"图像: {os.path.basename(image_path)}")
    print(f"预测缺陷类型: {predicted_type} (置信度: {confidence:.2f})")
    print(f"特征数量: {len(features)}")

    # 显示重要特征
    important_features = {
        '面积(零阶矩)': features[0],
        '重心X': features[1],
        '重心Y': features[2],
        '偏心率': features[26],
        '平均灰度': features[17],
        '灰度方差': features[18]
    }

    print("重要特征值:")
    for name, value in important_features.items():
        print(f"  {name}: {value:.3f}")

    return features


if __name__ == "__main__":
    import os

    # 测试特征提取
    test_feature_extraction()

    print("\n" + "=" * 60)
    print("特征描述:")
    print("=" * 60)

    extractor = WeldingFeatureExtractor()
    descriptions = extractor.get_feature_descriptions()
    for name, desc in list(descriptions.items())[:10]:  # 显示前10个特征描述
        print(f"{name}: {desc}")


class AdvancedFeatureExtractor:
    pass