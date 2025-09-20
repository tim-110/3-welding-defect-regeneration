# enhanced_data_augmentation.py
import imgaug.augmenters as iaa


def get_advanced_augmenter():
    """获取高级数据增强序列"""
    return iaa.Sequential([
        # 几何变换
        iaa.Fliplr(0.5),
        iaa.Flipud(0.3),
        iaa.Affine(
            rotate=(-30, 30),
            scale=(0.8, 1.2),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
        ),

        # 颜色变换
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0))),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),
        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),
        iaa.Sometimes(0.5, iaa.LinearContrast((0.8, 1.2))),
        iaa.Sometimes(0.3, iaa.Grayscale(alpha=(0.0, 1.0))),

        # 弹性变换
        iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.25)),
    ], random_order=True)