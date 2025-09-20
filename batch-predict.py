# batch_predict.py
import os
import cv2
import pandas as pd
from datetime import datetime

# 导入配置
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *


class BatchPredictor:
    def __init__(self):
        from predict import WeldingDefectPredictor
        self.predictor = WeldingDefectPredictor()

    def predict_folder(self, folder_path, output_csv=None):
        """预测文件夹中的所有图像"""
        if not os.path.exists(folder_path):
            print(f"❌ 文件夹不存在: {folder_path}")
            return

        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(folder_path)
                                if f.lower().endswith(ext)])

        if not image_files:
            print("❌ 文件夹中没有图像文件")
            return

        print(f"找到 {len(image_files)} 张图像")

        results = []
        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(folder_path, img_file)
            print(f"处理 {i}/{len(image_files)}: {img_file}")

            predicted_class, confidence, all_probs = self.predictor.predict_image(img_path)

            if predicted_class:
                result = {
                    '文件名': img_file,
                    '预测结果': predicted_class,
                    '置信度': confidence,
                    '气孔概率': all_probs[0],
                    '裂纹概率': all_probs[1],
                    '飞溅概率': all_probs[2],
                    '处理时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(result)
                print(f"  ✅ {predicted_class} ({confidence:.2%})")
            else:
                print(f"  ❌ 预测失败")

        # 保存结果到CSV
        if results and output_csv:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"✅ 结果已保存到: {output_csv}")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='批量预测焊接缺陷')
    parser.add_argument('folder', help='包含图像的文件夹路径')
    parser.add_argument('--output', '-o', help='输出CSV文件路径', default='prediction_results.csv')

    args = parser.parse_args()

    predictor = BatchPredictor()
    results = predictor.predict_folder(args.folder, args.output)

    if results:
        print(f"\n🎉 批量预测完成！共处理 {len(results)} 张图像")