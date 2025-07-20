# predict.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import config


def predict_single_image(image_path, model):
    """加载图片，预处理并进行预测"""
    try:
        img = image.load_img(image_path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # 创建一个批次
        img_array /= 255.0  # 归一化

        predictions = model.predict(img_array)

        # 获取预测结果
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = config.CLASS_NAMES[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        print(f"\nPrediction for '{image_path}':")
        print(f"-> Class: {predicted_class_name}")
        print(f"-> Confidence: {confidence:.2f}%")

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # 检查是否提供了图片路径作为参数
    if len(sys.argv) != 2:
        print("\nUsage: python predict.py <path_to_image.jpg>")
        return

    image_path = sys.argv[1]

    # 加载模型
    try:
        print(f"Loading model from {config.MODEL_PATH}...")
        model = tf.keras.models.load_model(config.MODEL_PATH)
        predict_single_image(image_path, model)
    except IOError:
        print(f"Error: Could not load model from {config.MODEL_PATH}. Have you trained it first by running train.py?")


if __name__ == '__main__':
    main()