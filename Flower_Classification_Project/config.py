# config.py
import os

# 数据集主目录
DATA_DIR = os.path.join('data', 'input_data')

# 模型训练相关配置
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 15
VALIDATION_SPLIT = 0.2
PATIENCE = 5  # <--- 在这里添加这一行！

# 类别配置
CLASS_NAMES = ['dandelion', 'roses', 'sunflowers', 'tulips']
N_CLASSES = len(CLASS_NAMES)

# 文件保存路径配置
OUTPUT_DIR = 'output'
MODEL_PATH = os.path.join('saved_models', 'flower_classifier_best_model.h5')
TRAINING_CURVES_PATH = os.path.join(OUTPUT_DIR, 'training_curves.png')
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
REPORT_PATH = os.path.join(OUTPUT_DIR, 'classification_report.txt')