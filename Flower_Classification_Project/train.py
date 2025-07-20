# train.py

import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, RandomFlip, RandomRotation, \
    RandomZoom, Rescaling
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import config  # <--- 在这里修正了！


def create_datasets():
    """使用Keras高阶API从目录创建训练和验证数据集"""
    print("--- Loading Datasets ---")

    train_ds, val_ds = image_dataset_from_directory(
        config.DATA_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="both",
        seed=1337,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode='categorical'
    )

    loaded_class_names = train_ds.class_names
    if loaded_class_names != config.CLASS_NAMES:
        print(f"Warning: Keras loaded classes as {loaded_class_names}, but config expects {config.CLASS_NAMES}.")
        print("Please ensure config.CLASS_NAMES matches the alphabetical order of subdirectories.")

    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.2),
    ])

    normalization_layer = Rescaling(1. / 255)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    print("--- Datasets created successfully ---")
    return train_ds, val_ds


def build_model():
    """构建基于ResNet50的迁移学习模型"""
    print("--- Building Model ---")
    inputs = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))

    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(config.N_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def plot_and_save_history(history):
    """绘制并保存训练和验证曲线"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(config.TRAINING_CURVES_PATH)
    print(f"Training curves saved to '{config.TRAINING_CURVES_PATH}'")
    plt.show()


def evaluate_model(model, validation_dataset):
    """在验证集/测试集上评估模型并保存结果"""
    print("\n--- Evaluating Model ---")

    y_true_one_hot = np.concatenate([y for x, y in validation_dataset], axis=0)
    y_true = np.argmax(y_true_one_hot, axis=1)

    y_pred_proba = model.predict(validation_dataset)
    y_pred = np.argmax(y_pred_proba, axis=1)

    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
    print("\nClassification Report:")
    print(report)
    with open(config.REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Classification report saved to '{config.REPORT_PATH}'")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix saved to '{config.CONFUSION_MATRIX_PATH}'")
    plt.show()


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    train_ds, val_ds = create_datasets()

    model = build_model()

    callbacks = [
        ModelCheckpoint(filepath=config.MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=config.PATIENCE, restore_best_weights=True, verbose=1)
    ]

    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )

    plot_and_save_history(history)

    evaluate_model(model, val_ds)


if __name__ == '__main__':
    main()