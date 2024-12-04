import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import os
import webbrowser

# 下载数据集
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)

# 加载数据集
batch_size = 32
img_height = 180
img_width = 180

# 划分数据集，80%用于训练，20%用于验证
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# 显示图像
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 构建卷积神经网络模型
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 模型摘要输出
model.summary()
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# 使用 TensorBoard 回调
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 训练模型并记录到 TensorBoard
epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback]
)

# 使用 matplotlib 绘制准确度和损失图
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.ylim(0, 4)
#保存图片
plt.savefig('c.png')
plt.show()

# 在代码中打开 TensorBoard
tensorboard_url = "http://localhost:6006/"
webbrowser.open(tensorboard_url)

augmented_train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

augmented_train_generator = augmented_train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training',
    class_mode='sparse',
    seed=123
)

augmented_val_generator = augmented_train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation',
    class_mode='sparse',
    seed=123
)

# 构建卷积神经网络模型
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 使用 TensorBoard 回调
augmented_log_dir = "logs/fit_augmented/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
augmented_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=augmented_log_dir, histogram_freq=1)

# 重新训练模型并记录到 TensorBoard
augmented_epochs = 50
augmented_history = model.fit(
    augmented_train_generator,
    validation_data=augmented_val_generator,
    epochs=augmented_epochs,
    callbacks=[augmented_tensorboard_callback]
)

# 使用 matplotlib 绘制重新训练的准确度和损失图
augmented_acc = augmented_history.history['accuracy']
augmented_val_acc = augmented_history.history['val_accuracy']
augmented_loss = augmented_history.history['loss']
augmented_val_loss = augmented_history.history['val_loss']

augmented_epochs_range = range(augmented_epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(augmented_epochs_range, augmented_acc, label='Training Accuracy (Augmented)')
plt.plot(augmented_epochs_range, augmented_val_acc, label='Validation Accuracy (Augmented)')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy with Data Augmentation')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.plot(augmented_epochs_range, augmented_loss, label='Training Loss (Augmented)')
plt.plot(augmented_epochs_range, augmented_val_loss, label='Validation Loss (Augmented)')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss with Data Augmentation')
plt.ylim(0.2, 1.5)
#保存图片
plt.savefig('d.png')
plt.show()

# 保存模型
model.save('flower_model.h5')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 加载模型
loaded_model = load_model('flower_model.h5')

# 确认模型输入大小
input_shape = loaded_model.input_shape
img_height, img_width = input_shape[1], input_shape[2]

def predict_image(img_path, model, img_height, img_width):
    # 加载图像并预处理
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # 进行预测
    prediction = model.predict(img_array)
    return np.argmax(prediction, axis=1)[0]  # 返回预测类别的索引

# 类别标签映射
class_names = ['rose', 'tulip', 'daisy', 'sunflower', 'dandelion']  # 示例标签

# 设置要预测的图像路径
img_path = 'test_photo.png'

# 预测图像类别
class_idx = predict_image(img_path, loaded_model, img_height, img_width)
class_name = class_names[class_idx]
print(f'Predicted class index: {class_idx}')
print(f'Predicted class name: {class_name}')
