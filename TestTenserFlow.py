import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# подготовка датасета CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# нормализация
train_images, test_images = train_images / 255.0, test_images / 255.0

# аугументация 
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(train_images)

# определение архитектуры модели
with tf.device('/CPU:0'):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # компиляция модели
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # обучение модели
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=64), epochs=10, validation_data=(test_images, test_labels))

# визуализация
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на тренировочном наборе')
plt.plot(history.history['val_accuracy'], label='Точность на проверочном наборе')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend(loc='lower right')
plt.title('Точность обучения и проверки')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери на тренировочном наборе')
plt.plot(history.history['val_loss'], label='Потери на проверочном наборе')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend(loc='upper right')
plt.title('Потери обучения и проверки')
plt.show()