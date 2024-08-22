import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input, Conv2D, Add, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import os
import matplotlib
matplotlib.use('Agg')  # หรือ 'Qt5Agg'
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Paths to directories
train_dir = 'D:/Project-Fertilizes_Egg/Dataset/datayolov8/crop/train'
validate_dir = 'D:/Project-Fertilizes_Egg/Dataset/datayolov8/crop/valid'
test_dir = 'D:/Project-Fertilizes_Egg/Dataset/datayolov8/crop/test'

# Parameters
batch_size = 32
image_size = (224, 224)
input_shape = (224, 224, 3)
num_classes = 1  # Binary classification

# Custom Data Generator with Augmentation
class AugmentedEggDataGenerator(Sequence):
    def __init__(self, directory, batch_size, image_size, shuffle=True, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.datagen = ImageDataGenerator(
            rotation_range=30,  # Increased rotation
            width_shift_range=0.2,  # Increased shift
            height_shift_range=0.2,
            shear_range=0.2,  # Increased shear
            zoom_range=0.2,  # Increased zoom
            horizontal_flip=True,
            vertical_flip=True,  # Added vertical flip
            fill_mode='nearest'
        )
        self.image_files, self.labels = self._load_dataset()
        self.on_epoch_end()

    def _load_dataset(self):
        image_files = []
        labels = []
        for class_dir in os.listdir(self.directory):
            class_path = os.path.join(self.directory, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(class_path, img_file))
                        labels.append(1 if class_dir == 'FER' else 0)
        return image_files, labels

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_files, batch_labels)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.image_files, self.labels))
            np.random.shuffle(combined)
            self.image_files, self.labels = zip(*combined)

    def __data_generation(self, batch_files, batch_labels):
        X = np.empty((self.batch_size, *self.image_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, 1), dtype=np.float32)

        for i, file in enumerate(batch_files):
            img = load_img(file, target_size=self.image_size)
            img_array = img_to_array(img) / 255.0
            if self.augment:
                img_array = self.datagen.random_transform(img_array)
            X[i,] = img_array
            y[i,] = batch_labels[i]

        return X, y

# Create data generators
train_data = AugmentedEggDataGenerator(train_dir, batch_size, image_size, augment=True)
val_data = AugmentedEggDataGenerator(validate_dir, batch_size, image_size)
test_data = AugmentedEggDataGenerator(test_dir, batch_size, image_size, shuffle=False)

# Check data distribution
def check_data_distribution(directory):
    data_counts = {'FER': 0, 'INF': 0}
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            data_counts[class_dir] = len(os.listdir(class_path))
    return data_counts

train_counts = check_data_distribution(train_dir)
val_counts = check_data_distribution(validate_dir)
test_counts = check_data_distribution(test_dir)

print(f'Training data distribution: {train_counts}')
print(f'Validation data distribution: {val_counts}')
print(f'Test data distribution: {test_counts}')

# Custom ResNet18 model
def resnet_block(inputs, num_filters, kernel_size=3, strides=1, activation='relu'):
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x

def resnet_layer(inputs, num_filters, strides=1, use_activation=True):
    x = resnet_block(inputs, num_filters, strides=strides, activation=None)
    if use_activation:
        x = ReLU()(x)
    return x

def resnet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs, 64)
    x = Dropout(0.2)(x)
    x = resnet_layer(x, 64)

    x = resnet_layer(x, 128, strides=2)
    x = Dropout(0.3)(x)
    x = resnet_layer(x, 128)

    x = resnet_layer(x, 256, strides=2)
    x = Dropout(0.3)(x)
    x = resnet_layer(x, 256)

    x = resnet_layer(x, 512, strides=2)
    x = Dropout(0.4)(x)
    x = resnet_layer(x, 512)

    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='sigmoid', kernel_regularizer=l2(0.001))(x)

    model = Model(inputs, outputs)
    return model

model = resnet18(input_shape, num_classes)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.5

lr_scheduler = LearningRateScheduler(scheduler)

checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=0.00001, mode='min')

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_data.labels), y=np.array(train_data.labels))
class_weights = {i : class_weights[i] for i in range(len(class_weights))}

history = model.fit(
    train_data,
    epochs=50,
    validation_data=val_data,
    class_weight=class_weights,
    callbacks=[early_stopping, lr_scheduler, checkpoint, reduce_lr]
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('accuracy_plot.png')

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('loss_plot.png')

# Evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Predict on the entire test dataset and count fertilized vs infertile predictions
y_true = test_data.labels
y_pred_probs = model.predict(test_data)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print(classification_report(y_true, y_pred, target_names=['INF', 'FER']))

conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(['INF', 'FER']))
plt.xticks(tick_marks, ['INF', 'FER'], rotation=45)
plt.yticks(tick_marks, ['INF', 'FER'])

thresh = conf_matrix.max() / 2
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment='center',
             color='white' if conf_matrix[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
