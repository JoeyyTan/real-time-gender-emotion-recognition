import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

# Define parameters
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
l2_strength = 0.01

# Directories
train_dir = 'C:/Users/LENOVO/Downloads/real-time-emotion-recognition/archive/train'
test_dir = 'C:/Users/LENOVO/Downloads/real-time-emotion-recognition/archive/test'

# Check if TensorFlow sees the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the GPU")
    for gpu in gpus:
        print(f"Device: {gpu}")
else:
    print("No GPU found. TensorFlow is using the CPU")

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.20,
    height_shift_range=0.20,
    brightness_range=[0.80, 1.20],
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT,
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',  
    shuffle=False
)

# Categories for the classes
categories = list(train_generator.class_indices.keys())

# Class Weights Calculation
num_train_samples = {
    'angry': 3964,
    'fear': 4086,
    'happy': 7191,
    'neutral': 4952,
    'sad': 4818
}

class_indices = train_generator.class_indices
labels = []
for class_name in class_indices:
    labels += [class_indices[class_name]] * num_train_samples[class_name]

classes = np.unique(labels)
class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=labels)
class_weights_dict = dict(zip(classes, class_weights))

class_weights_array = np.array([class_weights_dict[class_indices[class_name]] for class_name in sorted(class_indices, key=class_indices.get)])

# Custom weighted loss function
def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weighted_ce = cross_entropy * weights
        return K.sum(weighted_ce, axis=-1)
    return loss

# Model Definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3), kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(l2_strength))
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.000075)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

model.compile(optimizer=optimizer,
              loss=weighted_categorical_crossentropy(class_weights_array),
              metrics=['accuracy'])

# Learning Rate Scheduler and Custom Callback
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.35, patience=5, min_lr=1e-6)

class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        print(f"Learning Rate at the end of epoch {epoch + 1}: {lr:.6f}")

# Train the model
history = model.fit(
    train_generator,
    epochs=200,
    validation_data=validation_generator,
    callbacks=[lr_scheduler, LearningRateLogger()]
)

# Plotting functions
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_accuracy(history)
plot_loss(history)

# Function to create a test data generator for a given directory
def create_test_generator(test_dir):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return test_generator

# Evaluate the model on multiple test sets
def evaluate_test_set(model, test_generator, test_dir, categories):
    test_image_count = sum([len(files) for r, d, files in os.walk(test_dir)])
    steps = int(np.ceil(test_image_count / BATCH_SIZE))
    
    # Evaluate
    test_loss, test_acc = model.evaluate(test_generator, steps=steps)
    print(f'\nTest Directory: {test_dir}')
    print(f'Test Accuracy: {test_acc}, Test Loss: {test_loss}')
    
    # Predictions and classification report
    predictions = model.predict(test_generator, steps=steps)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    print(classification_report(true_classes, predicted_classes, target_names=categories))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {os.path.basename(test_dir)}')
    plt.show()

# Loop through each test directory and evaluate
test_generator = create_test_generator(test_dir)
evaluate_test_set(model, test_generator, test_dir, categories)


# After training your model or loading it
model.save('C:/Users/LENOVO/Downloads/real-time-emotion-recognition/cnn_trial.h5')  # Save it as .h5 format
