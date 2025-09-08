# %% [markdown]
# # Advanced TensorFlow / Keras Jupyter Notebook
# 
# **What this notebook contains (end-to-end, advanced):**
# - Reproducible environment setup and imports
# - Load dataset (MNIST) + optional instructions to load a structured CSV
# - Advanced preprocessing using `tf.data` pipeline (caching, prefetching, augmentation)
# - Build a robust CNN using Keras Functional API with BatchNorm + Dropout + weight decay
# - Training with callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
# - Learning-rate schedules and optimizers
# - Hyperparameter tuning example using Keras Tuner (Hyperband)
# - Evaluation: accuracy/loss plots, confusion matrix, classification report
# - Tips to export the notebook to .ipynb and how to run on GPU
# 
# **IMPORTANT:** This file is formatted as a Python script with cell markers ("# %%") so you can open
# it directly in Jupyter or VS Code ("Run Cell"). Do **not** run it here — it is intended for a Jupyter environment.

# %% [markdown]
# ## 1. Environment & Imports
# Set up imports, seeds, and optional GPU checks.

# %%
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# For advanced hyperparameter tuning
try:
    import kerastuner as kt
except Exception:
    # Newer package name
    import keras_tuner as kt

# For evaluation
from sklearn.metrics import classification_report, confusion_matrix

# Reproducibility (note: full determinism can be platform-dependent)
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print('TensorFlow version:', tf.__version__)

# Optional: GPU info
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('GPUs detected:', len(physical_devices))
    except Exception as e:
        print('GPU config error:', e)
else:
    print('No GPU detected — training will be slower on CPU.')

# %% [markdown]
# ## 2. Load dataset (MNIST) and create `tf.data` pipelines
# We'll use MNIST as a concrete example (handwritten digits). There's also a short snippet showing how to
# replace with a structured CSV dataset if desired.

# %%
# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Convert to float32 and expand channels
x_train = x_train.astype('float32')[..., np.newaxis]  # shape (N, 28, 28, 1)
x_test = x_test.astype('float32')[..., np.newaxis]

# Combine a small validation split from training
VAL_SPLIT = 0.1
num_val = int(len(x_train) * VAL_SPLIT)

x_val = x_train[:num_val]
y_val = y_train[:num_val]

x_train = x_train[num_val:]
y_train = y_train[num_val:]

print('Train shape:', x_train.shape, 'Val shape:', x_val.shape, 'Test shape:', x_test.shape)

# Optional: show a few samples
# %%
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(x_train[i].squeeze(), cmap='gray')
    ax.set_title(f'label={y_train[i]}')
    ax.axis('off')
plt.show()

# %% [markdown]
# ### Alternative: loading a structured CSV dataset (brief example)
# If you want to load a CSV, do the following (uncomment and adapt):
# ```python
# df = pd.read_csv('your_file.csv')
# # basic preprocessing, fillna, encoding etc.
# X = df.drop(columns=['target']).values
# y = df['target'].values
# # train_test_split and then convert to tf.data.Dataset.from_tensor_slices
# ```

# %% [markdown]
# ## 3. Preprocessing functions & data augmentation
# Build robust `tf.data` pipelines including normalization, augmentation, caching and prefetching.

# %%
BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (28, 28)  # for MNIST
NUM_CLASSES = 10

# Normalization layer (keeps dtype float32)
normalization_layer = layers.Rescaling(1.0 / 255)

# Simple augmentation pipeline using Keras preprocessing layers
augmentation_layers = keras.Sequential(
    [
        layers.RandomRotation(0.08, seed=SEED),
        layers.RandomTranslation(0.06, 0.06, seed=SEED),
        layers.RandomZoom(0.06, seed=SEED),
    ], name='augmentation')

# Create function to build tf.data.Dataset

def make_dataset(images, labels, training=False, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(10_000, seed=SEED)
    ds = ds.map(lambda x, y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(x_train, y_train, training=True)
val_ds = make_dataset(x_val, y_val, training=False)
test_ds = make_dataset(x_test, y_test, training=False)

# %% [markdown]
# ## 4. Model design — Keras Functional API (advanced CNN)
# We'll implement a compact but powerful convnet with weight decay, BatchNorm, and residual blocks.

# %%
from tensorflow.keras import regularizers

def conv_block(x, filters, kernel_size=3, stride=1, weight_decay=1e-4):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def residual_block(x, filters, weight_decay=1e-4):
    shortcut = x
    x = conv_block(x, filters, 3, 1, weight_decay)
    x = conv_block(x, filters, 3, 1, weight_decay)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def build_model(input_shape=(28, 28, 1), num_classes=NUM_CLASSES, weight_decay=1e-4, dropout_rate=0.3):
    inputs = keras.Input(shape=input_shape)
    x = conv_block(inputs, 32, kernel_size=3, weight_decay=weight_decay)
    x = conv_block(x, 64, kernel_size=3, weight_decay=weight_decay)
    x = layers.MaxPooling2D(2)(x)

    x = residual_block(x, 64, weight_decay)
    x = layers.MaxPooling2D(2)(x)

    x = conv_block(x, 128, kernel_size=3, weight_decay=weight_decay)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='mnist_resnet_like')
    return model

model = build_model()
model.summary()

# %% [markdown]
# ## 5. Compile: optimizer, loss, metrics, and advanced LR schedule
# We'll show examples of Adam with weight decay (via kernel_regularizer above) and
# a Cosine decay schedule. Also add metrics for top-1 accuracy.

# %%
INITIAL_LR = 1e-3
EPOCHS = 30

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=INITIAL_LR,
    first_decay_steps=10 * (len(x_train) // BATCH_SIZE)  # rough steps for 10 epochs
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %% [markdown]
# ## 6. Callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)

# %%
logdir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_mnist_model.h5', monitor='val_loss', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1),
]

# %% [markdown]
# ## 7. Train the model
# (This cell will actually run training if executed in a Jupyter environment.)

# %%
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# %% [markdown]
# ## 8. Plot accuracy & loss curves

# %%
plt.figure(figsize=(12, 5))
# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')
plt.show()

# %% [markdown]
# ## 9. Evaluate on test set and show classification report + confusion matrix

# %%
test_results = model.evaluate(test_ds)
print('Test loss, Test accuracy:', test_results)

# Get predictions for confusion matrix

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# To get the true labels from test_ds (batched), build an array
true_labels = np.concatenate([y for x, y in test_ds], axis=0)

print('\nClassification Report:')
print(classification_report(true_labels, y_pred))

# Confusion matrix
cm = confusion_matrix(true_labels, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %% [markdown]
# ## 10. Advanced: Hyperparameter tuning with Keras Tuner (Hyperband)
# We'll tune: learning rate, dropout_rate, weight_decay, and number of filters in first conv.

# %%

def build_model_tuner(hp):
    weight_decay = hp.Float('weight_decay', 1e-5, 1e-3, sampling='log')
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    first_filters = hp.Choice('first_filters', [16, 32, 48])
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(first_filters, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)

    x = residual_block(x, first_filters, weight_decay)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Setup tuner (note: this can consume significant compute)
# tuner = kt.Hyperband(
#     build_model_tuner,
#     objective='val_accuracy',
#     max_epochs=20,
#     factor=3,
#     directory='kt_dir',
#     project_name='mnist_tuning'
# )

# Quick example to run tuner (uncomment to run in Jupyter):
# tuner.search(train_ds, validation_data=val_ds, epochs=20)
# best_hp = tuner.get_best_hyperparameters(1)[0]
# print(best_hp.values)

# %% [markdown]
# ## 11. Extra: Batch size and learning-rate sweep (manual grid search)
# A simple manual loop showing how you could search different batch sizes and learning rates
# (useful when KerasTuner is not desired). This is a light-weight example and retrains the
# model multiple times, so run with caution.

# %%
# batch_sizes = [64, 128]
# lrs = [1e-3, 5e-4]
# best_val_acc = 0
# best_config = None
# for bs in batch_sizes:
#     for lr in lrs:
#         print('\nRunning config: batch_size=', bs, ' lr=', lr)
#         # rebuild datasets (must recreate batches)
#         train_ds_local = make_dataset(x_train, y_train, training=True, batch_size=bs)
#         val_ds_local = make_dataset(x_val, y_val, training=False, batch_size=bs)
#         model_local = build_model()
#         model_local.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
#                             loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#         hist = model_local.fit(train_ds_local, validation_data=val_ds_local, epochs=8)
#         val_acc = max(hist.history['val_accuracy'])
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_config = dict(batch_size=bs, lr=lr)
# print('Best config:', best_config, 'best_val_acc=', best_val_acc)

# %% [markdown]
# ## 12. Save & load model
# Save the best model (ModelCheckpoint used earlier), then load it with `keras.models.load_model`.

# %%
# model.save('final_mnist_model')
# loaded = keras.models.load_model('final_mnist_model')

# %% [markdown]
# ## 13. Notes & tips
# - For structured datasets, replace the image pipelines by feature preprocessing (normalization, categorical encoding)
#   and use `tf.data` pipelines with `from_tensor_slices` or `from_generator`.
# - For larger image tasks, prefer `tf.keras.applications` pretrained backbones (EfficientNet, ResNet) with fine-tuning.
# - When doing hyperparameter search, use early stopping and limit epochs to save compute.
# - Use mixed precision (`tf.keras.mixed_precision.set_global_policy('mixed_float16')`) for NVIDIA GPUs to speed up training.

# %% [markdown]
# ## 14. If you want an `.ipynb` file
# If you'd like this code packaged as a `.ipynb` notebook file, I can export it for you — tell me and I'll produce a downloadable notebook.

# %% [markdown]
# **End of notebook**
