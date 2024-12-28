import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

train_csv_file = "C:/Users/user/Desktop/capstone design/cell/boneage-training-dataset.csv"
test_csv_file = "C:/Users/user/Desktop/capstone design/cell/boneage-test-dataset.csv"
train_image_dir = "C:/Users/user/Desktop/capstone design//boneage-training-dataset/"
test_image_dir = "C:/Users/user/Desktop/capstone design//boneage-test-dataset/"

model_dir = 'C:/Users/user/Desktop/capstone design/CNN_saved_models/'
os.makedirs(model_dir, exist_ok=True)

batch_size = 32
img_height = 224
img_width = 224
initial_lr = 0.0001
epochs = 100

print("데이터를 로드하고 전처리합니다.")

train_df = pd.read_csv(train_csv_file)
train_df['image_path'] = train_df['id'].apply(lambda x: os.path.join(train_image_dir, f"{x}.png"))
train_df['gender'] = train_df['male'].map({True: 1, False: 0})
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

test_df = pd.read_csv(test_csv_file)
test_df['image_path'] = test_df['Case ID'].apply(lambda x: os.path.join(test_image_dir, f"{x}.png"))
test_df['gender'] = test_df['Sex'].map({'M': 1, 'F': 0})

def preprocess_image(image_path, gender, bone_age=None):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = image / 255.0
    gender = tf.cast(gender, tf.float32)
    gender = tf.expand_dims(gender, -1)
    if bone_age is not None:
        bone_age = tf.cast(bone_age, tf.float32)
        return {'image': image, 'gender': gender}, bone_age
    else:
        return {'image': image, 'gender': gender}

def create_dataset(df, batch_size=32, shuffle=True, is_training=True):
    paths = df['image_path'].values
    genders = df['gender'].values.astype(np.float32)
    if is_training and 'boneage' in df.columns:
        bone_ages = df['boneage'].values.astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((paths, genders, bone_ages))
        dataset = dataset.map(lambda x, y, z: preprocess_image(x, y, z), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((paths, genders))
        dataset = dataset.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle and is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_df, batch_size=batch_size, shuffle=True, is_training=True)
val_dataset = create_dataset(val_df, batch_size=batch_size, shuffle=False, is_training=True)
test_dataset = create_dataset(test_df, batch_size=batch_size, shuffle=False, is_training=False)

print("커스텀 CNN 회귀 모델을 생성합니다.")

input_image = layers.Input(shape=(img_height, img_width, 3), name='image')
input_gender = layers.Input(shape=(1,), dtype=tf.float32, name='gender')
gender_dense = layers.Dense(16, activation='relu')(input_gender)

x = layers.Conv2D(32, (3, 3), activation='relu')(input_image)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(256, (3, 3), activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Concatenate()([x, gender_dense])
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='linear')(x)
model = Model(inputs={'image': input_image, 'gender': input_gender}, outputs=output)
model.summary()

optimizer = Adam(learning_rate=initial_lr)
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])

checkpoint = ModelCheckpoint(
    filepath=os.path.join(
        model_dir, 'cnn_regression_model_epoch_{epoch:02d}.keras'),
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=False,
    mode='min'
)
best_val_checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_dir, 'cnn_regression_best_val_model.keras'),
    monitor='val_mae',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
callbacks = [early_stopping, reduce_lr, checkpoint, best_val_checkpoint]

print("모델 훈련을 시작합니다.")
history = None

try:
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    print("학습이 완료되었습니다.")
except KeyboardInterrupt:
    print("학습이 중단되었습니다.")
except Exception as e:
    print(f"학습 중 오류 발생: {e}")

plt.figure(figsize=(8, 6))
if history is not None:
    plt.plot(history.history['mae'], label='훈련 MAE')
    plt.plot(history.history['val_mae'], label='검증 MAE')
    plt.title('훈련 및 검증 MAE')
    plt.xlabel('에포크')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

plt.figure(figsize=(8, 6))
if history is not None:
    plt.plot(history.history['loss'], label='훈련 손실')
    plt.plot(history.history['val_loss'], label='검증 손실')
    plt.title('훈련 및 검증 손실')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.legend()
    plt.show()

print("테스트 데이터에 대한 예측을 수행합니다.")
y_pred = []

try:
    for batch in test_dataset:
        inputs = batch
        preds = model.predict(inputs)
        y_pred.extend(preds.flatten())

    y_pred = np.array(y_pred)
    y_pred = np.round(y_pred).astype(int)
    test_df['boneage_prediction'] = y_pred
    prediction_csv_path = 'C:/Users/user/Desktop/capstone design/boneage_test_predictions.csv'
    test_df[['Case ID', 'boneage_prediction']].to_csv(prediction_csv_path, index=False)
    print(f"예측 결과를 '{prediction_csv_path}' 파일로 저장하였습니다.")
except Exception as e:
    print(f"예측 중 오류 발생: {e}")
