import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False



normal_dir = "C:/Users/user/Desktop/capstone design/flatfoot/notpesplanus"


flatfoot_dir = "C:/Users/user/Desktop/capstone design/flatfoot/pesplanus"


model_dir = 'C:/Users/user/Desktop/capstone design/CNN_flatfoot_saved_models/'
os.makedirs(model_dir, exist_ok=True)


batch_size = 32
img_height = 224
img_width = 224
initial_lr = 0.0001
epochs = 50


print("데이터를 로드하고 전처리합니다.")


image_paths = []
labels = []


normal_image_paths = []
for filename in os.listdir(normal_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        normal_image_paths.append(os.path.join(normal_dir, filename))
        labels.append(0)  


flatfoot_image_paths = []
for filename in os.listdir(flatfoot_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        flatfoot_image_paths.append(os.path.join(flatfoot_dir, filename))
        labels.append(1) 


image_paths = normal_image_paths + flatfoot_image_paths
labels = [0]*len(normal_image_paths) + [1]*len(flatfoot_image_paths)


data_df = pd.DataFrame({
    'image_path': image_paths,
    'label': labels
})


total_images = len(data_df)
print(f"전체 이미지 개수: {total_images}장")


num_normal = len(normal_image_paths)
num_flatfoot = len(flatfoot_image_paths)
print(f" - 정상 이미지 개수: {num_normal}장")
print(f" - 평발 이미지 개수: {num_flatfoot}장")


data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)


train_df, temp_df = train_test_split(data_df, test_size=0.3, random_state=42, stratify=data_df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])


print(f"\n데이터 분할 결과:")
print(f" - 학습 세트: {len(train_df)}장")
print(f" - 검증 세트: {len(val_df)}장")
print(f" - 테스트 세트: {len(test_df)}장")


print("\n학습 세트 클래스 분포:")
print(train_df['label'].value_counts())

print("\n검증 세트 클래스 분포:")
print(val_df['label'].value_counts())

print("\n테스트 세트 클래스 분포:")
print(test_df['label'].value_counts())


def preprocess_image(image_path, label):
   
    image_contents = tf.io.read_file(image_path)
    try:
        image = tf.image.decode_jpeg(image_contents, channels=3)
        image.set_shape([None, None, 3])
    except Exception as e:
        tf.print("이미지 로드 실패:", image_path)
        raise e
    
    
    file_extension = tf.strings.lower(tf.strings.split(image_path, '.')[-1])
    
    
    def decode_jpeg():
        return tf.image.decode_jpeg(image_contents, channels=3)
    
    def decode_png():
        return tf.image.decode_png(image_contents, channels=3)
    
    
    image = tf.cond(
        tf.math.logical_or(tf.equal(file_extension, 'jpg'), tf.equal(file_extension, 'jpeg')),
        decode_jpeg,
        decode_png
    )
    
    
    image.set_shape([None, None, 3])
    
  
    image = tf.image.resize(image, [img_height, img_width])
    

    image = image / 255.0
    

    label = tf.cast(label, tf.float32)
    return image, label

def create_dataset(df, batch_size=32, shuffle=True, augment=False):
    paths = df['image_path'].values
    labels = df['label'].values.astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    if augment:
        def augment_image(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
            return image, label
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


train_dataset = create_dataset(train_df, batch_size=batch_size, shuffle=True, augment=True)
val_dataset = create_dataset(val_df, batch_size=batch_size, shuffle=False, augment=False)
test_dataset = create_dataset(test_df, batch_size=batch_size, shuffle=False, augment=False)


print("커스텀 CNN 분류 모델을 생성합니다.")


input_image = layers.Input(shape=(img_height, img_width, 3), name='image')

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

x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_image, outputs=output)

model.summary()

optimizer = Adam(learning_rate=initial_lr)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    filepath=os.path.join(
        model_dir, 'cnn_classification_model_epoch_{epoch:02d}.keras'),
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=False,
    mode='min'
)

best_val_checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_dir, 'cnn_classification_best_val_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
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
    plt.plot(history.history['accuracy'], label='훈련 정확도')
    plt.plot(history.history['val_accuracy'], label='검증 정확도')
    plt.title('훈련 및 검증 정확도')
    plt.xlabel('에포크')
    plt.ylabel('정확도')
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

print("테스트 데이터에 대한 평가를 수행합니다.")

evaluation = model.evaluate(test_dataset)
print(f"테스트 손실: {evaluation[0]}")
print(f"테스트 정확도: {evaluation[1]}")

y_true = []
y_pred_prob = []

for images, labels in test_dataset:
    preds = model.predict(images)
    y_pred_prob.extend(preds.flatten())
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred = (y_pred_prob >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
print("혼동 행렬:")
print(cm)

report = classification_report(
    y_true, y_pred,
    target_names=['정상', '평발']
)
print("분류 보고서:")
print(report)

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 곡선 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('거짓 양성 비율 (False Positive Rate)')
plt.ylabel('참 양성 비율 (True Positive Rate)')
plt.title('테스트 데이터의 ROC 곡선')
plt.legend(loc="lower right")
plt.show()
