import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import re
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.font_manager as fm
import matplotlib

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

train_dir = "C:/Users/user/Desktop/capstone design/Xray_Images/train"
validation_dir = "C:/Users/user/Desktop/capstone design/Xray_Images/validation"

model_dir = 'C:/Users/user/Desktop/capstone design/B0saved_models/'
os.makedirs(model_dir, exist_ok=True)

print("EfficientNetB0 모델을 생성합니다.")

input_shape = (224, 224, 3)

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

x = base_model.output
x = layers.BatchNormalization()(x)

x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)

x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)

x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = layers.BatchNormalization()(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model_files = [f for f in os.listdir(model_dir)
               if f.startswith('efficientnetb0_model_best_val_accuracy_epoch_')
               and f.endswith('.keras')]

if model_files:
    epochs_list = []
    for f in model_files:
        match = re.search(r'epoch_(\d+)', f)
        if match:
            epochs_list.append(int(match.group(1)))
    last_epoch = max(epochs_list)
    last_model_file = f'efficientnetb0_model_best_val_accuracy_epoch_{last_epoch:02d}.keras'
    last_model_path = os.path.join(model_dir, last_model_file)
    model = load_model(last_model_path)
    print(f"저장된 모델을 성공적으로 로드했습니다. (에포크: {last_epoch})")
    initial_epoch = last_epoch
else:
    print("저장된 모델을 찾을 수 없습니다. 새로운 모델로 학습을 시작합니다.")
    initial_epoch = 0

initial_lr = 0.0001

optimizer = AdamW(learning_rate=initial_lr, weight_decay=1e-5)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

def exclude_edges(img, exclude_fraction=0.05, target_size=(224, 224)):
    img_height, img_width = img.shape[:2]
    left = int(img_width * exclude_fraction)
    top = int(img_height * exclude_fraction)
    right = int(img_width * (1 - exclude_fraction))
    bottom = int(img_height * (1 - exclude_fraction))
    img_cropped = img[top:bottom, left:right, :]
    img_resized = tf.image.resize(img_cropped, target_size)
    img_resized = img_resized.numpy()
    return img_resized

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

batch_size = 32

input_size = (224, 224)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

train_labels = train_generator.classes
class_indices = train_generator.class_indices
class_counts = np.bincount(train_labels)

index_to_class = {v: k for k, v in class_indices.items()}

total_train_samples = len(train_labels)

print("훈련 데이터 클래스 분포:")
for i, count in enumerate(class_counts):
    class_name = index_to_class[i]
    percentage = (count / total_train_samples) * 100
    print(f" - 클래스 '{class_name}': {count}개 ({percentage:.2f}%)")

train_class_counts = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_class_counts),
    y=train_class_counts
)
class_weights = dict(enumerate(class_weights))

print("\n클래스 가중치:")
for class_index, weight in class_weights.items():
    class_name = index_to_class[class_index]
    print(f" - 클래스 '{class_name}': 가중치 = {weight:.2f}")

checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_dir, 'efficientnetb0_model_epoch_{epoch:02d}.keras'),
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=False,
    mode='min'
)

best_val_accuracy_checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_dir, 'efficientnetb0_best_val_accuracy_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

best_accuracy_checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_dir, 'efficientnetb0_best_accuracy_model.keras'),
    monitor='accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
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

class CurrentEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.current_epoch = epoch + 1

current_epoch_logger = CurrentEpochLogger()

callbacks = [
    early_stopping,
    reduce_lr,
    checkpoint,
    best_val_accuracy_checkpoint,
    best_accuracy_checkpoint,
    current_epoch_logger
]

epochs = 150

history = None

try:
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        initial_epoch=initial_epoch
    )
    print("학습이 완료되었습니다.")
except KeyboardInterrupt:
    print("학습이 중단되었습니다.")
finally:
    if hasattr(model, 'current_epoch'):
        final_epoch = model.current_epoch
    else:
        final_epoch = initial_epoch

    best_model_path = os.path.join(model_dir, 'efficientnetb0_best_val_accuracy_model.keras')
    if os.path.exists(best_model_path):
        best_model = load_model(best_model_path)
        final_model_path = os.path.join(
            model_dir, f'efficientnetb0_model_best_val_accuracy_epoch_{final_epoch:02d}.keras'
        )
        best_model.save(final_model_path)
        print(f"최고 검증 정확도 모델을 저장했습니다. (에포크: {final_epoch})")
    else:
        final_model_path = os.path.join(
            model_dir, f'efficientnetb0_model_epoch_{final_epoch:02d}.keras'
        )
        model.save(final_model_path)
        print(f"최종 모델을 저장했습니다. (에포크: {final_epoch})")

plt.figure(figsize=(8, 6))
if history is not None:
    plt.plot(
        range(initial_epoch + 1, final_epoch + 1),
        history.history['accuracy'],
        label='훈련 정확도'
    )
    plt.plot(
        range(initial_epoch + 1, final_epoch + 1),
        history.history['val_accuracy'],
        label='검증 정확도'
    )
plt.title('훈련 및 검증 정확도')
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
if history is not None:
    plt.plot(
        range(initial_epoch + 1, final_epoch + 1),
        history.history['loss'],
        label='훈련 손실'
    )
    plt.plot(
        range(initial_epoch + 1, final_epoch + 1),
        history.history['val_loss'],
        label='검증 손실'
    )
plt.title('훈련 및 검증 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.legend()
plt.show()

evaluation = model.evaluate(validation_generator)
print(f"검증 손실: {evaluation[0]}")
print(f"검증 정확도: {evaluation[1]}")

validation_generator.reset()
y_true = validation_generator.classes
y_pred_prob = model.predict(validation_generator)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

target_fpr = 0.10
optimal_idx = np.where(fpr <= target_fpr)[0][-1]
best_threshold = thresholds[optimal_idx]
print(f"선택한 임계값 (FPR <= {target_fpr*100}%): {best_threshold}")

y_pred = (y_pred_prob >= best_threshold).astype(int).reshape(-1)

cm = confusion_matrix(y_true, y_pred)
print("혼동 행렬:")
print(cm)

report = classification_report(
    y_true,
    y_pred,
    target_names=list(validation_generator.class_indices.keys())
)
print("분류 보고서:")
print(report)

class_indices = validation_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

tn, fp, fn, tp = cm.ravel()

total_val_samples = len(y_true)

false_positive_rate = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
false_negative_rate = (fn / (tp + fn)) * 100 if (tp + fn) > 0 else 0

print("\n검증 데이터에서 오분류 비율:")
print(f" - 정상 샘플을 비정상으로 예측한 비율: {false_positive_rate:.2f}%")
print(f" - 비정상 샘플을 정상으로 예측한 비율: {false_negative_rate:.2f}%")

from sklearn.metrics import roc_curve, auc

fpr_plot, tpr_plot, thresholds_plot = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr_plot, tpr_plot)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 곡선 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('거짓 양성 비율 (False Positive Rate)')
plt.ylabel('참 양성 비율 (True Positive Rate)')
plt.title('검증 데이터의 ROC 곡선')
plt.legend(loc="lower right")
plt.show()
