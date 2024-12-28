import tensorflow as tf
import numpy as np
import pandas as pd
import os


best_model_path = 'C:/Users/user/Desktop/capstone design/CNN_saved_models/cnn_regression_best_val_model.keras'


test_csv_file = "C:/Users/user/Desktop/capstone design/cell/boneage-test-dataset.csv"
test_image_dir = "C:/Users/user/Desktop/capstone design/boneage-test-dataset/"


img_height = 224
img_width = 224


print("최고 성능의 모델을 로드합니다.")
model = tf.keras.models.load_model(best_model_path)


print("테스트 데이터를 로드하고 전처리합니다.")
test_df = pd.read_csv(test_csv_file)


test_df['image_path'] = test_df['Case ID'].apply(lambda x: os.path.join(test_image_dir, f"{x}.png"))


test_df['gender'] = test_df['Sex'].map({'M': 1, 'F': 0})


def preprocess_image(image_path, gender):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)


    image = tf.image.resize(image, [img_height, img_width])


    image = image / 255.0


    gender = tf.cast(gender, tf.float32)
    gender = tf.expand_dims(gender, -1)

    return {'image': image, 'gender': gender}

def create_test_dataset(df, batch_size=32):
    paths = df['image_path'].values
    genders = df['gender'].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, genders))
    dataset = dataset.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


test_dataset = create_test_dataset(test_df, batch_size=32)


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


    prediction_csv_path = 'C:/Users/user/Desktop/capstone design/boneage_csv/boneage_test_predictions.csv'
    test_df[['Case ID', 'boneage_prediction']].to_csv(prediction_csv_path, index=False)
    print(f"예측 결과를 '{prediction_csv_path}' 파일로 저장하였습니다.")
except Exception as e:
    print(f"예측 중 오류 발생: {e}")