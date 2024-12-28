import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc


validation_dir = "C:/Users/user/Desktop/capstone design/Xray_Images/validation"


model = load_model('C:/Users/user/Desktop/capstone Design/saved_models/normal_abnormal_model_vgg16_finetuned.keras')


validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False  
)


validation_generator.reset()
y_true = validation_generator.classes
y_pred_prob = model.predict(validation_generator)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)


class_labels = list(validation_generator.class_indices.keys())


cm = confusion_matrix(y_true, y_pred)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('혼동 행렬')
plt.show()


print(classification_report(y_true, y_pred, target_names=class_labels))


fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC 곡선 (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('거짓 양성 비율 (False Positive Rate)')
plt.ylabel('참 양성 비율 (True Positive Rate)')
plt.title('ROC 곡선')
plt.legend(loc="lower right")
plt.show()


filenames = validation_generator.filenames


errors = np.where(y_pred != y_true)[0]


for idx in errors[:5]:
    img_path = os.path.join(validation_dir, filenames[idx])
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0

    predicted_label = class_labels[y_pred[idx]]
    actual_label = class_labels[y_true[idx]]

    plt.imshow(img_array)
    plt.title(f"실제: {actual_label}, 예측: {predicted_label}")
    plt.axis('off')
    plt.show()
