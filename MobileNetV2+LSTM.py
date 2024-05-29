import os
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns


# Define constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]
video_dir = "Real Life Violence Dataset/"

# # Function to extract frames from a video
# def frame_extraction(video_path):
#     video_reader = cv2.VideoCapture(video_path)
#     video_frame_count = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
#     skip_frames_window = max(int(video_frame_count / SEQUENCE_LENGTH), 1)
#     frame_list = []
#     for frame_counter in range(SEQUENCE_LENGTH):
#         video_reader.set(cv2.CAP_PROP_POS_FRAMES, skip_frames_window * frame_counter)
#         success, frame = video_reader.read()
#         if not success:
#             break
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
#         normalized_frame = resized_frame / 255
#         frame_list.append(normalized_frame)
#
#     video_reader.release()
#     return frame_list
#
# # Function to create dataset
# def create_dataset():
#     features = []
#     labels = []
#     video_file_paths = []
#
#     for class_index, class_name in enumerate(CLASSES_LIST):
#         print("Extracting data of class:", class_name)
#         files_list = os.listdir(os.path.join(video_dir, class_name))
#         for file_name in files_list:
#             video_file_path = os.path.join(video_dir, class_name, file_name)
#             frames = frame_extraction(video_file_path)
#             if len(frames) == SEQUENCE_LENGTH:
#                 features.append(frames)
#                 labels.append(class_index)
#                 video_file_paths.append(video_file_path)
#
#     features = np.asarray(features)
#     labels = np.asarray(labels)
#     return features, labels, video_file_paths
#
# if __name__ == "__main__":
#     print("Feature extracting...")
#     features, labels, video_file_paths = create_dataset()
#     np.save("Features/features_Own.npy", features)
#     np.save("Features/labels_Own.npy", labels)
#     np.save("Features/video_file_paths_Own.npy", video_file_paths)
#     print("Saved feature files successfully")
#
features, labels = np.load("Features/features_Own.npy"), np.load("Features/labels_Own.npy")
one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size=0.2,
                                                                            shuffle=True, random_state=42)

# Load MobileNetV2
mobilenet = MobileNetV2(include_top=False, weights="imagenet")
mobilenet.trainable = True

for layer in mobilenet.layers[:-40]:
    layer.trainable = False

# Function to create model
def create_model():
    model = Sequential()
    model.add(Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(mobilenet))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))

    lstm_fw = LSTM(units=32)
    lstm_bw = LSTM(units=32, go_backwards=True)

    model.add(Bidirectional(lstm_fw, backward_layer=lstm_bw))
    model.add(Dropout(0.25))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(len(CLASSES_LIST), activation='softmax'))

    model.summary()

    return model

# Create model
Own_dataset_mobi_Lstm = create_model()

early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Create ReduceLROnPlateau Callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.6,
                                                 patience=5,
                                                 min_lr=0.00005,
                                                 verbose=1)

# Compile the model
Own_dataset_mobi_Lstm.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

# Fit the model
MobBiLSTM_model_history = Own_dataset_mobi_Lstm.fit(x=features_train, y=labels_train, epochs=300, batch_size=8,
                                                    shuffle=True, validation_split=0.2,
                                                    callbacks=[early_stopping_callback, reduce_lr])

# Evaluate the model
model_evaluation_history = Own_dataset_mobi_Lstm.evaluate(features_test, labels_test)
labels_predict = Own_dataset_mobi_Lstm.predict(features_test)
labels_predict = np.argmax(labels_predict, axis=1)
labels_test_normal = np.argmax(labels_test, axis=1)
AccScore = accuracy_score(labels_predict, labels_test_normal)
print('Accuracy Score is:', AccScore)

# Save the model
Own_dataset_mobi_Lstm.save('Own_dataset_mobi_Lstm3.h5')

# Plot confusion matrix
cm = confusion_matrix(labels_test_normal, labels_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix of mobilenet_v2")
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()



# Print classification report
print("\nClassification Report of mobilenet_v2:")
print(classification_report(labels_test_normal, labels_predict))

# Plot ROC curve
fpr, tpr, _ = roc_curve(labels_test_normal, labels_predict)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of mobilenet_v2')
plt.legend(loc="lower right")
plt.savefig('ROC Curve of mobilenet_v2.png')
plt.show()

# Plot loss curve
history = MobBiLSTM_model_history
plt.figure(figsize=[6, 4])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'blue', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves of Mobilenet_v2', fontsize=12)
plt.savefig('Loss Curves of Mobilenet_v2.png')
plt.show()

