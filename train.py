import pickle
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

import keras
from keras import layers
from keras import ops
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np
import csv

model_save_path = "./Model/model.h5"

label_save_path = "./Model/label.csv"

data_dict = pickle.load(open('./data.pickle','rb'))

seen = []

data=np.asarray(data_dict['data'])
labels=np.asarray(data_dict['labels'])

if labels.dtype.kind in 'UO':  # Check if labels are strings
    le = LabelEncoder()
    labels = le.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

x_train = x_train/255
x_test = x_test/255

num_classes = len(np.unique(labels))  # Assuming labels are integers now
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = keras.Sequential([
    keras.Input(shape=(224,224,3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train,y_train,epochs = 5, batch_size = 64)

model.evaluate(x_test,y_test)

model.summary()

model.save(model_save_path)

with open(label_save_path, 'w', newline='') as f:
    wr=csv.writer(f)
    for val in data_dict['labels']:
        if val in seen:
            pass
        else:
            seen.append(val)
            wr.writerow([val])

# predict_result = model.predict(np.array([x_test[0]]))
# print(np.squeeze(predict_result))
# print(np.argmax(np.squeeze(predict_result)))