from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
import cv2
import os
import numpy as np

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(path):
    inputs = []

    for dir in os.listdir(path):
        for img in os.listdir(path + "/" + dir):
            image = cv2.imread(path + "/" + dir + "/" + img, cv2.IMREAD_GRAYSCALE)
            scale = scale_to_range(image)
            inputs.append(matrix_to_vector(scale))
    return inputs

def convert_output(alphabet, path):
    outputs = []
    i = 0
    for dir in os.listdir(path):
        for img in os.listdir(path + "/" + dir):
            label = np.zeros(len(alphabet))
            label[i] = 1
            outputs.append(label)
        i+=1

    return outputs

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=40000, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, x_train, y_train, epochs):
    x_train = np.array(x_train, np.float32)  # dati ulaz
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi na date ulaze

    print("\nTraining started...")
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    ann.summary()
    ann.fit(x_train, y_train, epochs=epochs, batch_size=10, verbose=1, shuffle=True)
    print("\nTraining completed...")
    return ann


def validate_ann(ann, x_validate, y_validate):
    x_validate = np.array(x_validate, np.float32)
    y_validate = np.array(y_validate, np.float32)

    print("\nValidating started...")
    loss, acc = ann.evaluate(x_validate, y_validate, batch_size=5, verbose=1)
    print("loss: " + str(loss), "acc: " + str(acc))
    print("\nValidating completed...")
    return loss, acc

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

if __name__ == '__main__':
    alphabet = ["mute", "none", "pause", "resume", "volumedown", "volumeup"]

    inputs = prepare_for_ann("data")
    inputs_valid = prepare_for_ann("valid")
    outputs = convert_output(alphabet, "data")
    outputs_valid = convert_output(alphabet, "valid")

    ann = create_ann(len(alphabet))
    ann = train_ann(ann, inputs, outputs, epochs=20)
    loss, acc = validate_ann(ann, inputs_valid, outputs_valid)

    ann.save('model.h5')