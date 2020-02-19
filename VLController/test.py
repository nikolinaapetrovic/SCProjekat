import numpy as np
import cv2
import os
import keras
import matplotlib.pyplot as plt

from keras.optimizers import SGD
from keras.optimizers import Adam

from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers.core import Dense, Flatten
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def image_prepare(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    scale = scale_to_range(image)
    return matrix_to_vector(scale)

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def testMute(model, alphabet):
    test_inputs = []
    for img in os.listdir("test/mute"):
        prepared_image = image_prepare("test/mute/" + img)
        test_inputs.append(prepared_image)

    result = model.predict(np.array(test_inputs, np.float32))

    count = 0

    for r in result:
        win = winner(r)
        if alphabet[win] == "mute":
            count+= 1
    print("MUTE")
    print("Broj pogodjenih testova je ", count, " od ukupno ", len(result))
    print("To je ", count/len(result)*100, "%")
    print("-------------------------------")

def testPause(model, alphabet):
    test_inputs = []
    for img in os.listdir("test/pause"):
        prepared_image = image_prepare("test/pause/" + img)
        test_inputs.append(prepared_image)

    result = model.predict(np.array(test_inputs, np.float32))

    count = 0

    for r in result:
        win = winner(r)
        if alphabet[win] == "pause":
            count += 1
    print("PAUSE")
    print("Broj pogodjenih testova je ", count, " od ukupno ", len(result))
    print("To je ", count / len(result) * 100, "%")
    print("-------------------------------")

def testResume(model, alphabet):
    test_inputs = []
    for img in os.listdir("test/resume"):
        prepared_image = image_prepare("test/resume/" + img)
        test_inputs.append(prepared_image)

    result = model.predict(np.array(test_inputs, np.float32))

    count = 0

    for r in result:
        win = winner(r)
        if alphabet[win] == "resume":
            count += 1
    print("RESUME")
    print("Broj pogodjenih testova je ", count, " od ukupno ", len(result))
    print("To je ", count / len(result) * 100, "%")
    print("-------------------------------")

def testVolumeDown(model, alphabet):
    test_inputs = []
    for img in os.listdir("test/volumedown"):
        prepared_image = image_prepare("test/volumedown/" + img)
        test_inputs.append(prepared_image)

    result = model.predict(np.array(test_inputs, np.float32))

    count = 0

    for r in result:
        win = winner(r)
        if alphabet[win] == "volumedown":
            count += 1
    print("VOLUMEDOWN")
    print("Broj pogodjenih testova je ", count, " od ukupno ", len(result))
    print("To je ", count / len(result) * 100, "%")
    print("-------------------------------")

def testVolumeUp(model, alphabet):
    test_inputs = []
    for img in os.listdir("test/volumeup"):
        prepared_image = image_prepare("test/volumeup/" + img)
        test_inputs.append(prepared_image)

    result = model.predict(np.array(test_inputs, np.float32))

    count = 0

    for r in result:
        win = winner(r)
        if alphabet[win] == "volumeup":
            count += 1
    print("VOLUMEUP")
    print("Broj pogodjenih testova je ", count, " od ukupno ", len(result))
    print("To je ", count / len(result) * 100, "%")
    print("-------------------------------")

def testNone(model, alphabet):
    test_inputs = []
    for img in os.listdir("test/none"):
        prepared_image = image_prepare("test/none/" + img)
        test_inputs.append(prepared_image)

    result = model.predict(np.array(test_inputs, np.float32))

    count = 0

    for r in result:
        win = winner(r)
        if alphabet[win] == "none":
            count += 1
    print("NONE")
    print("Broj pogodjenih testova je ", count, " od ukupno ", len(result))
    print("To je ", count / len(result) * 100, "%")
    print("-------------------------------")

if __name__ == "__main__":
    model = keras.models.load_model('model.h5')
    alphabet = ["mute", "none", "pause", "resume", "volumedown", "volumeup"]

    testMute(model, alphabet)
    testPause(model, alphabet)
    testResume(model, alphabet)
    testVolumeDown(model, alphabet)
    testVolumeUp(model, alphabet)
    testNone(model, alphabet)