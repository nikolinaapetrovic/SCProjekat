import cv2
import keras
import numpy as np
import vlc

def region_of_interest(frame):
    return frame[200:400, 200:400]

def skin_mask(roi):
    lower_skin = np.array([0,10,60], dtype=np.uint8)
    upper_skin = np.array([20,150,255], dtype=np.uint8)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((3,3),np.uint8)

    mask = cv2.dilate(mask,kernel,iterations = 4)
    mask = cv2.GaussianBlur(mask,(5,5),100)

    return mask

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def controlVLC(command, player):
    if command == "pause":
        player.pause()
    elif command == "resume":
        player.play()
    elif command == "mute":
        player.audio_set_volume(0)
    elif command == "volumeup":
        current = player.audio_get_volume()
        if current + 10 >= 100:
            player.audio_set_volume(99)
        else:
            player.audio_set_volume(current + 10)
    elif command == "volumedown":
        current = player.audio_get_volume()
        if current - 10  <= 0:
            player.audio_set_volume(0)
        else:
            player.audio_set_volume(current - 10)


if __name__ == '__main__':
    last_key = None
    alphabet = ["mute", "none", "pause", "resume", "volumedown", "volumeup"]
    model = keras.models.load_model('model.h5')

    cap = cv2.VideoCapture(0)
    counter = 0

    player = vlc.MediaPlayer("video.mp4")
    player.play()

    while (1):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (100, 300), (100, 300), (0, 255, 0), 0)

        roi = frame[200:400, 200:400]
        mask = skin_mask(roi)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)

        scale = scale_to_range(mask)
        vector = matrix_to_vector(scale)
        input = np.reshape(vector, (1, 40000))

        result = model.predict(np.array(input[0:1], np.float32))
        win = winner(result[0])
        print(win)

        if len(alphabet) >= win:
            current_key = alphabet[win]
        else:
            current_key = None

        if last_key == current_key:
            counter+=1
        else:
            counter = 0

        if counter == 30:
            controlVLC(alphabet[win], player)
            print(current_key)
            counter = 0
        last_key = current_key

        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()