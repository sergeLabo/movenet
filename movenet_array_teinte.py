
import os
from time import time, sleep
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf

global ON, STOP
ON = 0
STOP = 0


def draw(frame, keypoints_with_scores):
    """
    x sur la largeur
    y sur la hauteur
        keypoints_with_scores[0][0] =
        [   [0.6292529  0.5551566  0.7030978 ]
            [0.5557218  0.60308343 0.8078997 ]
            ...]
    item = [0.83882695 0.5022 0.00405499] <class 'numpy.ndarray'>
    """

    keypoints = []
    for item in keypoints_with_scores[0][0]:
        # type(item) = <class 'numpy.ndarray'>
        # type(item[1]) = <class 'numpy.float32'>
        if item[2] > 0.5:
            x = int(item[1]*640)
            y = int(item[0]*480)
            keypoints.append([x, y])
            cv2.circle(frame, (x, y), 6, color=(0,0,255), thickness=-1)
        else:
            keypoints.append(None)

    return frame, keypoints

def get_frame_teintes(frame, keypoints):
    """ frame_teintes = array de 17 valeurs de Hue
        keypoints = [[125, 781], ... , [...]]
    """
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    arr = np.empty(17)
    if keypoints:
        for i, keypoint in enumerate(keypoints):
            if keypoint: # numpy.float64
                x = keypoint[0]
                y = keypoint[1]
                a = 1
                ROI = HSV[y-a:y+a, x-a:x+a]
                arr[i] = np.nanmean(ROI)
            else:
                arr[i] = None
    return arr

def get_gap_teinte(histo, i):
    """Ecarts entre
            - la i ème de histo
            - la dernière histo[-1]
    Retourne un float64
    """
    gap = None
    try:
        gap = abs(np.nanmean(np.subtract(histo[i], histo[-1])))
    except:
        pass
    return gap

interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_thunder_3.tflite")
interpreter.allocate_tensors()

cv2.namedWindow('color', cv2.WND_PROP_FULLSCREEN)
cv2.namedWindow('Hue Variation', cv2.WND_PROP_FULLSCREEN)
cam = cv2.VideoCapture(0)
t0 = time()
nbr = 0
teinte_histo = np.array([])

def on_pause():
    global ON, STOP
    while ON:
        STOP = 1
        sleep(0.1)
        print("Pause ...", STOP)
    STOP = 0

while 1:
    ret, frame = cam.read()
    if not ret:
        continue

    if not STOP:
        image = tf.expand_dims(frame, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.image.resize_with_pad(image, 256, 256)

        input_image = tf.cast(image, dtype=tf.float32)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        interpreter.invoke()

        # Output is a [1, 1, 17, 3] numpy array.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        # Copie de la frame sans les keypoints
        frame_copy = frame.copy()
        # frame est avec les keypoints
        frame, keypoints = draw(frame, keypoints_with_scores)
        frame_teintes = get_frame_teintes(frame_copy, keypoints)  # (17,)

        # Ajout à l'historique
        if not teinte_histo.any():
            teinte_histo = frame_teintes  # (17,)
        else:
            teinte_histo = np.vstack((teinte_histo, frame_teintes))
            # Limitation à 50 items
            teinte_histo = teinte_histo[-50:, ]
            # teinte_histo.shape = (50, 17)

        # Calcul des écarts
        gaps = np.array([])
        for i in range(teinte_histo.shape[0]-1):
            gap = get_gap_teinte(teinte_histo, i)
            gaps = np.append(gaps, gap, axis=None)

        gaps = list(gaps)
        if len(teinte_histo) >= 50:
            black = np.zeros((500, 1000, 3), dtype = "uint8")
            for i in range(len(gaps)):
                try:
                    # From top-left corner to bottom-right corner of rectangle.
                    t1 = 20*(i+1)-20
                    t2 = 500-int(gaps[i]*2)
                    t3 = 20*(i+1)
                    t4 = 500
                    cv2.rectangle(black, (t1,t2), (t3,t4), (0, 255, 0), -1)
                except:
                    pass
            cv2.imshow('Hue Variation', black)

        # Calcul du FPS, affichage toutes les 10 s
        if time() - t0 > 10:
            print("FPS =", round(nbr/10, 2))
            t0, nbr = time(), 0
        nbr += 1

    # Affichage de l'image
    cv2.imshow('color', frame)

    k = cv2.waitKey(1)

    # Pour quitter
    if k == 27:
        break
    # Space pour pause
    if k == 32:
        if ON == 0:
            ON = 1
            tempo = Thread(target=on_pause)
            tempo.start()
    if k == 98:  # b
        if ON == 1:
            ON = 0


cv2.destroyAllWindows()
