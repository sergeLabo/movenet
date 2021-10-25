
import os
from time import time, sleep
from random import randint

import cv2
import numpy as np
import tensorflow as tf

# # COLOR = []
# # for i in range(17):
    # # a = randint(0, 256)
    # # b = randint(0, 256)
    # # c = randint(0, 256)
    # # COLOR.append(f'[{a}, {b}, {c}]')
# # print(COLOR)
# # os._exit(0)


COLOR =[
        [0, 0, 255],
        [0, 255, 0],
        [120, 0, 20],
        [120, 0, 225],
        [80, 120, 0],
        [255, 0, 0],

        [150, 255, 0],
        [150, 160, 255],
        [200, 80, 160],
        [250, 80, 0],
        [250, 129, 202],
        [121, 140, 90],
        [74, 41, 221],

        [88, 218, 141],
        [23, 69, 163],
        [170, 56, 33],
        [18, 195, 56],
        [80, 80, 0],
        [0, 255, 80]]

EDGES = [   (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (3, 1),
            (4, 2),
            (1, 2),
            (5, 6),
            (5, 7),
            (5, 11),
            (6, 8),
            (6, 12),
            (7, 9),
            (8, 10),
            (11, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16)]


def draw(frame, black, keypoints_with_scores):
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
            x = int(item[1]*720)
            y = int((item[0]*720))
            keypoints.append([x, y])
            cv2.circle(frame, (x, y), 6, color=(0,0,255), thickness=-1)
            cv2.circle(black, (x, y), 6, color=(0,0,255), thickness=-1)
        else:
            keypoints.append(None)

    for i, (a, b) in enumerate(EDGES):
        """EDGES = (   (0, 1)
        keypoints[0] = [513, 149]
        """
        if not keypoints[a] or not keypoints[b]:
            continue
        ax = keypoints[a][0]
        ay = keypoints[a][1]
        bx = keypoints[b][0]
        by = keypoints[b][1]
        cv2.line(frame, (ax, ay), (bx, by), COLOR[i], 4)
        cv2.line(black, (ax, ay), (bx, by), COLOR[i], 4)

    return frame, black, keypoints


def multiple_detection(black, template, i):
    # Multiple détection
    res = cv2.matchTemplate(black, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5 # larger values means good fit
    loc = np.where( res >= threshold)
    width, height = template.shape[1], template.shape[0]  # de l'image à détecter
    for pt in zip(*loc[::-1]):
        cv2.rectangle(black, pt, (pt[0] + width, pt[1] + height), (255, 0, 0), 1)
        cv2.putText(black,                      # image
                    str(i),                     # text
                    pt,                         # position
                    cv2.FONT_HERSHEY_SIMPLEX,   # police
                    2,                          # taille police
                    (0, 0, 255),                # couleur
                    2)                          # épaisseur
        break
    return black


interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_thunder_3.tflite")
interpreter.allocate_tensors()

cv2.namedWindow('color', cv2.WND_PROP_FULLSCREEN)
cv2.namedWindow('skeleton', cv2.WND_PROP_FULLSCREEN)
black = np.zeros((720, 720, 3), dtype = "uint8")

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
t0 = time()
nbr = 0

template1 = cv2.imread('bras_droit.png')
template2 = cv2.imread('bras_gauche.png')
template3 = cv2.imread('bras_droit_tendu.png')
template4 = cv2.imread('bras_gauche_tendu.png')

while 1:
    ret, frame = cam.read()
    if not ret:
        continue

    black = np.zeros((720, 720, 3), dtype = "uint8")

    frame = frame[:, 280:1000]

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

    # frame est avec les keypoints
    frame, black, keypoints = draw(frame, black, keypoints_with_scores)

    for i, template in enumerate([template1, template2, template3, template4]):
        black = multiple_detection(black, template, i)

    # Affichage de l'image
    # # cv2.imshow('color', frame)
    cv2.imshow('skeleton', black)

    # Calcul du FPS, affichage toutes les 10 s
    if time() - t0 > 10:
        print("FPS =", round(nbr/10, 2))
        t0, nbr = time(), 0
    nbr += 1

    k = cv2.waitKey(1)

    # # # Uniquement pour la création de mon image 'skeleton.png'
    # # if k == 115:
        # # f = 'bras_gauche_tendu.png'
        # # cv2.imwrite(f, black)
        # # print(f"Enregistrement de {f} ok")

    # Pour quitter
    if k == 27:
        break

cv2.destroyAllWindows()
