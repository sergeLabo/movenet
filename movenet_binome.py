
from time import time


import cv2
import numpy as np
import tensorflow as tf

def get_my_keypoints(keypoints_with_scores):
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

        else:
            keypoints.append(None)

    return keypoints

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
                a = 2
                ROI = HSV[y-a:y+a, x-a:x+a]
                arr[i] = np.nanmean(ROI)
            else:
                arr[i] = None
    return arr

def draw(frame, keypoints_g, keypoints_d):

    for item in keypoints_g[0][0]:
        if item[2] > 0.5:
            x = int(item[1]*640)
            y = int(item[0]*720)
            cv2.circle(frame, (x, y), 6, color=(255,0,255), thickness=-1)

    for item in keypoints_d[0][0]:
        if item[2] > 0.5:
            x = int(item[1]*640) + 640
            y = int(item[0]*720)
            cv2.circle(frame, (x, y), 6, color=(0,255,0), thickness=-1)

    return frame

def draw_text(frame, maxi):
    """frame_teintes_ = np array de 17"""
    try:
        text = str(maxi)
    except:
        text = ""
    frame = cv2.putText(frame,                      # image
                        text,                       # text
                        (50, 200),                  # position
                        cv2.FONT_HERSHEY_SIMPLEX,   # police
                        6,                          # taille police
                        (0, 0, 255),               # couleur
                        12)                          # épaisseur
    return frame

interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_thunder_3.tflite")
interpreter.allocate_tensors()

cv2.namedWindow('color', cv2.WND_PROP_FULLSCREEN)
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

t0 = time()
nbr = 0
maxi = 0

while 1:
    ret, frame = cam.read()
    if not ret:
        continue

    fg = frame[0:720, 0:640]
    fd = frame[0:720, 640:1280]
    frame_copy_g = fg.copy()
    frame_copy_d = fd.copy()

    # Gauche ###################################################################
    image = tf.expand_dims(fg, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.image.resize_with_pad(image, 256, 256)
    input_image = tf.cast(image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    # Output is a [1, 1, 17, 3] numpy array.
    keypoints_with_scores_g = interpreter.get_tensor(output_details[0]['index'])
    keypoints_g = get_my_keypoints(keypoints_with_scores_g)
    frame_teintes_g = get_frame_teintes(frame_copy_g, keypoints_g)

    # Droite ###################################################################
    image = tf.expand_dims(fd, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.image.resize_with_pad(image, 256, 256)
    input_image = tf.cast(image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    # Output is a [1, 1, 17, 3] numpy array.
    keypoints_with_scores_d = interpreter.get_tensor(output_details[0]['index'])
    keypoints_d = get_my_keypoints(keypoints_with_scores_d)
    frame_teintes_d = get_frame_teintes(frame_copy_d, keypoints_d)

    # ##########################################################################
    gap = np.nanmean(np.subtract(frame_teintes_g, frame_teintes_d))
    print(gap, frame_teintes_g, frame_teintes_d)
    # Jeu de la tournée
    if gap:
        try:
            if abs(int(float(gap))) > maxi:
                maxi = abs(int(float(gap)))
        except:
            pass

    frame = draw_text(frame, maxi)
    frame = draw(frame, keypoints_with_scores_g, keypoints_with_scores_d)

    # Calcul du FPS, affichage toutes les 10 s
    if time() - t0 > 10:
        print("FPS =", round(nbr/10, 2))
        t0, nbr = time(), 0
    nbr += 1

    cv2.imshow('color', frame)

    k = cv2.waitKey(1)

    if k == 32:  # space
        maxi = 0
    # Pour quitter
    if k == 27:
        break

cv2.destroyAllWindows()
