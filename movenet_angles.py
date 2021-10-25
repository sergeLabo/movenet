
from time import time, sleep
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf
from playsound import playsound


COLOR =[[0, 0, 255],
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

# TODO fait doublon avec OS
EDGES = [   (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (3, 1),
            (4, 2),
            (1, 2),
            (5, 6),
            (5, 7),  # bras gauche
            (5, 11),
            (6, 8),  # bras droit
            (6, 12),
            (7, 9),  # avant bras gauche
            (8, 10), # avant bras droit
            (11, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16)]

NAMES = {   'nez': 0,
            'oeuil gauche': 1,
            'oeuil droit': 2,
            'oreille gauche': 3,
            'oreille droit': 4,
            'epaule gauche': 5,
            'epaule droit': 6,
            'coude gauche': 7,
            'coude droit': 8,
            'poignet gauche': 9,
            'poignet droit': 10,
            'hanche gauche': 11,
            'hanche droit': 12,
            'genou gauche': 13,
            'genou droit': 14,
            'cheville gauche': 15,
            'cheville droit': 16}

OS = {   'nez oeuil gauche': (0, 1),
         'nez oeuil droit':  (0, 2),
         'nez oreille gauche': (0, 3),
         'nez oreille droit': (0, 4),
         'oreille gauche oeuil gauche': (3, 1),
         'oreille oeuil droit': (4, 2),
         'oeuil oeuil droit': (1, 2),
         'epaules': (5, 6),
         'bras gauche': (5, 7),
         'epaule hanche gauche': (5, 11),
         'bras droit': (6, 8),
         'epaule hanche droit': (6, 12),
         'avant bras gauche': (7, 9),
         'avant bras droit': (8, 10),
         'hanches': (11, 12),
         'femur gauche': (11, 13),
         'femur droit': (12, 14),
         'tibia gauche': (13, 15),
         'tibia droit': (14, 16)}

ACTIONS = {
            0: {'bras droit':  (170, 190), 'avant bras droit':  (70, 90)},
            1: {'bras gauche': (-10, 10), 'avant bras gauche': (70, 90)},
            2: {'bras droit':  (170, 190), 'avant bras droit':  (170, 190)},
            3: {'bras gauche': (-10, 10), 'avant bras gauche': (-10, 10)},
            4: {'bras droit': (170, 190), 'avant bras droit': (260, 280)},
            5: {'bras gauche': (-10, 10), 'avant bras gauche': (70, 90)},
            6: {'bras droit': (250, 270), 'avant bras droit': (190, 210)},
            7: {'bras gauche': (70, 90), 'avant bras gauche': (70, 90)},
            8: {'bras droit': (210, 230), 'avant bras droit': (210, 230)},
            9: {'bras gauche': (35, 55), 'avant bras gauche': (35, 55)},
            10: {'bras droit': (70, 90), 'avant bras droit': (70, 90)},
            11: {'bras gauche': (-80, -60), 'avant bras gauche': (-80, -60)},
            12: {'bras droit': (130, 150), 'avant bras droit': (130, 150)},
            13: {'bras gauche': (-55, -35), 'avant bras gauche': (-55, -35)},
            14: {'femur droit': (220, 240), 'tibia droit': (220, 240)},
            15: {'femur gauche': (40, 60), 'tibia gauche': (40, 60)}
            }

COMBINAISONS = {16: (0, 3),
                17: (0, 5),
                18: (0, 7),
                19: (2, 9),
                20: (2, 11),
                21: (2, 13),
                22: (1, 2),
                23: (1, 4),
                24: (1, 6),
                25: (3, 8),
                26: (3, 10),
                27: (3, 12),
                28: (14, 2),
                29: (14, 3),
                30: (14, 0),
                31: (14, 1),
                32: (15, 0),
                33: (15, 1),
                34: (15, 2),
                35: (15, 3)
                }

class Player:
    """Joue les notes de piano dans des threads.
    Les notes ne sont relancées que un certain temps après avoir été jouées,
    0.3 seconde à essayer.
    """
    def __init__(self):
        """Les fichiers sont lus puis joué si appelé.
        running = liste de 36, si 0 pas en cours, si 1 en cours
        note est un int, le numéro de la note dans tout le script
        """
        # Tous les noms de fichiers notes
        self.notes = {}
        for i in range(36):
            self.notes[i] = './samples/' + str(i) + '.ogg'

        # Suivi des notes en cours
        self.running = []
        # time est l'instant du dernier lancement
        for i in range(36):
            self.running.append([0, time()])

        # La boucle du débloqueur
        self.loop = 1
        self.unblocker_thread()

    def play_note(self, note):
        """Play seulement si non bloqué"""

        if self.running[note][0] == 0:
            self.running[note] = 1, time()
            t = Thread(target=self._player, args=(note, ))
            t.start()

    def _player(self, note):
        note_file = self.notes[note]
        playsound(note_file)

    def unblocker_thread(self):
        t = Thread(target=self.unblocker)
        t.start()

    def unblocker(self):
        """Une note jouée ne peut pas être rejouée avant 4 seconde,
        il faut déblocker 0.3 seconde après que running soit passé à 1
        """
        while self.loop:
            sleep(0.03)
            t = time()
            for i in range(36):
                if t - self.running[i][1] > 4:
                    self.running[i] = [0, t]



class Movenet:
    """La reconnaissance de squelette"""

    def __init__(self):
        # Model performant, simple pose mais lourd
        # CUDA est utile !
        model_path="lite-model_movenet_singlepose_thunder_3.tflite"
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Seuil de confiance de la détection des squelettes
        self.threshold = 0.5

    def skeleton_detection(self, frame):
        image = tf.expand_dims(frame, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.image.resize_with_pad(image, 256, 256)
        input_image = tf.cast(image, dtype=tf.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())
        self.interpreter.invoke()
        # Output is a [1, 1, 17, 3] numpy array.
        self.keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        # Construction de ma liste de 17 keypoints = self.keypoints
        self.get_keypoints()

    def get_keypoints(self):
        """keypoints_with_scores = TODO à retrouver
        keypoints = [None, [200, 300], None, [100, 700], ...] = 17 items
        """
        keypoints = []
        for item in self.keypoints_with_scores[0][0]:
            if item[2] > self.threshold:
                x = int(item[1]*720)
                y = int((item[0]*720))
                keypoints.append([x, y])

            else:
                keypoints.append(None)
        self.keypoints = keypoints



class GesturesDetection(Movenet):
    """Détection des gestes"""

    def __init__(self):
        # Héritage de Movenet
        super().__init__()

        self.my_player = Player()

        # Tous les angles
        self.angles = {}
        # Etat de toutes les actions
        self.actions = {}
        print("dir de GesturesDetection", dir(self))

    def gestures_detection(self):
        """Le squelette est défini dans self.keypoints
        Affichage du squelette,
        récupération des angles,
        détection des gestes
        """
        self.draw_keypoints_edges()
        self.get_all_angles()
        self.draw_angles()
        self.detect_actions()
        self.play_note()

    def draw_keypoints_edges(self):
        """
        keypoints = [None, [200, 300], None, [100, 700], ...] = 17 items
        L'index correspond aux valeurs dans NAMES
        """

        # Dessin des points détectés
        for point in self.keypoints:
            if point:
                x = point[0]
                y = point[1]
                cv2.circle(self.black, (x, y), 6, color=(0,0,255), thickness=-1)

        # Dessin des os
        for i, (a, b) in enumerate(EDGES):
            """EDGES = (   (0, 1)
            keypoints[0] = [513, 149]
            """
            if not self.keypoints[a] or not self.keypoints[b]:
                continue
            ax = self.keypoints[a][0]
            ay = self.keypoints[a][1]
            bx = self.keypoints[b][0]
            by = self.keypoints[b][1]
            cv2.line(self.black, (ax, ay), (bx, by), COLOR[i], 4)

    def get_angle(self, p1, p2):
        """Angle entre horizontal et l'os
        origin p1
        p1 = numéro d'os
        tg(alpha) = y2 - y1 / x2 - x1
        """
        alpha = None

        if self.keypoints[p1] and self.keypoints[p2]:
            x1, y1 = self.keypoints[p1][0], self.keypoints[p1][1]
            x2, y2 = self.keypoints[p2][0], self.keypoints[p2][1]
            if x2 - x1 != 0:
                tg_alpha = (y2 - y1) / (x2 - x1)
                if x2 > x1:
                    alpha = int((180/np.pi) * np.arctan(tg_alpha))
                else:
                    alpha = 180 - int((180/np.pi) * np.arctan(tg_alpha))
        return alpha

    def get_all_angles(self):
        """angles = {'tibia droit': 128}
        origine = 1er de OS (14, 16)
        angles idem cercle trigo
        """
        angles = {}
        for os, (p1, p2) in OS.items():
            angles[os] = self.get_angle(p1, p2)

        self.angles = angles

    def draw_angles(self):
        """dessin des valeurs d'angles
        angles = {'tibia droit': 128}
        """
        for os, (p1, p2) in OS.items():
            if self.keypoints[p1] and self.keypoints[p1]:
                alpha = self.angles[os]
                if alpha:
                    u = int((self.keypoints[p1][0] + self.keypoints[p2][0])/2)
                    v = int((self.keypoints[p1][1] + self.keypoints[p2][1])/2)
                    cv2.putText(self.black,                 # image
                                str(alpha),                 # text
                                (u, v-20),                     # position
                                cv2.FONT_HERSHEY_SIMPLEX,   # police
                                1,                          # taille police
                                (0, 255, 255),              # couleur
                                2)                          # épaisseur

    def detect_actions(self):
        """
        ACTIONS = {0: {'bras droit':  (170, 190), 'avant bras droit':  (70, 90)},
        COMBINAISONS = {16: (0, 3),
        actions = {0: 0 ou 1}
        """

        actions = {}

        # ACTIONS
        for action, val in ACTIONS.items():
            # val = {'bras droit':  (-15, 15), 'avant bras droit':  (70, 90)}
            act = 0

            for os, (mini, maxi) in val.items():
                if self.angles[os]:
                    if mini < self.angles[os] < maxi:
                        act = 1
                    actions[action] = act
                else:
                    actions[action] = 0

        # COMBINAISONS
        # actions = {0: 1}
        # Liste des notes à 1
        note_list = []
        for note, val in actions.items():
            if val:
                note_list.append(note)

        for k, v in COMBINAISONS.items():
            if v[0] in note_list and v[1] in note_list:
                actions[k] = 1

        self.actions = actions

    def play_note(self):
        for note, val in self.actions.items():
            if val == 1:
                # Je joue la note
                self.my_player.play_note(note)
                # Je reset, le player gère le blockage
                self.actions[note] = 0




class Visualisation(GesturesDetection):
    """Visualisation et capture avec OpenCV"""

    def __init__(self):
        # Héritage de GesturesDetection
        super().__init__()

        cv2.namedWindow('skeleton', cv2.WND_PROP_FULLSCREEN)
        self.black = np.zeros((720, 720, 3), dtype = "uint8")

        # Définition de la webcam
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def run(self):
        t0 = time()
        nbr = 0
        while 1:
            ret, frame = self.cam.read()
            if not ret:
                continue

            # Réinititialisation de l'image noire
            self.black = np.zeros((720, 720, 3), dtype = "uint8")

            # Coupe de la frame en image carrée 720x720
            frame = frame[:, 280:1000]

            self.skeleton_detection(frame)
            self.gestures_detection()

            # Affichage de l'image
            cv2.imshow('skeleton', self.black)

            # Calcul du FPS, affichage toutes les 10 s
            if time() - t0 > 10:
                print("FPS =", round(nbr/10, 2))
                t0, nbr = time(), 0
            nbr += 1

            k = cv2.waitKey(1)
            # Pour quitter
            if k == 27:
                break

        cv2.destroyAllWindows()



if __name__ == '__main__':

    vis = Visualisation()
    vis.run()
