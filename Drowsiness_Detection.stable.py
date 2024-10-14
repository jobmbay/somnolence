import cv2
import imutils
import numpy as np
import dlib
import tensorflow as tf
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import mediapipe as mp

# Initialiser la lecture de musique
mixer.init()
mixer.music.load("music.wav")

# Charger le modèle MobileNetV2 pré-entraîné
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Fonction pour préparer les images pour MobileNetV2
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # MobileNetV2 utilise des images de 224x224 pixels
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Fonction de détection de somnolence par EAR
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialisation des constantes
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Initialiser MediaPipe pour le dessin
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Capture vidéo
cap = cv2.VideoCapture(0)
flag = 0
no_face_flag = 0

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Préparer l'image pour le modèle CNN
        preprocessed_img = preprocess_image(frame)
        predictions = model.predict(preprocessed_img)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        distraction_label = decoded_predictions[0][1]
        confidence = decoded_predictions[0][2]

        # Vérifier les prédictions pour les distractions
        if distraction_label in ['cell_phone', 'look_away', 'drowsy'] and confidence > 0.5:
            cv2.putText(frame, f"Distraction detected: {distraction_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mixer.music.play()

        # Détection du visage avec dlib pour la somnolence
        subjects = detect(gray, 0)

        if subjects:
            no_face_flag = 0  # Réinitialiser l'alerte de visage non détecté
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)

                # Convertir en RGB pour MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Dessiner les contours du visage
                        mp_drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_CONTOURS,
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
                        )

                # Calculer l'EAR pour la somnolence
                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < thresh:
                    flag += 1
                    if flag >= frame_check:
                        cv2.putText(frame, "****************EYE CLOSED!****************", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mixer.music.play()
                else:
                    flag = 0

        else:
            no_face_flag += 1  # Incrémenter si aucun visage n'est détecté
            if no_face_flag >= frame_check:  # Alerte si le visage n'est pas détecté pendant plusieurs frames
                cv2.putText(frame, "****************NO FACE DETECTED!****************", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()

        # Afficher le résultat
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# Libérer les ressources
cv2.destroyAllWindows()
cap.release()
