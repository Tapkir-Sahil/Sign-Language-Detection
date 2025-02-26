import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands # type: ignore
mp_drawing_utils=mp.solutions.drawing_utils # type: ignore
mp_drawing_styles=mp.solutions.drawing_styles # type: ignore

hands=mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)


DATA_DIR="./data";

for dir_ in os.listdir(DATA_DIR):
  for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
    img=cv2.imread(os.path.join(DATA_DIR,dir_,img_path))
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    result=hands.process(img_rgb)
    if result.multi_hand_landmarks:
      for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing_utils.draw_landmarks(
          img_rgb,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style()
        )


    plt.figure()
    plt.imshow(img_rgb)

plt.show()    