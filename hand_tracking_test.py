# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import json
import websocket
from mediapipe.tasks.python.components.containers.landmark import Landmark

'''
try:
    ws= websocket.WebSocket()
    ws.connect("ws://localhost:8885")
    print("Connected")
except ConnectionRefusedError:
    print("Connection refused")
    exit()
'''

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    static_image_mode=False,   
    max_num_hands=2,           
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            all_hands_data = []

            tip_ids = [4, 8, 12, 16, 20]
            
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label =handedness.classification[0].label

                landmarks_of_interest = {}

                for tip_id in tip_ids:
                    landmark = hand_landmarks.landmark[tip_id]

                    landmarks_of_interest[str(tip_id)] ={
                        "x":round(landmark.x, 5),
                        "y":round(landmark.y, 5),
                        "z":round(landmark.z, 5)
                    }

                all_hands_data.append({
                    "hand_label": hand_label,
                    "landmarks": landmarks_of_interest
                })




                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            
            
            json_data = json.dumps({"hands":all_hands_data})
            #ws.send(json_data)
            print(json_data)
        


        cv2.imshow('Hand Tracking', frame)


        if cv2.waitKey(1) & 0xFF == 27:
            break
print("Closing")
#ws.close()
cap.release()
cv2.destroyAllWindows()



