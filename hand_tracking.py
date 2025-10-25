# -*- coding: utf-8 -*-
import math
import cv2
import mediapipe as mp
import json
import websocket
from mediapipe.tasks.python.components.containers.landmark import Landmark

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

with mp_hands.Hands(
    static_image_mode=False,   
    max_num_hands=2,           
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    THRESHOLD = 0.1
    ZOOM_THRESHOLD = 0.1


    def get_distance(p1, p2):
        if p1 is None or p2 is None: return None
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def get_dict_distance(p1_dict, p2_dict):
        if p1_dict is None or p2_dict is None: return None
        return math.sqrt((p1_dict["x"] - p2_dict["x"]) ** 2 + (p1_dict["y"] - p2_dict["y"]) ** 2)

    def extract_landmarks(hand_landmarks,ids_to_extract):
        landmarks_of_interest = {}
        for tip_id in ids_to_extract:
            landmarks = hand_landmarks.landmark[tip_id]
            landmarks_of_interest[str(tip_id)] = {
                'x': round(landmarks.x, 5),
                'y': round(landmarks.y, 5),
                'z': round(landmarks.z, 5)
            }
        return landmarks_of_interest


    def check_expansion_zoom(result):
        if len(result.multi_hand_landmarks) == 2:
            hand0_palm = result.multi_hand_landmarks[0].landmark[12]
            hand1_palm = result.multi_hand_landmarks[1].landmark[12]

            distance = get_distance(hand0_palm, hand1_palm)
            return distance
        return None

    def recognize_single_hand_gesture(landmarks_of_interest):
        pos4 = landmarks_of_interest.get("4")
        pos8 = landmarks_of_interest.get("8")
        pos12 = landmarks_of_interest.get("12")
        pos16 = landmarks_of_interest.get("16")
        pos20 = landmarks_of_interest.get("20")


        dist_4_8 = get_dict_distance(pos4, pos8)
        dist_8_12 = get_dict_distance(pos8, pos12)
        dist_4_12 = get_dict_distance(pos4, pos12)
        dist_4_16 = get_dict_distance(pos4, pos16)
        dist_12_16 = get_dict_distance(pos12, pos16)
        dist_16_20 = get_dict_distance(pos16, pos20)



        dist = [dist_4_8, dist_8_12, dist_4_12, dist_12_16, dist_16_20, dist_4_16]





        if any(d is None for d in dist):
            return "none"

        if (dist_12_16<THRESHOLD) and (dist_16_20 < ZOOM_THRESHOLD):
            if dist_4_8 < THRESHOLD:
                return "pinch"

        if (dist_8_12 < THRESHOLD) and (dist_4_16 < THRESHOLD):
            return "drag"

        if (dist_16_20<ZOOM_THRESHOLD) and (dist_8_12 < ZOOM_THRESHOLD):
            if dist_4_8 > ZOOM_THRESHOLD:
                return "zoom_in"
            else:
                return "zoom_out"




        return "none"

    # gesture state
    last_expansion_dist = None
    current_gesture_state = "none"
    drag_start_pos = None




    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_global_action = "none"

        if result.multi_hand_landmarks:
            all_hands_data = []
            tip_ids = [4, 8, 12, 16, 20]

            expansion_dist = check_expansion_zoom(result)
            if expansion_dist is not None:
                current_global_action = "expansion_zoom"
                if last_expansion_dist is not None:
                    delta = expansion_dist - last_expansion_dist
                    if delta > ZOOM_THRESHOLD:
                        print("zoom in")
                    elif delta < ZOOM_THRESHOLD:
                        print("zoom out")

                last_expansion_dist = expansion_dist
            else:
                last_expansion_dist = None

            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label

                landmarks_of_interest = extract_landmarks(hand_landmarks, tip_ids)

                if current_global_action == "none":

                    gesture = "none"
                    if hand_label == "Right":
                        gesture = recognize_single_hand_gesture(landmarks_of_interest)

                    if gesture == "drag":
                        if current_gesture_state != "drag":
                            print("---drag start!---")

                            drag_start_pos = {
                                "x": (landmarks_of_interest["8"]["x"] + landmarks_of_interest["12"]["x"]) / 2,
                                "y": (landmarks_of_interest["8"]["y"] + landmarks_of_interest["12"]["y"]) / 2
                            }

                        else:
                            if drag_start_pos:
                                current_pos = {
                                    "x": (landmarks_of_interest["8"]["x"] + landmarks_of_interest["12"]["x"]) / 2,
                                    "y": (landmarks_of_interest["8"]["y"] + landmarks_of_interest["12"]["y"]) / 2
                                }
                                delta_x = current_pos["x"] - drag_start_pos["x"]
                                delta_y = current_pos["y"] - drag_start_pos["y"]
                                print(f"drag_ing: dx={delta_x:.3f}, dy={delta_y:.3f}")
                    else:
                        if current_gesture_state == "drag":
                            print("---drag end!---")
                            drag_start_pos = None
                        if "zoom" in gesture:
                            print("current gesture state:", gesture)

                current_gesture_state = gesture

                all_hands_data.append({"hand_label": hand_label, "gesture:": gesture, "landmarks": landmarks_of_interest})

                mp_drawing.draw_landmarks(
                     frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )


            json_data = json.dumps({"hands":all_hands_data})
            print(json_data)



        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
