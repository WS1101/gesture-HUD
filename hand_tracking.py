# -*- coding: utf-8 -*-
import math
import cv2
import mediapipe as mp
import json
import websocket
import time
from mediapipe.tasks.python.components.containers.landmark import Landmark


def connect_websocket():
    while True:
        try:
            ws = websocket.WebSocket()
            ws.connect("ws://localhost:8884")
            print("✓ WebSocket Connected")
            return ws
        except ConnectionRefusedError:
            print("✗ Connection refused. Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            print(f"✗ Connection error: {e}. Retrying in 2 seconds...")
            time.sleep(2)


ws = connect_websocket()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# --- Gesture recognization function ---
def get_distance(p1, p2):
    if p1 is None or p2 is None: return None
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_dict_distance(p1_dict, p2_dict):
    if p1_dict is None or p2_dict is None: return None
    return math.sqrt((p1_dict["x"] - p2_dict["x"]) ** 2 + (p1_dict["y"] - p2_dict["y"]) ** 2)


def extract_landmarks(hand_landmarks, ids_to_extract):
    landmarks_of_interest = {}
    for tip_id in ids_to_extract:
        landmarks = hand_landmarks.landmark[tip_id]
        landmarks_of_interest[str(tip_id)] = {
            'x': round(landmarks.x, 5),
            'y': round(landmarks.y, 5),
            'z': round(landmarks.z, 5)
        }
    return landmarks_of_interest


# 개선된 손 크기 계산: 손목에서 각 손가락 끝까지 평균 거리
def get_hand_size(landmarks_of_interest):
    pos0 = landmarks_of_interest.get("0")  # 손목
    finger_tips = ["8", "12", "16", "20"]  # 검지, 중지, 약지, 새끼 끝

    if pos0 is None:
        return None

    distances = []
    for tip_id in finger_tips:
        tip_pos = landmarks_of_interest.get(tip_id)
        if tip_pos:
            dist = get_dict_distance(pos0, tip_pos)
            if dist is not None:
                distances.append(dist)

    if len(distances) == 0:
        return None

    # 평균 거리 사용 (더 안정적)
    avg_distance = sum(distances) / len(distances)
    return avg_distance


def recognize_single_hand_gesture(landmarks_of_interest, pinch_low, pinch_high, zoom_threshold,
                                  pointer_ratio, previous_gesture):
    # 손 크기 기준값 계산
    hand_size = get_hand_size(landmarks_of_interest)
    if hand_size is None or hand_size < 0.01:  # 너무 작으면 무시
        return "none"

    pos4 = landmarks_of_interest.get("4")
    pos8 = landmarks_of_interest.get("8")
    pos12 = landmarks_of_interest.get("12")
    pos16 = landmarks_of_interest.get("16")
    pos20 = landmarks_of_interest.get("20")

    dist_4_8 = get_dict_distance(pos4, pos8)
    dist_8_12 = get_dict_distance(pos8, pos12)
    dist_12_16 = get_dict_distance(pos12, pos16)
    dist_16_20 = get_dict_distance(pos16, pos20)

    dist = [dist_4_8, dist_8_12, dist_12_16, dist_16_20]

    if any(d is None for d in dist):
        return "none"

    # 비율로 변환 (거리 / 손 크기) - 정규화
    ratio_4_8 = dist_4_8 / hand_size
    ratio_8_12 = dist_8_12 / hand_size
    ratio_12_16 = dist_12_16 / hand_size
    ratio_16_20 = dist_16_20 / hand_size

    # 디버깅용 출력
    print(f"Ratios - 4_8: {ratio_4_8:.3f}, 8_12: {ratio_8_12:.3f}")

    # 히스테리시스를 적용한 핀치 제스처 판단
    if previous_gesture == "pinch" or previous_gesture == "left_pinch":
        # 이미 핀치 상태 → 더 큰 값에서만 해제 (pinch_high)
        is_pinch = (ratio_4_8 < pinch_high) and (ratio_8_12 > pinch_low * 2.0)
    else:
        # 핀치 아닌 상태 → 더 작은 값에서만 진입 (pinch_low)
        is_pinch = (ratio_4_8 < pinch_low) and (ratio_8_12 > pinch_low * 2.0)

    if is_pinch:
        return "pinch"

    # 핀치 줌 제스처 (히스테리시스 없이 단순하게)
    if (ratio_16_20 < zoom_threshold) and (ratio_8_12 < pointer_ratio) and (ratio_12_16 > pointer_ratio * 1.2):
        return "pinch_zoom"

    return "pointer"


# --- function End ---

# websocket connection function(Continuous connectivity)
def safe_send(ws_connection, data):
    try:
        ws_connection.send(data)
        return True
    except (BrokenPipeError, ConnectionResetError, websocket.WebSocketConnectionClosedException) as e:
        print(f"\n✗ WebSocket connect refused: {e}")
        print("→ reconnection trying")
        return False
    except Exception as e:
        print(f"\n✗ send err: {e}")
        return False


with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
) as hands:
    # 비율 기반 임계값 (정규화된 값)
    PINCH_LOW = 0.25  # 핀치 진입: 손 크기의 25% 이하
    PINCH_HIGH = 0.50  # 핀치 해제: 손 크기의 50% 이상
    ZOOM_THRESHOLD = 0.22  # 줌: 손 크기의 22% 이하
    POINTER_RATIO = 0.30  # 포인터 판단용

    last_send_time = 0
    THROTTLE_INTERVAL = 0.016

    current_gesture_state = "none"
    previous_gesture = "none"
    gesturekey = ""

    while cap.isOpened():
        current_time = time.time()

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_global_action = "none"

        action_payload = {
            "action": "none", "x": 0.0, "y": 0.0, "current_dist": 0.0
        }

        if result.multi_hand_landmarks:
            all_hands_data = []
            tip_ids = [0, 4, 8, 12, 16, 20]  # 0번(손목) 추가

            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                landmarks_of_interest = extract_landmarks(hand_landmarks, tip_ids)

                gesture = "none"

                if hand_label == "Right":
                    gesture = recognize_single_hand_gesture(landmarks_of_interest, PINCH_LOW, PINCH_HIGH,
                                                            ZOOM_THRESHOLD, POINTER_RATIO, previous_gesture)

                    if gesture == "pinch_zoom":
                        current_dist = get_dict_distance(landmarks_of_interest.get("4"),
                                                         landmarks_of_interest.get("8"))
                        zoom_center = landmarks_of_interest.get("4")

                        action_payload["action"] = "pinch_zoom"
                        action_payload["current_dist"] = current_dist if current_dist is not None else 0.0
                        action_payload["x"] = zoom_center["x"] if zoom_center else 0.0
                        action_payload["y"] = zoom_center["y"] if zoom_center else 0.0

                    elif gesture == "pinch":
                        current_pinch_pos = landmarks_of_interest.get("4")
                        action_payload["action"] = "pinch"
                        if current_pinch_pos:
                            action_payload["x"] = current_pinch_pos["x"]
                            action_payload["y"] = current_pinch_pos["y"]

                    elif gesture == "pointer":
                        pointer_pos = landmarks_of_interest.get("8")
                        action_payload["action"] = "pointer"
                        if pointer_pos:
                            action_payload["x"] = pointer_pos["x"]
                            action_payload["y"] = pointer_pos["y"]

                    else:
                        action_payload["action"] = "none"

                    current_gesture_state = gesture
                    current_global_action = gesture

                elif hand_label == "Left" and current_global_action == "none":
                    gesture = recognize_single_hand_gesture(landmarks_of_interest, PINCH_LOW, PINCH_HIGH,
                                                            ZOOM_THRESHOLD, POINTER_RATIO, previous_gesture)

                    if gesture == "pinch":
                        current_pinch_pos = landmarks_of_interest.get("4")
                        action_payload["action"] = "left_pinch"
                        if current_pinch_pos:
                            action_payload["x"] = current_pinch_pos["x"]
                            action_payload["y"] = current_pinch_pos["y"]

                        current_gesture_state = "left_pinch"
                        current_global_action = "left_pinch"
                    else:
                        # 왼손이 pinch가 아니면 명확히 none
                        action_payload["action"] = "none"
                        current_gesture_state = "none"
                        current_global_action = "none"

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            # 최종 제스처 키 결정
            if current_global_action != "none":
                gesturekey = current_global_action
            else:
                gesturekey = current_gesture_state

            # pinch_zoom이 아니면 current_dist 제거
            if gesturekey != "pinch_zoom":
                if "current_dist" in action_payload:
                    del action_payload["current_dist"]

            # 전송 로직 (이전 방식 복원)
            is_none_message = (gesturekey == "none")

            if is_none_message or (current_time - last_send_time) >= THROTTLE_INTERVAL:
                if is_none_message and current_gesture_state == "none":
                    pass
                else:
                    json_data = json.dumps({gesturekey: action_payload})

                    if not safe_send(ws, json_data):
                        ws.close()
                        ws = connect_websocket()
                        safe_send(ws, json_data)

                    print(json_data)
                    last_send_time = current_time
                    previous_gesture = gesturekey
                    current_gesture_state = gesturekey

        else:
            if current_gesture_state != "none":
                print("---All gestures done! (No hands)---")
                current_global_action = "none"
                current_gesture_state = "none"
                previous_gesture = "none"

            json_data = json.dumps({"none": {"action": "none"}})
            if not safe_send(ws, json_data):
                ws.close()
                ws = connect_websocket()
                safe_send(ws, json_data)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
ws.close()
