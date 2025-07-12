import csv
import copy
import itertools
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model.keypoint_classifier import KeyPointClassifier

annotated_image = None

def draw_landmarks_on_image(image: np.ndarray, detection_result, number, mode, keypoint_classifier, keypoint_classifier_labels):
    annotated = image.copy()

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),# Ring
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17)  # Palm edge
    ]

    for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
        
        label = detection_result.handedness[i][0].category_name 
        corrected_label = 'Right' if label == 'Left' else 'Left' 
        color = (0, 0, 255) if corrected_label == 'Left' else (255, 0, 0)

        for start_idx, end_idx in HAND_CONNECTIONS:
            x1 = int(hand_landmarks[start_idx].x * image.shape[1])
            y1 = int(hand_landmarks[start_idx].y * image.shape[0])
            x2 = int(hand_landmarks[end_idx].x * image.shape[1])
            y2 = int(hand_landmarks[end_idx].y * image.shape[0])
            cv2.line(annotated, (x1, y1), (x2, y2), color, 3)

        for landmark in hand_landmarks:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)

        # --- Custom classification ---
        landmark_list = calc_landmark_list(annotated, hand_landmarks)
        pre_processed_landmark_list = pre_process_landmark(landmark_list)
        
        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
        if 0 <= hand_sign_id < len(keypoint_classifier_labels):
            gesture_name = keypoint_classifier_labels[hand_sign_id]
        else:
            gesture_name = "Unknown"

        cv2.putText(annotated, f"{corrected_label} - {gesture_name}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if mode == 1 and len(pre_processed_landmark_list) > 0:
            logging_csv(number, mode, pre_processed_landmark_list)

    if mode == 1:
        cv2.putText(annotated, f"Mode: Key Logging ", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(annotated, f"Number: {number}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
           
    return annotated



def make_result_callback(number_ref, mode_ref, classifier, labels):
    def callback(result, output_image, timestamp_ms):
        global annotated_image
        image = output_image.numpy_view()
        annotated_image = draw_landmarks_on_image(
            image,
            result,
            number_ref[0],
            mode_ref[0],
            classifier,
            labels  
        )
    return callback



def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 25):
        csv_path = 'model/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def select_mode(key, mode):
    number = -1
    
    if 97 <= key <= 122:  # a ~ z
        number = key - 97 # Convert ASCII to 0-25
    if key == 48:  # 0
        mode = 0
    if key == 49:  # 1
        mode = 1
    return number, mode    


def main():
    global annotated_image
    annotated_image = None

    model_path = r'C:\Users\chira\OneDrive\Desktop\Coding Projects\Hand Detection\gesture_recognizer.task' 
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    number = [-1]  # mutable reference
    mode = [0]

    keypoint_classifier = KeyPointClassifier()

 
    with open('model/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

   
    callback = make_result_callback(number, mode, keypoint_classifier, keypoint_classifier_labels)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=callback,
        num_hands=2
    )

    with GestureRecognizer.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(1)
        cap.set(3, 1280)  # Set width
        cap.set(4, 720)  # Set height

        while True:
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break

            number[0], mode[0] = select_mode(key, mode[0])

            successful, img = cap.read()
            img = cv2.flip(img, 1)
            if successful:
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                frame_timestamp_ms = int(time.time() * 1000)
                landmarker.recognize_async(mp_image, frame_timestamp_ms)

                if annotated_image is not None:
                    cv2.imshow("Gesture Recognizer", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imshow("Gesture Recognizer", img)

        cap.release()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
