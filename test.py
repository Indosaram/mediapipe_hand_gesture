import time

import cv2
import mediapipe as mp


# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Initializing the drawng utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils
# (0) in VideoCapture is used to connect to your computer's default camera
# video_path = "video.mp4"
video_path = 0
capture = cv2.VideoCapture(video_path)

is_cap_gesture = False
is_captured = False
right_hand_capture_criteria = [
    0.0039673909279764885,
    0.031911073797311396,
    0.041649862650306346,
    0.03400417985501125,
    0.018600322025566385,
]

right_hand_size = 0.13


prev_pos = [None, None]
start_time = time.time()
while capture.isOpened():
    # capture frame by frame

    ret, frame = capture.read()

    if not ret:
        capture = cv2.VideoCapture(video_path)
        ret, frame = capture.read()
    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Converting the from from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic_model.process(image)

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the Facial Landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(
            color=(255, 0, 255), thickness=1, circle_radius=1
        ),
        mp_drawing.DrawingSpec(
            color=(0, 255, 255), thickness=1, circle_radius=1
        ),
    )
    # Drawing Right hand Land Marks
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )

    # Drawing Left hand Land Marks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )

    if is_cap_gesture:
        cv2.putText(
            image,
            "Get ready for the action",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 0),
            2,
        )
    elif is_captured:
        cv2.putText(
            image,
            "The action is recognized",
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 0),
            2,
        )

    # if s is pressed, start capturing the gesture
    if cv2.waitKey(5) & 0xFF == ord('s'):
        if results.right_hand_landmarks is not None:
            is_cap_gesture = True
            cap_start_time = time.time()
            right_hand_x = []
            right_hand_y = []
            right_hand_center = results.right_hand_landmarks.landmark[9]
            for landmark in [4, 8, 12, 16, 20]:
                right_hand = results.right_hand_landmarks.landmark[landmark]
                right_hand_x.append(right_hand.x)
                right_hand_y.append(right_hand.y)

    if is_cap_gesture and results.right_hand_landmarks is not None:
        is_captured = []
        right_hand_center_after = results.right_hand_landmarks.landmark[9]
        right_hand_size_after = (
            results.right_hand_landmarks.landmark[4].x
            - results.right_hand_landmarks.landmark[20].x
        )
        for idx, landmark in enumerate([4, 8, 12, 16, 20]):
            right_hand = results.right_hand_landmarks.landmark[landmark]
            if (
                (
                    (right_hand.x - right_hand_x[0]) ** 2
                    + (right_hand.y - right_hand_y[0] ** 2)
                )
                * right_hand_size_after
                / right_hand_size
                >= right_hand_capture_criteria[idx]
                and (right_hand_center.x - right_hand_center_after.x) ** 2
                + (right_hand_center.y - right_hand_center_after.y) ** 2
                <= 0.0005
            ):
                is_captured.append(True)
            else:
                is_captured.append(False)

        if not all(is_captured):
            right_hand_landmarks = results.right_hand_landmarks.landmark
            if (
                right_hand_landmarks[8].y > right_hand_center_after.y
                and right_hand_landmarks[12].y > right_hand_center_after.y
                and right_hand_landmarks[16].y > right_hand_center_after.y
                and right_hand_landmarks[20].y > right_hand_center_after.y
            ):
                is_cap_gesture = False
        elif is_captured and all(is_captured):
            is_cap_gesture = False

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)
    time.sleep(0.05)
    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
