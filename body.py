import cv2
import mediapipe as mp
import os

str=('python codetest.py')
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Get Size of frame
        size = frame.shape  # 取得攝影機影像尺寸
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

            if x*640 > 130 and x*640 < 600 and y*480 > 60 and y*480 < 450 :
                print("yes")
                os.system(str)




        except:
            pass
        cv2.rectangle(image, (190, 60), (540, 185), (0, 0, 255), 2)  # 畫出觸碰區
        cv2.rectangle(image, (130, 180), (600,450), (0, 0, 255), 2)  # 畫出觸碰區
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()