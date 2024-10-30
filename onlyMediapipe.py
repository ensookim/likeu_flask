import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# MediaPipe 포즈 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 비디오 파일 열기
cap = cv2.VideoCapture('./test/curry.mp4')

# 각도 데이터와 키포인트 좌표 저장 리스트
data_list = []

# 포즈 좌표를 위한 변수들 초기화
shooting = False  # 슛 중인지 상태
shoot_start_frame = None  # 슛 시작 프레임
max_hand_y = float('inf')

# 스켈레톤 연결 인덱스 정의 (머리와 얼굴 키포인트 제외)
skeleton_pairs = [
    (11, 13), (13, 15),                     # 왼쪽 팔
    (15, 17), (15, 21), (15, 19), (17, 19), # 왼쪽 손목과 손 관절  
    (12, 14), (14, 16),                     # 오른쪽 팔
    (16, 22), (16, 18), (16, 20), (18, 20), # 오른쪽 손목과 손 관절
    (11, 12), (11, 23), (23, 24), (12, 24), # 몸통
    (24, 26), (26, 28), (28, 32), (28, 30), (30, 32),   # 오른쪽 다리
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),   # 왼쪽 다리
]

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중간 점
    c = np.array(c)  # 세 번째 점
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 포즈 추정 및 RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_pose = pose.process(frame_rgb)

    # 포즈 및 좌표 처리
    if result_pose.pose_landmarks:
        # 현재 프레임에서 관절 좌표 가져오기
        current_landmarks = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                             for landmark in result_pose.pose_landmarks.landmark]

        # 왼쪽, 오른쪽 팔꿈치 각도 계산
        left_elbow_angle = calculate_angle(current_landmarks[11], current_landmarks[13], current_landmarks[15])
        right_elbow_angle = calculate_angle(current_landmarks[12], current_landmarks[14], current_landmarks[16])

        # 무릎 각도 계산
        left_knee_angle = calculate_angle(current_landmarks[11], current_landmarks[23], current_landmarks[25])
        right_knee_angle = calculate_angle(current_landmarks[12], current_landmarks[24], current_landmarks[26])

        # 슛 시작 판별 (팔꿈치 각도가 90도 이하인 경우)
        if (left_elbow_angle < 90 or right_elbow_angle < 90) and not shooting:
            shooting = True
            shoot_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # 슛 중인 상태에서는 각도 및 키포인트 데이터를 매 프레임 저장
        if shooting:
            data_list.append({
                'Frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                'Left Elbow Angle': left_elbow_angle,
                'Right Elbow Angle': right_elbow_angle,
                'Left Knee Angle': left_knee_angle,
                'Right Knee Angle': right_knee_angle,
                # 키포인트 좌표 추가
                'Left Shoulder X': current_landmarks[11][0],
                'Left Shoulder Y': current_landmarks[11][1],
                'Right Shoulder X': current_landmarks[12][0],
                'Right Shoulder Y': current_landmarks[12][1],
                'Left Elbow X': current_landmarks[13][0],
                'Left Elbow Y': current_landmarks[13][1],
                'Right Elbow X': current_landmarks[14][0],
                'Right Elbow Y': current_landmarks[14][1],
                'Left Hip X': current_landmarks[23][0],
                'Left Hip Y': current_landmarks[23][1],
                'Right Hip X': current_landmarks[24][0],
                'Right Hip Y': current_landmarks[24][1],
                # 필요에 따라 더 많은 키포인트 추가 가능
            })

            # 슛 종료 조건 (손의 높이를 기준으로 판단)
            if max_hand_y > min(current_landmarks[15][1], current_landmarks[16][1]):
                max_hand_y = min(max_hand_y, min(current_landmarks[15][1], current_landmarks[16][1]))
            else:
                # 슛 종료 시각화 및 기록 중단
                shooting = False

        # 스켈레톤 그리기
        for pair in skeleton_pairs:
            start_idx, end_idx = pair
            if start_idx < len(current_landmarks) and end_idx < len(current_landmarks):
                cv2.line(frame, current_landmarks[start_idx], current_landmarks[end_idx], (255, 0, 0), 2)
                cv2.circle(frame, current_landmarks[start_idx], 5, (0, 255, 0), -1)

        # 프레임 표시
        cv2.imshow("skeleton", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# CSV 파일로 저장
df = pd.DataFrame(data_list)
df.to_csv('shoot_angles_with_coordinates.csv', index=False)
print("각도 및 키포인트 데이터가 'shoot_angles_with_coordinates.csv'로 저장되었습니다.")
