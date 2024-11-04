import cv2
import mediapipe as mp
import csv

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 동영상 파일 경로
video_path = 'stretching.mp4'
cap = cv2.VideoCapture(video_path)

# CSV 파일 생성
csv_file = open('pose_landmarks.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)

landmark_names = [
    "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner",
    "Right Eye", "Right Eye Outer", "Left Ear", "Right Ear", "Mouth Left",
    "Mouth Right", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Index",
    "Right Index", "Left Thumb", "Right Thumb", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel",
    "Right Heel", "Left Foot Index", "Right Foot Index"
]

# CSV 파일 헤더 작성 (랜드마크 이름 + 각 좌표 (x, y))
header = []
for name in landmark_names:
    header.extend([f'{name}_x', f'{name}_y'])
csv_writer.writerow(header)

# 동영상 프레임 처리
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # BGR 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 포즈 추정 수행
    results = pose.process(image_rgb)

    # 랜드마크 추출 및 저장
    if results.pose_landmarks:
        row = []
        for landmark in results.pose_landmarks.landmark:
            row.extend([landmark.x, landmark.y])
        csv_writer.writerow(row)

# 리소스 해제
cap.release()
csv_file.close()
pose.close()
