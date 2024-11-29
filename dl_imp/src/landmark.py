import os
from pathlib import Path
import cv2
import mediapipe as mp
import csv

# 각 landmark의 이름 목록
landmark_names = [
    "Nose", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Index",
    "Right Index", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel",
    "Right Heel", "Left Foot Index", "Right Foot Index"
]

class PoseEstimator:
    def __init__(self, video_path, output_path, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.video_path = video_path
        self.output_path = Path(output_path)
        self.label = Path(video_path).parent.stem  # 폴더명을 레이블로 설정
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        self.width, self.height, self.fps = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.cap.get(cv2.CAP_PROP_FPS),
        )
        
        self.video_writer = cv2.VideoWriter(
            str(self.output_path / f'{Path(video_path).stem}_output.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

    def extract_pose(self):
        # CSV 파일에 저장할 경로 설정
        csv_output_path = self.output_path / f'{Path(self.video_path).stem}_pose_landmarks.csv'
        
        with open(csv_output_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # CSV 파일의 헤더 작성
            header = ["Label", "Frame"]
            for name in landmark_names:
                header.extend([f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_visibility'])
            writer.writerow(header)
            
            frame_count = 0
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                # 프레임을 RGB로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                # 포즈 랜드마크가 감지된 경우
                if results.pose_landmarks:
                    row = [self.label, frame_count]
                    for lm in results.pose_landmarks.landmark:
                        # 각 랜드마크의 x, y, z, visibility 값을 추가
                        row.extend([lm.x, lm.y, lm.z, lm.visibility])
                    writer.writerow(row)

                    # 랜드마크를 영상 위에 그리기
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                    )

                # 결과 비디오에 프레임 추가
                self.video_writer.write(frame)
                frame_count += 1

        # 리소스 해제
        self.release_resources()

    def release_resources(self):
        self.cap.release()
        self.video_writer.release()
        self.pose.close()

def process_videos(input_folder, output_folder):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for video_file in Path(input_folder).glob("*.mp4"):
        estimator = PoseEstimator(
            video_path=str(video_file),
            output_path=output_path,
        )
        estimator.extract_pose()
        print(f"{video_file} 생성 완료")

    print("모든 영상 생성 완료")

# 예시 호출
process_videos(
    input_folder='./data/Step',
    output_folder='./data/Step/output_3',
)
