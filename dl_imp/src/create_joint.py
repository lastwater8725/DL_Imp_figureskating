import cv2
import mediapipe as mp
import csv
from pathlib import Path


landmark_names = [
    "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner",
    "Right Eye", "Right Eye Outer", "Left Ear", "Right Ear", "Mouth Left",
    "Mouth Right", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Index",
    "Right Index", "Left Thumb", "Right Thumb", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel",
    "Right Heel", "Left Foot Index", "Right Foot Index"
]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class PoseEstimator:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = Path(output_path)
        self.pose = mp.solutions.pose
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"파일 못 염:{input_path}")
        
        self.width, self.height, self.fps = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.cap.get(cv2.CAP_PROP_FPS),
        )
        
        self.video_writer = cv2.VideoWriter(
            str(self.output_path / f'{Path(input_path).stem}_output.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

    def create_csv(self):
            csv_output_path = open('pose_landmars.csv', mode = 'w', newline='')
            csv_writer = csv.writer(csv_output_path)
            header = []
            for name in landmark_names:
                    header.extend([f'{name}_x', f'{name}_y'])
            csv_writer.writerow(header)

            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                     break
                
                video_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(video_rgb)

                if results.pose_landmarks:
                    row = []
                    for landmark in results.pose_landmarks.landmark:
                        row.extend([landmark.x, landmark.y])
                    csv_writer.writerow(row)
            
        
    def release_resources(self):
        self.cap.release()
        self.video_writer.release()
        self.pose.close()


def process_videos(input_folder, output_folder):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for video_file in Path(input_folder).glob("*.mp4"):
        estimator = PoseEstimator(
            input_path=str(video_file),
            output_path=output_path,
        )
        estimator.create_csv()
        print(f"{video_file} 생성 완료")


    print("모든 영상 생성완료")
            
# 예시 호출
process_videos(
    input_folder='./data/Step',
    output_folder='./data/step/output_2',
)





