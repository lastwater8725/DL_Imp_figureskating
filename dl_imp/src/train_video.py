import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.models.video import r3d_18
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class Normalize255(object):
    def __call__(self, tensor):
        """
        입력: (C, H, W) 형태의 텐서
        """
        return tensor.float() / 255.0

class CustomVideoDataset(Dataset):
    def __init__(self, video_dir, transform=None, num_frames=16):  
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames  # 사용할 프레임 수 지정
        self.video_files = []
        self.labels = []
        
        for class_name in os.listdir(video_dir):
            class_dir = os.path.join(video_dir, class_name)
            if os.path.isdir(class_dir):
                for video_file in os.listdir(class_dir):
                    if video_file.endswith(('.mp4', '.avi')):
                        self.video_files.append(os.path.join(class_dir, video_file))
                        self.labels.append(class_name)
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(set(self.labels))}
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        # 비디오 로드
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        # 프레임 수 조정
        total_frames = len(frames)
        if total_frames == 0:
            raise ValueError(f"No frames found in video: {video_path}")
            
        # 균일한 간격으로 프레임 선택
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = [frames[i] for i in indices]
        
        # 텐서로 변환
        frames = torch.FloatTensor(np.array(frames))
        
        # 차원 순서 변경:
        frames = frames.permute(3, 0, 1, 2)  # (C, T, H, W)
        
        if self.transform:
            # 각 프레임별로 transform 적용
            frame_list = []
            for t in range(frames.size(1)):
                frame = frames[:, t, :, :]  # (C, H, W)
                frame = self.transform(frame)  # transform 적용
                frame_list.append(frame)
            frames = torch.stack(frame_list, dim=1)  # (C, T, H, W)로 다시 조합
        
        return frames, label

# 데이터 전처리 정의 수정
transform = Compose([
    Resize((128, 128)),
    Normalize255(),
    CenterCrop((112, 112)),
    Normalize(mean=[0.43216, 0.394666, 0.37645], 
             std=[0.22803, 0.22145, 0.216989])
]) 

# 데이터셋 생성 (프레임 수 지정)
dataset = CustomVideoDataset(
    video_dir="./datasets/", 
    transform=transform,
    num_frames=16  
)


# Train/Test 분할
train_indices, test_indices = train_test_split(
    range(len(dataset)), 
    test_size=0.2, 
    random_state=42, 
    stratify=dataset.labels
)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# DataLoader 생성 
train_loader = DataLoader(
    train_dataset, 
    batch_size=8, 
    shuffle=True, 
    num_workers=0,  
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=8, 
    shuffle=False, 
    num_workers=0,  
    pin_memory=True
)
# 데이터셋 생성 후 정보 출력
print("\nDataset Info:")
print(f"Total number of videos: {len(dataset)}")
print(f"Number of classes: {len(dataset.class_to_idx)}")
print(f"Class mapping: {dataset.class_to_idx}")
print(f"Train set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# 첫 번째 배치 shape 확인
for videos, labels in train_loader:
    print(f"\nBatch shapes:")
    print(f"Videos: {videos.shape}")  
    print(f"Labels: {labels.shape}")
    break


# 모델 설정
model = r3d_18(pretrained=True)
num_classes = len(dataset.class_to_idx)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def evaluate(model, data_loader, device, class_names):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(data_loader, desc='Evaluating'):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 정확도 계산
    accuracy = 100 * correct / total
    

    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # 혼동 행렬 생성
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, cm, all_preds, all_labels

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# 학습 루프
num_epochs = 10
best_accuracy = 0
class_names = ['jump', 'spin', 'step']  # 클래스 이름 설정


try:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for videos, labels in pbar:
                # GPU로 데이터 이동
                videos = videos.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # 평가
        test_accuracy, test_report, conf_matrix, _, _ = evaluate(
            model, test_loader, device, class_names)
        
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print(f"Loss: {epoch_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Time: {epoch_time:.2f}s")
        print("\nClassification Report:")
        print(test_report)
        
        # 최고 성능 모델 저장
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_accuracy,
            }, 'best_model.pth')
            
            # 최고 성능 모델의 혼동 행렬 저장
            plot_confusion_matrix(conf_matrix, class_names)

except Exception as e:
    print(f"Error occurred: {str(e)}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'emergency_checkpoint.pth')

finally:
    print("\nFinal Evaluation:")
    final_accuracy, final_report, final_cm, _, _ = evaluate(
        model, test_loader, device, list(dataset.class_to_idx.keys()))
    print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")