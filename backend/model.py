"""
ResNet18 기반 표정 분류 모델 로딩 및 추론
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from typing import Tuple, Optional
import numpy as np

# 표정 클래스 정의 (모델 학습 시 사용된 순서와 일치해야 함)
EXPRESSION_CLASSES = ['angry', 'happy', 'natural', 'sad']
NUM_CLASSES = len(EXPRESSION_CLASSES)

# 이미지 전처리 transform (ImageNet 기준)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ExpressionResNet18(nn.Module):
    """
    ResNet18 기반 표정 분류 + Feature Extraction 모델
    - backbone: ResNet18의 conv layers (feature extractor)
    - fc: 표정 분류용 fully connected layer
    """
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False):
        super().__init__()
        
        # ImageNet pretrained ResNet18 로드
        base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # backbone: avgpool까지 (512차원 feature 출력)
        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
        )
        
        # 표정 분류 헤드
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 입력 이미지 텐서 [B, 3, 224, 224]
        
        Returns:
            logits: 표정 분류 로짓 [B, num_classes]
            features: 512차원 임베딩 [B, 512]
        """
        # backbone으로 feature 추출
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # [B, 512]
        
        # 표정 분류
        logits = self.fc(features)
        
        return logits, features
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature만 추출 (L2 정규화 포함)"""
        with torch.no_grad():
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
            features = F.normalize(features, p=2, dim=1)
        return features
    
    def predict_expression(self, x: torch.Tensor) -> Tuple[int, str]:
        """표정 예측"""
        with torch.no_grad():
            logits, _ = self.forward(x)
            pred_idx = torch.argmax(logits, dim=1).item()
        return pred_idx, EXPRESSION_CLASSES[pred_idx]


def load_model(checkpoint_path: str, device: torch.device) -> ExpressionResNet18:
    """
    체크포인트에서 모델 로드
    
    Args:
        checkpoint_path: .pth 파일 경로
        device: torch device (cuda 또는 cpu)
    
    Returns:
        로드된 모델 (eval 모드)
    """
    model = ExpressionResNet18(num_classes=NUM_CLASSES, pretrained=False)
    
    # state_dict 로드
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # 'module.' prefix 제거 (DataParallel로 학습된 경우)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ 모델 로드 완료: {checkpoint_path}")
    print(f"✓ Device: {device}")
    
    return model


def preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    PIL 이미지를 모델 입력 텐서로 변환
    
    Args:
        image: PIL Image
        device: torch device
    
    Returns:
        전처리된 텐서 [1, 3, 224, 224]
    """
    # RGB로 변환 (RGBA나 grayscale 처리)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    tensor = image_transform(image)
    tensor = tensor.unsqueeze(0)  # batch dimension 추가
    tensor = tensor.to(device)
    
    return tensor


def extract_embedding(
    model: ExpressionResNet18, 
    image: Image.Image, 
    device: torch.device
) -> Tuple[np.ndarray, int, str]:
    """
    이미지에서 임베딩과 표정 예측
    
    Args:
        model: 로드된 모델
        image: PIL Image
        device: torch device
    
    Returns:
        embedding: L2 정규화된 512차원 임베딩 (numpy array)
        expression_idx: 표정 클래스 인덱스
        expression_label: 표정 레이블 문자열
    """
    tensor = preprocess_image(image, device)
    
    with torch.no_grad():
        logits, features = model(tensor)
        
        # L2 정규화
        features = F.normalize(features, p=2, dim=1)
        
        # 표정 예측
        expression_idx = torch.argmax(logits, dim=1).item()
        expression_label = EXPRESSION_CLASSES[expression_idx]
        
        embedding = features.cpu().numpy().flatten()
    
    return embedding, expression_idx, expression_label
