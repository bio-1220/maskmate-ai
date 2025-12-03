"""
ResNet18 기반 표정 분류 모델 로딩 및 추론
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from typing import Tuple, List, Optional
import numpy as np

# 표정 클래스 정의 (기본값, checkpoint에서 덮어쓸 수 있음)
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
    - backbone: ResNet18의 fc 이전까지 (feature extractor)
    - expression_classifier: 표정 분류용 fc layer
    """
    def __init__(self, num_classes: int = NUM_CLASSES, classes: Optional[List[str]] = None):
        super().__init__()
        
        # 기본 ResNet18 구조 생성
        resnet = models.resnet18(weights=None)
        
        # backbone: fc 이전까지 (avgpool 포함)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 표정 분류 헤드
        self.expression_classifier = nn.Linear(512, num_classes)
        
        # 클래스 정보 저장
        self.classes = classes if classes is not None else EXPRESSION_CLASSES.copy()
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 입력 이미지 텐서 [B, 3, 224, 224]
        
        Returns:
            features: 512차원 임베딩 [B, 512]
            logits: 표정 분류 로짓 [B, num_classes]
        """
        # backbone으로 feature 추출
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # [B, 512]
        
        # 표정 분류
        logits = self.expression_classifier(features)
        
        return features, logits
    
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
            features, logits = self.forward(x)
            pred_idx = torch.argmax(logits, dim=1).item()
        return pred_idx, self.classes[pred_idx]


def load_model(checkpoint_path: str, device: torch.device) -> ExpressionResNet18:
    """
    체크포인트에서 모델 로드
    
    지원하는 체크포인트 형식:
    1) {"state_dict": <resnet18 state_dict>, "classes": ["angry", "happy", ...]}
    2) 직접 state_dict (torchvision resnet18 키 형식)
    
    Args:
        checkpoint_path: .pth 파일 경로
        device: torch device (cuda 또는 cpu)
    
    Returns:
        로드된 모델 (eval 모드)
    """
    # 체크포인트 로드
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # state_dict와 classes 추출
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        classes = ckpt.get("classes", None)
    else:
        state_dict = ckpt
        classes = None
    
    # classes가 있으면 전역 변수도 업데이트
    if classes is not None:
        global EXPRESSION_CLASSES
        EXPRESSION_CLASSES = classes
    
    # 임시 resnet18에 state_dict 로드
    resnet = models.resnet18(weights=None)
    
    # fc layer의 출력 차원 확인
    fc_weight_key = 'fc.weight'
    if fc_weight_key in state_dict:
        num_classes = state_dict[fc_weight_key].shape[0]
    else:
        num_classes = NUM_CLASSES
    
    # fc layer 교체 (num_classes가 다를 수 있음)
    resnet.fc = nn.Linear(512, num_classes)
    
    # state_dict 로드
    resnet.load_state_dict(state_dict, strict=True)
    
    # ExpressionResNet18 인스턴스 생성 및 구조 복사
    model = ExpressionResNet18(
        num_classes=num_classes,
        classes=classes if classes else EXPRESSION_CLASSES
    )
    
    # backbone과 classifier를 로드된 resnet에서 복사
    model.backbone = nn.Sequential(*list(resnet.children())[:-1])
    model.expression_classifier = resnet.fc
    
    # device로 이동 및 eval 모드
    model = model.to(device)
    model.eval()
    
    print(f"✓ 모델 로드 완료: {checkpoint_path}")
    print(f"✓ Device: {device}")
    print(f"✓ Classes: {model.classes}")
    
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
        features, logits = model(tensor)
        
        # L2 정규화
        features = F.normalize(features, p=2, dim=1)
        
        # 표정 예측
        expression_idx = torch.argmax(logits, dim=1).item()
        expression_label = model.classes[expression_idx]
        
        embedding = features.cpu().numpy().flatten()
    
    return embedding, expression_idx, expression_label
