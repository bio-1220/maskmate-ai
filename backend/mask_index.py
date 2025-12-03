"""
탈 이미지 인덱싱 모듈
- masks/ 폴더의 모든 탈 이미지를 로드하고 임베딩 계산
- 메모리에 캐싱
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch

from model import ExpressionResNet18, extract_embedding, EXPRESSION_CLASSES


@dataclass
class MaskInfo:
    """탈 정보를 담는 데이터 클래스"""
    path: str           # 상대 경로 (masks/yangban/mask1.jpg)
    name: str           # 탈 이름 (폴더명 기반)
    embedding: np.ndarray   # 512차원 임베딩
    expression_idx: int     # 표정 인덱스
    expression_label: str   # 표정 레이블


def build_mask_index(
    masks_dir: str,
    model: ExpressionResNet18,
    device: torch.device
) -> List[MaskInfo]:
    """
    masks/ 폴더 아래의 모든 탈 이미지를 인덱싱
    
    폴더 구조:
        masks/
            yangban/
                yangban1.jpg
                yangban2.png
            bune/
                bune1.jpg
            ...
    
    Args:
        masks_dir: 탈 이미지 폴더 경로
        model: 로드된 모델
        device: torch device
    
    Returns:
        MaskInfo 리스트
    """
    masks_path = Path(masks_dir)
    
    if not masks_path.exists():
        print(f"⚠ 경고: {masks_dir} 폴더가 없습니다. 빈 인덱스를 반환합니다.")
        return []
    
    mask_index: List[MaskInfo] = []
    supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    print(f"탈 이미지 인덱싱 시작: {masks_dir}")
    
    # 하위 폴더 순회 (각 폴더명 = 탈 종류)
    for category_dir in sorted(masks_path.iterdir()):
        if not category_dir.is_dir():
            continue
        
        mask_name = category_dir.name
        
        for image_file in sorted(category_dir.iterdir()):
            if image_file.suffix.lower() not in supported_extensions:
                continue
            
            try:
                # 이미지 로드 및 임베딩 추출
                image = Image.open(image_file)
                embedding, expr_idx, expr_label = extract_embedding(model, image, device)
                
                # 상대 경로 저장
                relative_path = str(image_file.relative_to(masks_path))
                
                mask_info = MaskInfo(
                    path=relative_path,
                    name=mask_name,
                    embedding=embedding,
                    expression_idx=expr_idx,
                    expression_label=expr_label
                )
                mask_index.append(mask_info)
                
                print(f"  ✓ {relative_path} - 표정: {expr_label}")
                
            except Exception as e:
                print(f"  ✗ {image_file}: {e}")
                continue
    
    print(f"인덱싱 완료: 총 {len(mask_index)}개 탈 이미지")
    
    return mask_index


def compute_recommendations(
    face_embedding: np.ndarray,
    face_expression_idx: int,
    mask_index: List[MaskInfo],
    top_k: int = 3,
    cosine_weight: float = 0.7,
    expression_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    얼굴 임베딩과 모든 탈 임베딩 비교하여 추천
    
    Args:
        face_embedding: L2 정규화된 얼굴 임베딩
        face_expression_idx: 얼굴 표정 인덱스
        mask_index: 탈 정보 리스트
        top_k: 반환할 추천 개수
        cosine_weight: 코사인 유사도 가중치 (기본 0.7)
        expression_weight: 표정 일치 가중치 (기본 0.3)
    
    Returns:
        추천 결과 리스트 (score 내림차순)
    """
    if not mask_index:
        return []
    
    results = []
    
    for mask in mask_index:
        # 코사인 유사도 계산 (L2 정규화되어 있으므로 dot product)
        cosine_sim = float(np.dot(face_embedding, mask.embedding))
        
        # 표정 일치 여부
        expression_match = int(face_expression_idx == mask.expression_idx)
        
        # 최종 점수
        combined_score = cosine_weight * cosine_sim + expression_weight * expression_match
        
        results.append({
            'mask_path': mask.path,
            'mask_name': mask.name,
            'cosine_similarity': cosine_sim,
            'expression_match': bool(expression_match),
            'mask_expression': mask.expression_label,
            'combined_score': combined_score
        })
    
    # 점수 내림차순 정렬
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return results[:top_k]
