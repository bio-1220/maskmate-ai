"""
탈 이미지 인덱싱 모듈
- masks/ 폴더의 모든 탈 이미지를 로드하고 임베딩 계산
- flat 구조 (masks/각시_natural.jpg) 및 서브폴더 구조 모두 지원
- 메모리에 캐싱
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch

from model import ExpressionResNet18, extract_embedding, EXPRESSION_CLASSES


# 지원하는 이미지 확장자
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


@dataclass
class MaskInfo:
    """탈 정보를 담는 데이터 클래스"""
    path: str               # 상대 경로 (masks/각시_natural.jpg 또는 masks/yangban/mask1.jpg)
    name: str               # 탈 이름 (파일명 또는 폴더명 기반)
    embedding: np.ndarray   # 512차원 임베딩
    expression_idx: int     # 모델이 분류한 표정 인덱스
    expression_label: str   # 모델이 분류한 표정 레이블
    filename_expression: Optional[str] = None  # 파일명에서 추출한 표정 (디버그용)


def parse_mask_filename(filename: str) -> Tuple[str, Optional[str]]:
    """
    파일명에서 마스크 이름과 표정을 파싱
    
    파일명 형식: "마스크이름_표정.확장자"
    예: "각시_natural.jpg" → ("각시", "natural")
        "양반_sad.png" → ("양반", "sad")
        "처용탈.jpg" → ("처용탈", None)
    
    Args:
        filename: 파일명 (확장자 포함)
    
    Returns:
        (mask_name, expression_label or None)
    """
    # 확장자 제거
    stem = Path(filename).stem
    
    # 마지막 "_"를 기준으로 분리
    if '_' in stem:
        # 마지막 "_" 위치 찾기
        last_underscore = stem.rfind('_')
        mask_name = stem[:last_underscore]
        expression = stem[last_underscore + 1:].lower()
        
        # 유효한 표정인지 확인
        valid_expressions = {'angry', 'happy', 'natural', 'sad'}
        if expression in valid_expressions:
            return mask_name, expression
        else:
            # "_"가 있지만 표정이 아닌 경우 (예: "하회_양반.jpg")
            return stem, None
    else:
        return stem, None


def build_mask_index(
    masks_dir: str,
    model: ExpressionResNet18,
    device: torch.device
) -> List[MaskInfo]:
    """
    masks/ 폴더 아래의 모든 탈 이미지를 인덱싱
    
    지원하는 폴더 구조:
    1) Flat 구조:
        masks/
            각시_natural.jpg
            양반_sad.jpg
            ...
    
    2) 서브폴더 구조:
        masks/
            yangban/
                yangban1.jpg
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
    
    print(f"탈 이미지 인덱싱 시작: {masks_dir}")
    
    # 모든 이미지 파일을 재귀적으로 찾기
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(masks_path.glob(f"**/*{ext}"))
        image_files.extend(masks_path.glob(f"**/*{ext.upper()}"))
    
    # 정렬
    image_files = sorted(set(image_files))
    
    print(f"  발견된 이미지 파일: {len(image_files)}개")
    
    for image_file in image_files:
        try:
            # 상대 경로 계산
            relative_path = str(image_file.relative_to(masks_path))
            
            # 마스크 이름 결정
            # 1) 서브폴더가 있으면 폴더명 사용
            # 2) flat 구조면 파일명에서 파싱
            parts = relative_path.split(os.sep)
            
            if len(parts) > 1:
                # 서브폴더 구조: 폴더명을 마스크 이름으로
                mask_name = parts[0]
                filename_expression = None
            else:
                # flat 구조: 파일명에서 파싱
                mask_name, filename_expression = parse_mask_filename(image_file.name)
            
            # 이미지 로드 및 임베딩 추출
            image = Image.open(image_file)
            embedding, expr_idx, expr_label = extract_embedding(model, image, device)
            
            mask_info = MaskInfo(
                path=relative_path,
                name=mask_name,
                embedding=embedding,
                expression_idx=expr_idx,
                expression_label=expr_label,
                filename_expression=filename_expression
            )
            mask_index.append(mask_info)
            
            # 로그 출력
            if filename_expression:
                match_status = "✓" if filename_expression == expr_label else "≠"
                print(f"  ✓ {relative_path} - 모델: {expr_label}, 파일명: {filename_expression} {match_status}")
            else:
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
