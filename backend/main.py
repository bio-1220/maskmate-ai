"""
FastAPI 백엔드 서버
- 탈 추천 API 제공
"""
import os
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import io

from model import load_model, extract_embedding, EXPRESSION_CLASSES
from mask_index import build_mask_index, compute_recommendations, MaskInfo


# ============== 설정 ==============
MODEL_PATH = os.environ.get("MODEL_PATH", "expression_resnet18_best.pth")
MASKS_DIR = os.environ.get("MASKS_DIR", "masks")
TOP_K = int(os.environ.get("TOP_K", "3"))
COSINE_WEIGHT = float(os.environ.get("COSINE_WEIGHT", "0.7"))
EXPRESSION_WEIGHT = float(os.environ.get("EXPRESSION_WEIGHT", "0.3"))

# ============== 전역 변수 ==============
device: torch.device = None
model = None
mask_index: List[MaskInfo] = []


# ============== Pydantic 모델 ==============
class MaskRecommendation(BaseModel):
    mask_path: str
    mask_name: str
    cosine_similarity: float
    expression_match: bool
    mask_expression: str
    combined_score: float


class RecommendationResponse(BaseModel):
    face_expression: str
    recommendations: List[MaskRecommendation]


# ============== Lifespan (서버 시작/종료 시 실행) ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델과 탈 인덱스 로드"""
    global device, model, mask_index
    
    print("=" * 50)
    print("🎭 탈 추천 서버 초기화 중...")
    print("=" * 50)
    
    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"⚠ 경고: 모델 파일이 없습니다: {MODEL_PATH}")
        print("  → 모델 파일을 backend/ 폴더에 넣어주세요.")
    else:
        model = load_model(MODEL_PATH, device)
        
        # 탈 인덱스 빌드
        mask_index = build_mask_index(MASKS_DIR, model, device)
    
    print("=" * 50)
    print("✓ 서버 준비 완료!")
    print(f"  - 로드된 탈 이미지: {len(mask_index)}개")
    print("=" * 50)
    
    yield
    
    # 종료 시 정리
    print("서버 종료 중...")


# ============== FastAPI 앱 ==============
app = FastAPI(
    title="탈 추천 API",
    description="얼굴 사진 기반 한국 전통 탈 추천 서비스",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (프론트엔드와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서는 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== 엔드포인트 ==============
@app.get("/")
async def root():
    """헬스체크"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "mask_count": len(mask_index)
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(file: UploadFile = File(...)):
    """
    얼굴 이미지를 받아 어울리는 탈을 추천
    
    Args:
        file: 업로드된 얼굴 이미지 파일
    
    Returns:
        face_expression: 감지된 얼굴 표정
        recommendations: 추천된 탈 목록 (TOP_K개)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 로드되지 않았습니다. 모델 파일을 확인해주세요."
        )
    
    if not mask_index:
        raise HTTPException(
            status_code=503,
            detail="탈 이미지가 인덱싱되지 않았습니다. masks/ 폴더를 확인해주세요."
        )
    
    # 파일 타입 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드 가능합니다."
        )
    
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 임베딩 및 표정 추출
        face_embedding, face_expr_idx, face_expr_label = extract_embedding(
            model, image, device
        )
        
        # 추천 계산
        recommendations = compute_recommendations(
            face_embedding=face_embedding,
            face_expression_idx=face_expr_idx,
            mask_index=mask_index,
            top_k=TOP_K,
            cosine_weight=COSINE_WEIGHT,
            expression_weight=EXPRESSION_WEIGHT
        )
        
        return RecommendationResponse(
            face_expression=face_expr_label,
            recommendations=[MaskRecommendation(**rec) for rec in recommendations]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}"
        )


@app.get("/masks/{mask_path:path}")
async def get_mask_image(mask_path: str):
    """탈 이미지 파일 제공"""
    full_path = Path(MASKS_DIR) / mask_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    
    return FileResponse(full_path)


@app.get("/expressions")
async def get_expressions():
    """사용 가능한 표정 클래스 목록"""
    return {"expressions": EXPRESSION_CLASSES}


# ============== 실행 ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
