"""
FastAPI 백엔드 서버
- 탈 추천 API 제공
- Fine-tuned vs Baseline 비교 투표 기능
"""
import os
import json
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import io

from model import load_model, extract_embedding, EXPRESSION_CLASSES, preprocess
from mask_index import build_mask_index, compute_recommendations, MaskInfo


# ============== 설정 ==============
MODEL_PATH = os.environ.get("MODEL_PATH", "expression_resnet18_best.pth")
MASKS_DIR = os.environ.get("MASKS_DIR", "masks")
VOTES_DIR = os.environ.get("VOTES_DIR", "votes")
VOTES_FILE = os.path.join(VOTES_DIR, "votes.json")
TOP_K = int(os.environ.get("TOP_K", "3"))
COSINE_WEIGHT = float(os.environ.get("COSINE_WEIGHT", "0.7"))
EXPRESSION_WEIGHT = float(os.environ.get("EXPRESSION_WEIGHT", "0.3"))

# ============== 전역 변수 ==============
device: torch.device = None
model = None  # Fine-tuned model
baseline_model = None  # Baseline (pretrained) ResNet18
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
    baseline_top1: MaskRecommendation | None = None  # Baseline 1등


class VoteRequest(BaseModel):
    vote: str  # "finetuned" or "baseline"


class VoteResponse(BaseModel):
    finetuned: int
    baseline: int
    total: int


# ============== 투표 관련 함수 ==============
def load_votes() -> dict:
    """투표 결과 로드 (파일 없으면 초기화)"""
    if not os.path.exists(VOTES_FILE):
        return {"finetuned": 0, "baseline": 0}
    try:
        with open(VOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"finetuned": 0, "baseline": 0}


def save_votes(votes: dict):
    """투표 결과 저장"""
    os.makedirs(VOTES_DIR, exist_ok=True)
    with open(VOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(votes, f, ensure_ascii=False, indent=2)


# ============== Baseline 모델 관련 ==============
class BaselineResNet18(nn.Module):
    """순수 ImageNet pretrained ResNet18 (fine-tuning 없음)"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # feature extractor (fc 제외)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # 표정 분류는 random (4 classes)
        self.expression_classifier = nn.Linear(512, 4)
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        expression_logits = self.expression_classifier(features)
        return features, expression_logits


def extract_baseline_embedding(model, image, device):
    """Baseline 모델로 임베딩 추출"""
    model.eval()
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features, logits = model(img_tensor)
        features = features.squeeze(0)
        features = features / features.norm()
        
        # Baseline은 표정 분류가 random이므로 cosine만 의미있음
        expr_idx = logits.argmax(dim=1).item()
        expr_label = EXPRESSION_CLASSES[expr_idx]
    
    return features.cpu().numpy(), expr_idx, expr_label


# build_baseline_mask_index 제거 - baseline도 같은 mask_index 사용


# ============== Lifespan (서버 시작/종료 시 실행) ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모델과 탈 인덱스 로드"""
    global device, model, baseline_model, mask_index
    
    print("=" * 50)
    print("🎭 탈 추천 서버 초기화 중...")
    print("=" * 50)
    
    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Fine-tuned 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"⚠ 경고: 모델 파일이 없습니다: {MODEL_PATH}")
        print("  → 모델 파일을 backend/ 폴더에 넣어주세요.")
    else:
        model = load_model(MODEL_PATH, device)
        mask_index = build_mask_index(MASKS_DIR, model, device)
    
    # Baseline 모델 로드
    print("Baseline ResNet18 로드 중...")
    baseline_model = BaselineResNet18().to(device)
    baseline_model.eval()
    print("✓ Baseline 모델 로드 완료")
    
    # 투표 디렉토리 생성
    os.makedirs(VOTES_DIR, exist_ok=True)
    
    print("=" * 50)
    print("✓ 서버 준비 완료!")
    print(f"  - 탈 이미지: {len(mask_index)}개")
    print("=" * 50)
    
    yield
    
    print("서버 종료 중...")


# ============== FastAPI 앱 ==============
app = FastAPI(
    title="탈 추천 API",
    description="얼굴 사진 기반 한국 전통 탈 추천 서비스 + 모델 비교 투표",
    version="1.1.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "baseline_loaded": baseline_model is not None,
        "mask_count": len(mask_index)
    }


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(file: UploadFile = File(...)):
    """
    얼굴 이미지를 받아 어울리는 탈을 추천
    Fine-tuned 모델 결과 + Baseline 1등도 함께 반환
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
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드 가능합니다."
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Fine-tuned 모델 추천
        face_embedding, face_expr_idx, face_expr_label = extract_embedding(
            model, image, device
        )
        
        recommendations = compute_recommendations(
            face_embedding=face_embedding,
            face_expression_idx=face_expr_idx,
            mask_index=mask_index,
            top_k=TOP_K,
            cosine_weight=COSINE_WEIGHT,
            expression_weight=EXPRESSION_WEIGHT
        )
        
        # Baseline 모델 추천 (1등만) - 같은 mask_index 사용, 얼굴만 baseline으로
        baseline_top1 = None
        if baseline_model is not None and mask_index:
            baseline_embedding, baseline_expr_idx, _ = extract_baseline_embedding(
                baseline_model, image, device
            )
            baseline_recs = compute_recommendations(
                face_embedding=baseline_embedding,
                face_expression_idx=baseline_expr_idx,
                mask_index=mask_index,  # 같은 mask_index 사용
                top_k=1,
                cosine_weight=1.0,  # Baseline은 cosine만 사용
                expression_weight=0.0
            )
            if baseline_recs:
                baseline_top1 = MaskRecommendation(**baseline_recs[0])
        
        return RecommendationResponse(
            face_expression=face_expr_label,
            recommendations=[MaskRecommendation(**rec) for rec in recommendations],
            baseline_top1=baseline_top1
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/vote", response_model=VoteResponse)
async def vote(request: VoteRequest):
    """투표 저장"""
    if request.vote not in ["finetuned", "baseline"]:
        raise HTTPException(status_code=400, detail="Invalid vote. Use 'finetuned' or 'baseline'.")
    
    votes = load_votes()
    votes[request.vote] += 1
    save_votes(votes)
    
    return VoteResponse(
        finetuned=votes["finetuned"],
        baseline=votes["baseline"],
        total=votes["finetuned"] + votes["baseline"]
    )


@app.get("/votes", response_model=VoteResponse)
async def get_votes():
    """현재 투표 결과 조회"""
    votes = load_votes()
    return VoteResponse(
        finetuned=votes["finetuned"],
        baseline=votes["baseline"],
        total=votes["finetuned"] + votes["baseline"]
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
