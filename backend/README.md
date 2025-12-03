# 탈 추천 백엔드 서버

PyTorch ResNet18 기반 감정 분석 모델을 사용하여 한국 전통 탈을 추천하는 FastAPI 서버입니다.

## 📁 폴더 구조

```
backend/
├── main.py              # FastAPI 서버 메인
├── model.py             # ResNet18 모델 정의 및 로딩
├── mask_index.py        # 탈 이미지 인덱싱 로직
├── requirements.txt     # Python 의존성
├── README.md           # 이 파일
├── expression_resnet18_best.pth   # ← 여기에 모델 파일 넣기 (50MB)
└── masks/              # ← 여기에 탈 이미지 폴더 넣기
    ├── yangban/
    │   ├── yangban1.jpg
    │   └── yangban2.png
    ├── bune/
    │   └── bune1.jpg
    └── ...
```

## 🚀 설치 및 실행

### 1. 가상환경 생성 (권장)

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 모델 파일 배치

`expression_resnet18_best.pth` 파일을 `backend/` 폴더에 넣어주세요.

### 4. 탈 이미지 배치

`masks/` 폴더를 만들고 아래 구조로 탈 이미지를 넣어주세요:

```
masks/
├── yangban/      # 양반탈
│   ├── img1.jpg
│   └── img2.jpg
├── bune/         # 부네탈
│   └── img1.jpg
├── chwibal/      # 취발이탈
│   └── img1.jpg
└── ...
```

### 5. 서버 실행

```bash
# 개발 모드 (자동 reload)
python main.py

# 또는 uvicorn 직접 사용
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

서버가 시작되면 http://localhost:8000 에서 확인할 수 있습니다.

## 📡 API 엔드포인트

### `GET /`
헬스체크 및 서버 상태 확인

### `POST /recommend`
얼굴 이미지를 업로드하여 추천 받기

**Request:**
- `file`: 이미지 파일 (multipart/form-data)

**Response:**
```json
{
  "face_expression": "happy",
  "recommendations": [
    {
      "mask_path": "yangban/yangban1.jpg",
      "mask_name": "yangban",
      "cosine_similarity": 0.85,
      "expression_match": true,
      "mask_expression": "happy",
      "combined_score": 0.895
    },
    ...
  ]
}
```

### `GET /masks/{path}`
탈 이미지 파일 제공

### `GET /expressions`
사용 가능한 표정 클래스 목록

## ⚙️ 환경변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `MODEL_PATH` | `expression_resnet18_best.pth` | 모델 파일 경로 |
| `MASKS_DIR` | `masks` | 탈 이미지 폴더 경로 |
| `TOP_K` | `3` | 추천 결과 개수 |
| `COSINE_WEIGHT` | `0.7` | 코사인 유사도 가중치 |
| `EXPRESSION_WEIGHT` | `0.3` | 표정 일치 가중치 |

## 🔧 모델 구조

```
ExpressionResNet18
├── backbone (ResNet18 conv layers → 512차원)
└── fc (512 → 4 표정 클래스)
```

표정 클래스: `angry`, `happy`, `natural`, `sad`

## 📝 점수 계산

```
final_score = 0.7 × cosine_similarity + 0.3 × expression_match
```

- `cosine_similarity`: 얼굴 임베딩과 탈 임베딩의 코사인 유사도
- `expression_match`: 표정 일치 시 1, 불일치 시 0

## 🐛 문제 해결

### "모델 파일이 없습니다"
→ `expression_resnet18_best.pth`를 `backend/` 폴더에 넣어주세요.

### "탈 이미지가 인덱싱되지 않았습니다"
→ `masks/` 폴더와 하위 폴더에 이미지 파일이 있는지 확인하세요.

### GPU 메모리 부족
→ CPU로 자동 fallback됩니다. 서버 로그에서 Device 확인 가능.

### CUDA 관련 오류
→ PyTorch GPU 버전이 CUDA 버전과 호환되는지 확인하세요.
  CPU 전용으로 사용하려면: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
