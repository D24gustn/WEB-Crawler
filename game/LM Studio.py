import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm

# 텍스트 토크나이저 및 시퀀스 패딩
from tensorflow.keras.preprocessing.text     import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 모델 및 학습 도구
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 시각화
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ─────────────────────────────────────────────────────────────────────────────
# 1) LM Studio 설정
# ─────────────────────────────────────────────────────────────────────────────
API_URL    = "http://127.0.0.1:1234/v1/completions"
MODEL_NAME = "google/gemma-3-12b"

# ─────────────────────────────────────────────────────────────────────────────
# 2) 프롬프트 구성
# ─────────────────────────────────────────────────────────────────────────────
def make_batch_prompt(reviews):
    prompt = (
        "아래 리뷰 목록의 감정을 분류합니다. 가능한 값: 0(부정), 1(중립), 2(긍정)\n"
        "형식: 숫자만, 각 리뷰 앞에 번호와 함께 나열하세요.\n"
    )
    for i, r in enumerate(reviews, start=1):
        prompt += f"{i}. {r}\n"
    prompt += "답:"
    return prompt

# ─────────────────────────────────────────────────────────────────────────────
# 3) 응답 파싱
# ─────────────────────────────────────────────────────────────────────────────
def parse_response(text, expected):
    tokens = [t.strip() for t in text.replace(',', ' ').split()]
    vals = [int(t) for t in tokens if t.isdigit()]
    if len(vals) < expected:
        vals += [None] * (expected - len(vals))
    return vals[:expected]

# ─────────────────────────────────────────────────────────────────────────────
# 4) LM Studio로 감성 분류 요청
# ─────────────────────────────────────────────────────────────────────────────
def classify_batch(reviews):
    payload = {
        "model":       MODEL_NAME,
        "prompt":      make_batch_prompt(reviews),
        "max_tokens":  5,
        "temperature": 0.0
    }
    r = requests.post(API_URL, json=payload)
    r.raise_for_status()  # HTTP 오류 발생 시 예외
    resp = r.json()
    return parse_response(resp["choices"][0]["text"], len(reviews))

# ─────────────────────────────────────────────────────────────────────────────
# 5) CSV 파일 불러오기
# ─────────────────────────────────────────────────────────────────────────────
CSV_INPUT  = r"D:\_DeepNLP25\site\cache\merged_reviews.csv"
CSV_OUTPUT = r"D:\_DeepNLP25\site\cache\reviews_labeled.csv"

df = pd.read_csv(CSV_INPUT, encoding="utf-8-sig", low_memory=False)
if "label" not in df.columns:
    df["label"] = None
df["model_used"] = None

# ─────────────────────────────────────────────────────────────────────────────
# 6) 배치 처리 (체크포인트 로직 제거됨)
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 10
for start in tqdm(range(0, len(df), BATCH_SIZE), desc="LLM 분류 중"):
    end   = min(start + BATCH_SIZE, len(df))
    batch = df.iloc[start:end]

    # 이미 라벨링된 경우 건너뜀
    if batch["label"].notna().all():
        continue

    reviews = batch["review"].tolist()
    labels  = classify_batch(reviews)

    # 상위 몇 개 결과 확인 (디버그용)
    print(df.iloc[start : start + 3][["review", "label"]])
    
    # 이번 배치 중 부정(0), 중립(1)인 리뷰만 추려 보기 (옵션)
    filtered = df.iloc[start:end]
    subset  = filtered[filtered["label"].isin([0,1])]
    if not subset.empty:
        print("\n🔍 이번 배치 중 0/1 감정 리뷰:")
        print(subset[["review","label"]].head(5))

    # 라벨 및 모델명 업데이트
    for idx, lab in zip(batch.index, labels):
        df.at[idx, "label"]       = lab
        df.at[idx, "model_used"]  = "gemma-lmstudio"

# ─────────────────────────────────────────────────────────────────────────────
# 7) 결과 저장
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
print("✅ 감정 분류 및 저장 완료.")
