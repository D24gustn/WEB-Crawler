# app.py
import os
import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────
MODEL_H5       = "best_lstm.h5"
TOKENIZER_PKL  = "tokenizer.pkl"
CSV_LABELED    = r"D:\_DeepNLP25\site\cache\reviews_labeled.csv"
MAX_LEN        = 100
FONT_PATH      = "C:/Windows/Fonts/malgun.ttf"
LABEL_MAP      = {0: "부정", 1: "중립", 2: "긍정"}

# ─────────────────────────────────────────────────────────────────────────────
# 1) 모델·토크나이저·원본 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
model     = load_model(MODEL_H5)
tokenizer = pickle.load(open(TOKENIZER_PKL, "rb"))

df = pd.read_csv(CSV_LABELED, encoding="utf-8-sig", low_memory=False)
if df["label"].dtype == object:
    df["label"] = df["label"].map({"부정":0, "중립":1, "긍정":2})
df = df.dropna(subset=["review","label"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 2) 전처리 함수
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(s: str) -> str:
    s = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z0-9 ]", "", str(s))
    return s.lower().strip()

df["clean"] = df["review"].map(clean_text)

# ─────────────────────────────────────────────────────────────────────────────
# 3) 전체 데이터 배치 예측 (verbose=0 으로 메시지 숨김)
# ─────────────────────────────────────────────────────────────────────────────
seqs  = tokenizer.texts_to_sequences(df["clean"])
pads  = pad_sequences(seqs, maxlen=MAX_LEN, padding="post")
probs = model.predict(pads, verbose=0)
df["pred"] = np.argmax(probs, axis=1)
df["confidence"] = np.max(probs, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="리뷰 감성 분석 대시보드", layout="wide")
st.title("📊 리뷰 감성 분석 & 피드백 (원본 CSV 직접 수정)")

# 4.1) 리뷰 입력 & 예측
st.header("📩 리뷰 입력 & 예측")
user_review = st.text_area("분석할 리뷰를 입력하세요.", height=150)

if st.button("예측하기"):
    # 예측
    seq  = tokenizer.texts_to_sequences([clean_text(user_review)])
    pad  = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    prob = model.predict(pad, verbose=0)[0]
    pred = int(np.argmax(prob))
    conf = float(np.max(prob))
    st.write(f"🔍 예측 결과: **{LABEL_MAP[pred]}**  (신뢰도: {conf:.2f})")

    # 피드백 받기
    st.write("### 혹시 분류가 잘못되었나요?")
    correct = st.radio(
        "올바른 라벨을 선택하세요.",
        options=[0,1,2],
        format_func=lambda x: LABEL_MAP[x]
    )

    if st.button("✅ 피드백 저장 (원본 CSV 업데이트)"):
        # 1) 메모리상의 df에도 반영
        mask = df["review"] == user_review
        if mask.any():
            df.loc[mask, "label"] = correct
            df.loc[mask, "pred"]  = correct
            # 2) 바로 원본 CSV에 덮어쓰기
            df.to_csv(CSV_LABELED, index=False, encoding="utf-8-sig")
            st.success("원본 reviews_labeled.csv가 업데이트되었습니다!")
        else:
            st.error("해당 리뷰가 원본 데이터에 없습니다—업데이트 실패.")

# 4.2) 전체 정확도
accuracy = (df["pred"] == df["label"]).mean()
st.markdown(f"**전체 데이터 정확도:** {accuracy:.3f}")

# 4.3) 예측 라벨 분포
st.subheader("예측 라벨 분포")
dist = pd.Series(df["pred"]).map(LABEL_MAP).value_counts()
st.bar_chart(dist)

# 4.4) 감정별 워드클라우드
st.subheader("감정별 워드클라우드")
cols = st.columns(3)
for i, name in LABEL_MAP.items():
    texts = " ".join(df[df["pred"]==i]["clean"])
    if not texts:
        cols[i].write(f"{name} 리뷰가 없습니다.")
    else:
        wc = WordCloud(
            font_path=FONT_PATH,
            width=400, height=200,
            background_color="white"
        ).generate(texts)
        cols[i].image(wc.to_array(), caption=name, use_container_width=True)

# 4.5) 감정별 상위 20단어 빈도
st.subheader("감정별 상위 20단어 빈도")
for i, name in LABEL_MAP.items():
    tokens  = [tok for txt in df[df["pred"]==i]["clean"] for tok in txt.split()]
    counter = Counter(tokens)
    top20   = counter.most_common(20)
    if not top20:
        st.markdown(f"**{name} 리뷰에 단어가 충분하지 않습니다.**")
    else:
        words, counts = zip(*top20)
        st.markdown(f"**{name} Top 20 단어**")
        st.bar_chart(pd.Series(counts, index=words))
