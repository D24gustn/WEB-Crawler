# train.py

import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text   import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models              import Sequential
from tensorflow.keras.layers              import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks           import ModelCheckpoint, EarlyStopping

# 1) 경로 및 하이퍼파라미터 설정
CSV_INPUT     = r"D:\_DeepNLP25\site\cache\reviews_labeled.csv"
TOKENIZER_PKL = "tokenizer.pkl"
MODEL_H5      = "best_lstm.h5"
MAX_WORDS     = 20000
MAX_LEN       = 100
TEST_SIZE     = 0.2
RANDOM_SEED   = 42
EPOCHS        = 10
BATCH_SIZE    = 64

# 2) 데이터 불러오기 & 라벨 정수화
df = pd.read_csv(CSV_INPUT, encoding="utf-8-sig", low_memory=False)
mapping = {"부정":0, "중립":1, "긍정":2}
if df['label'].dtype == object:
    df['label'] = df['label'].map(mapping).fillna(df['label'])

# 안전하게 float, 문자열 혼합을 처리
df = df.dropna(subset=['label']).reset_index(drop=True)
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label']).reset_index(drop=True)
df['label'] = df['label'].astype(int)

# 3) 텍스트 정제
def clean_text(s: str) -> str:
    s = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z0-9 ]", "", str(s))
    return s.lower().strip()

df['clean'] = df['review'].apply(clean_text)

# 4) 훈련/검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(
    df['clean'], df['label'],
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=df['label']
)

# 5) 토크나이저 학습 & 저장
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
with open(TOKENIZER_PKL, "wb") as f:
    pickle.dump(tokenizer, f)

# 6) 시퀀스 변환 & 패딩
X_train_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    maxlen=MAX_LEN, padding="post"
)
X_val_seq   = pad_sequences(
    tokenizer.texts_to_sequences(X_val),
    maxlen=MAX_LEN, padding="post"
)

# 7) LSTM 모델 구성
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(3, activation="softmax")
])
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# 8) 학습 및 체크포인트
ckpt = ModelCheckpoint(MODEL_H5, monitor="val_loss", save_best_only=True)
es   = EarlyStopping(patience=3, monitor="val_loss", restore_best_weights=True)

history = model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[ckpt, es]
)

# 9) 최종 안전장치로 모델 저장
model.save(MODEL_H5)
print(f"✅ Model trained and saved to {MODEL_H5}")
