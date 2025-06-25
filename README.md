1️⃣ 리뷰 수집 (크롤링)
스크립트: build.reviews.py
설명: Steam AppReviews API + HTML 스크래핑으로 한글 리뷰 수집
출력: steam_reviews_cache.csv

2️⃣ CSV 병합 & 중복 제거
스크립트: merge_reviews.py
입력: 여러 개로 나뉜 *.csv
설명: 컬럼명 통일 → 중복(review_id 또는 review 본문) 제거
출력: merged_reviews.csv

3️⃣ LLM 감성 라벨링
스크립트: LM Studio.py
입력: merged_reviews.csv
설명: Google/Gemma-3-12b (LM Studio) API 호출로

0 = 부정
1 = 중립
2 = 긍정
자동 라벨링
출력: reviews_labeled.csv

4️⃣ 모델 학습
스크립트: train.py
입력: reviews_labeled.csv
설명:
텍스트 정제 → train/val 분할
Keras Tokenizer 학습 → 시퀀스 패딩
Embedding → BiLSTM → Dropout → Dense(softmax)
체크포인트 & EarlyStopping 적용
출력:
tokenizer.pkl (Keras Tokenizer)
best_lstm.h5 (학습된 LSTM+BiLSTM 모델)

5️⃣ Streamlit 대시보드
스크립트: streamlit.py
입력:
best_lstm.h5
tokenizer.pkl
(reviews_labeled.csv 교정판)
주요 기능:
리뷰 입력 → 실시간 감성 예측 (라벨 + 신뢰도)
전체 예측 분포 바 차트
감정별 워드클라우드 & Top-20 단어 빈도
잘못된 라벨 피드백 → 원본 CSV 자동 업데이트 → 모델 재학습 버튼










도구



ChatGPT는 실수를 할 수 있습니다


