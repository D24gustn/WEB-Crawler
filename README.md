
![image](https://github.com/user-attachments/assets/cec3660d-cf66-4009-b169-cb090f4e153b)

1️⃣ CSV 병합 & 중복 제거
스크립트: merge_reviews.py
입력: 여러 개로 나뉜 *.csv
설명: 컬럼명 통일 → 중복(review_id 또는 review 본문) 제거
출력: merged_reviews.csv

2️⃣ LLM 감성 라벨링
스크립트: LM Studio.py
입력: merged_reviews.csv
설명: Google/Gemma-3-12b (LM Studio) API 호출로

0 = 부정
1 = 중립
2 = 긍정
출력: reviews_labeled.csv

3️⃣ 모델 학습
스크립트: train.py
입력: reviews_labeled.csv
내용:
리뷰 문장에서 불필요한 기호 제거
단어를 숫자로 변환
LSTM 모델 학습
생성 파일:
tokenizer.pkl (단어↔번호 매핑)
best_lstm.h5 (학습된 모델)

4️⃣ Streamlit 대시보드
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


![image](https://github.com/user-attachments/assets/28e3c614-3076-44d0-894b-509e575c539b)


