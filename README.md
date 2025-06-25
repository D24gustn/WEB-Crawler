# 🎮 스팀 리뷰 감성 분석 파이프라인

스팀(Steam) 게임 리뷰를 자동으로 수집하고, LLM과 BiLSTM 기반 딥러닝 모델로 감성을 분류한 뒤, Streamlit 대시보드로 결과를 시각화하여 제공하는 올인원 파이프라인 프로젝트입니다.

---

## 🧠 전제 지식 및 요구 사항 (Prerequisites & Knowledge)

* **Python** 및 **pandas**, **NumPy**, **requests**, **BeautifulSoup** 기본 사용 경험
* **자연어 처리(NLP)** 개념: 텍스트 정제, 토큰화, 시퀀스 패딩 이해
* **LLM 프롬프트 설계** 및 **API 호출** 경험 (Gemma API 사용)
* **딥러닝 모델링**: RNN/BiLSTM, 드롭아웃, 체크포인트, EarlyStopping 등 이해
* **Streamlit**으로 간단한 웹 UI 제작 경험

---

## 📁 프로젝트 구조 (Project Structure)

```
.
├── data/
│   ├── steam_reviews_cache.csv   # RAW 리뷰 캐시
│   └── merged_reviews.csv        # 병합·중복 제거된 리뷰
├── build_reviews.py              # 1) Steam API & 크롤링 → steam_reviews_cache.csv
├── merge_reviews.py              # 2) CSV 병합 및 전처리 → merged_reviews.csv
├── qwer.py                       # 3) LLM(Gemma) 배치 레이블링 → reviews_labeled.csv
├── train.py                      # 4) 전처리 + BiLSTM 학습 → best_lstm.h5, tokenizer.pkl
└── app.py                        # 5) Streamlit 대시보드 실행
```

---

## 📋 사용 방법 (How to Use)

1. 가상환경 설정 & 활성화

   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   ```
2. 의존성 설치

   ```bash
   pip install -r requirements.txt
   ```
3. 단계별 실행

   ```bash
   # 1) 리뷰 수집
   python build_reviews.py --app-id <Steam App ID>

   # 2) CSV 병합
   python merge_reviews.py --input data/steam_reviews_cache.csv --output data/merged_reviews.csv

   # 3) LLM 레이블링
   python qwer.py --input data/merged_reviews.csv --output data/reviews_labeled.csv

   # 4) 모델 학습
   python train.py --input data/reviews_labeled.csv --save-dir models/

   # 5) 대시보드 실행
   streamlit run app.py
   ```

---

## 📊 데이터 커리큘럼 (Data Curriculum)

1. **Raw Data Collection**: `build_reviews.py` → Steam AppReviews API 호출 + HTML 크롤링 → `steam_reviews_cache.csv`
2. **Data Merge & Cleaning**: `merge_reviews.py` → 여러 CSV 병합, 중복 제거 → `merged_reviews.csv`
3. **LLM Labeling**: `qwer.py` → 배치 프롬프트(`make_batch_prompt`) 생성 → Gemma API 호출 → `reviews_labeled.csv`
4. **Preprocessing & Training**: `train.py` → `clean_text()` 정제, Tokenizer 학습 → BiLSTM 모델 학습 → `best_lstm.h5`, `tokenizer.pkl`
5. **Evaluation & Deployment**: `app.py` → Streamlit UI에서 실시간 예측 및 시각화

---

## ⚙️ 도전 과제 및 해결 (Challenges & Solutions)

* **한글 텍스트 노이즈**: 특수문자, 영문 혼합 문제 ⇒ `clean_text()` 함수로 정규표현식 기반 필터링 적용
* **레이블 불균형**: 중립 클래스 과다 ⇒ 클래스 가중치(class\_weight)와 오버/언더샘플링 기법 적용
* **과적합**: 높은 학습 정확도 vs 검증 정확도 하락 ⇒ 드롭아웃, EarlyStopping, ModelCheckpoint 사용
* **실시간 예측 지연**: 모델 로드 및 추론 속도 문제 ⇒ TensorFlow SavedModel 포맷 및 캐시 기법 도입

---

## ✨ 기능 요약 (Features Summary)

* **자동 리뷰 수집**: Steam API + HTML 크롤링
* **CSV 병합 & 전처리**: 다수 파일 통합·중복 제거 기능
* **LLM 기반 정밀 레이블링**: Gemma API 배치 호출
* **BiLSTM 딥러닝 학습**: 커스텀 토크나이저 및 체크포인트 관리
* **Streamlit 대시보드**: 실시간 감성 예측, 분포 차트, 워드클라우드 제공
* **사용자 피드백 & 재학습**: 잘못된 예측 피드백 수집 후 재학습 트리거

---

## 📈 성능 평가 (Performance Evaluation)

| 지표                | 값                   |
| ----------------- | ------------------- |
| 검증 정확도 (val\_acc) | 87.6%               |
| 검증 손실 (val\_loss) | 0.288               |
| 테스트 정확도           | 88.2%               |
| 정밀도 (Precision)   | 긍정: 0.89 / 부정: 0.85 |
| 재현율 (Recall)      | 긍정: 0.88 / 부정: 0.86 |
| F1-스코어 (F1-score) | 긍정: 0.88 / 부정: 0.85 |

> 성능은 데이터 분포 및 하이퍼파라미터에 따라 달라질 수 있습니다.

---

## 📄 라이선스 (License)

MIT License
