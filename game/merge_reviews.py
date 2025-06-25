import os
import pandas as pd

# ——————————————————————————————————————
# 설정: CSV들이 들어있는 폴더와 저장할 경로
# ——————————————————————————————————————
csv_dir    = r"D:\_DeepNLP25\site\reviews"   # <-- 여기를 본인 폴더로 변경
output_csv = r"D:\_DeepNLP25\site\cache\merged_reviews.csv"

# ——————————————————————————————————————
# 1) 폴더 내 모든 CSV 파일 목록 생성
# ——————————————————————————————————————
file_paths = [
    os.path.join(csv_dir, fname)
    for fname in os.listdir(csv_dir)
    if fname.lower().endswith(".csv")
]

# ——————————————————————————————————————
# 2) 각 CSV 읽어서 컬럼명 매핑 후 리스트에 쌓기
# ——————————————————————————————————————
dfs = []
for path in file_paths:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    # 중복 컬럼 제거
    df = df.loc[:, ~df.columns.duplicated()]

    col_map = {}
    for col in df.columns:
        lc = col.lower()
        # 리뷰 본문
        if any(k in lc for k in ["review", "리뷰", "comment", "content", "text"]):
            col_map[col] = "review"
        # 리뷰 ID
        elif ("id" in lc and ("review" in lc or "리뷰" in lc)) or lc == "id":
            col_map[col] = "review_id"
        # 감성/라벨
        elif "label" in lc or "sentiment" in lc:
            col_map[col] = "label"
        # 게임명/사이트 등
        elif "game" in lc or "site" in lc:
            col_map[col] = "game"

    df = df.rename(columns=col_map)
    # 매핑 후에도 중복 컬럼 제거
    df = df.loc[:, ~df.columns.duplicated()]
    dfs.append(df)

# ——————————————————————————————————————
# 3) 병합 및 중복 제거
# ——————————————————————————————————————
merged = pd.concat(dfs, ignore_index=True)

# review_id가 있으면 ID 기준, 없으면 본문(review) 기준으로 중복 제거
if "review_id" in merged.columns:
    merged = merged.drop_duplicates(subset=["review_id"])
else:
    merged = merged.drop_duplicates(subset=["review"])

# ——————————————————————————————————————
# 4) 결과 저장
# ——————————————————————————————————————
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"✅ Merged CSV saved to: {output_csv}")
