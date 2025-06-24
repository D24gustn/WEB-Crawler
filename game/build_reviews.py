# build_reviews_korean.py
import os
import time
import requests
from typing import List, Dict, Optional
import pandas as pd

# ——————————————————————————————————————
# 설정
# ——————————————————————————————————————
PRELOAD_GAMES = [
    "PlayerUnknown's Battlegrounds",
    "Counter-Strike: Global Offensive",
    "Monster Hunter: World",
    "Cyberpunk 2077",
    "Elden Ring",
    "Valheim",
    "Grand Theft Auto V",
    "Rust",
    "The Witcher 3: Wild Hunt",
    "Red Dead Redemption 2",
    "Hades",
    "Stardew Valley",
    "Dark Souls III",
    "Sekiro: Shadows Die Twice",
    "DOOM Eternal",
    "Phasmophobia",
    "Baldur's Gate 3",
    "Rainbow Six Siege",
    "ARK: Survival Evolved",
    "Fall Guys",
    "Team Fortress 2",
    "Resident Evil 2",
    "Fallout 4"
]

TOTAL_REVIEWS_TARGET = 50000  # 전체 목표 리뷰 개수

CSV_DIR  = r"D:\_DeepNLP25\site\cache"
os.makedirs(CSV_DIR, exist_ok=True)
CSV_PATH = os.path.join(CSV_DIR, "steam_reviews_cache.csv")

# ——————————————————————————————————————
# 1) Steam 검색 → appID 추출 함수
#    (&l=korean&cc=KR 추가, selector 확장)
# ——————————————————————————————————————
def get_app_id(game_name: str) -> Optional[int]:
    search_url = (
        f"https://store.steampowered.com/search/"
        f"?term={requests.utils.quote(game_name)}"
        f"&l=korean&cc=KR"
    )
    try:
        resp = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    except:
        return None

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")

    a_tag = (
        soup.select_one("div#search_resultsRows a")
        or soup.select_one("a.search_result_row")
    )
    if not a_tag or "href" not in a_tag.attrs:
        return None

    href = a_tag["href"]
    parts = href.split("/")
    for idx, p in enumerate(parts):
        if p == "app" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None

# ——————————————————————————————————————
# 2) Steam AppReviews API 호출 (한국어 리뷰만, 중복 제거)
#    (day_range를 365로 늘려 과거 1년치 리뷰 수집)
# ——————————————————————————————————————
def fetch_korean_reviews(appid: int, max_reviews: int = 200, max_pages: int = 50) -> List[Dict]:
    reviews: List[Dict] = []
    seen_ids = set()
    cursor = "*"
    batch = 100
    page_count = 0
    empty_streak = 0

    while len(reviews) < max_reviews and page_count < max_pages:
        page_count += 1
        try:
            r = requests.get(
                f"https://store.steampowered.com/appreviews/{appid}",
                params={
                    "json": 1,
                    "filter": "all",
                    "language": "korean",
                    "day_range": 365,       # 과거 1년치
                    "num_per_page": batch,
                    "cursor": cursor
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            data = r.json()
        except:
            break

        revs = data.get("reviews", [])
        if not revs:
            empty_streak += 1
            if empty_streak >= 2:
                break
            continue
        empty_streak = 0

        for rev in revs:
            review_id = rev.get("recommendationid") or rev.get("reviewid")
            if not review_id or review_id in seen_ids:
                continue

            text = rev.get("review", "").strip()
            if not text:
                continue

            voted_up = rev.get("voted_up", False)
            seen_ids.add(review_id)
            reviews.append({
                "game":      None,       # 나중에 채워넣기
                "review_id": review_id,
                "review":    text,
                "voted_up":  voted_up
            })
            if len(reviews) >= max_reviews:
                break

        cursor = data.get("cursor", "")
        if not cursor:
            break

        time.sleep(0.2)

    return reviews

# ——————————————————————————————————————
# 3) 전체 게임 순차 리뷰 수집 & CSV 저장
# ——————————————————————————————————————
if __name__ == "__main__":
    all_rows = []
    total_loaded = 0
    remaining_games = len(PRELOAD_GAMES)

    for game in PRELOAD_GAMES:
        remaining_games -= 1
        remain_target = TOTAL_REVIEWS_TARGET - total_loaded
        if remain_target <= 0:
            break

        per_game = -(-remain_target // (remaining_games + 1))
        appid = get_app_id(game)
        if appid is None:
            print(f"⚠️ Could not find appID for \"{game}\"")
            continue

        raw_revs = fetch_korean_reviews(appid, max_reviews=per_game)
        for item in raw_revs:
            item["game"] = game
        all_rows.extend(raw_revs)
        total_loaded += len(raw_revs)
        print(f"{game}: {len(raw_revs)} reviews fetched, total so far: {total_loaded}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        # voted_up → “긍정/부정” 라벨 매핑 (중립 없음)
        df["label"] = df["voted_up"].map({True: "긍정", False: "부정"})
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"✅ CSV saved to {CSV_PATH} (총 {len(df)}행)")
    else:
        print("⚠️ 수집된 리뷰가 없어 CSV가 생성되지 않았습니다.")
