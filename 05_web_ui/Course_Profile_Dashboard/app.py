from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
import time
from urllib.request import Request, urlopen

app = Flask(__name__)

DATA_PATH = Path(__file__).parent / "course_profiles.csv"
REVIEW_CACHE_PATH = Path(__file__).parent / "review_cache.json"


def load_data():
    df = pd.read_csv(DATA_PATH)

    for col in ["ratings", "review_count", "course_students_enrolled", "skill_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


df = load_data()


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def clean_records(dataframe):
    records = dataframe.to_dict(orient="records")
    return [
        {key: clean_value(value) for key, value in record.items()}
        for record in records
    ]


def load_review_cache():
    if not REVIEW_CACHE_PATH.exists():
        return {}
    try:
        with open(REVIEW_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


review_cache = load_review_cache()


def save_review_cache():
    with open(REVIEW_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(review_cache, f, indent=2)


def extract_review_count(html):
    """
    Coursera pages can expose review counts in several JSON/HTML forms.
    The extractor deliberately checks reviewCount/review_count before
    looser 'reviews' text so a value such as 3246 remains 3246.
    """
    patterns = [
        r'"reviewCount"\s*:\s*"?(?P<n>\d[\d,]*)',
        r'"review_count"\s*:\s*"?(?P<n>\d[\d,]*)',
        r'"numReviews"\s*:\s*"?(?P<n>\d[\d,]*)',
        r'"reviewsCount"\s*:\s*"?(?P<n>\d[\d,]*)',
        r'"ratingCount"\s*:\s*"?(?P<n>\d[\d,]*)',
        r'(?P<n>\d[\d,]*)\s+(?:student\s+)?reviews?',
        r'(?P<n>\d[\d,]*)\s+ratings?'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, html, flags=re.I)
        if matches:
            # Keep the complete integer; never slice/truncate it.
            values = []
            for item in matches:
                try:
                    values.append(int(str(item).replace(",", "")))
                except ValueError:
                    pass
            if values:
                return max(values)

    return None


def fetch_review_count(url):
    if not url or not str(url).startswith("http"):
        return None

    try:
        req = Request(
            str(url),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/140.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urlopen(req, timeout=12) as response:
            html = response.read().decode("utf-8", errors="ignore")
        return extract_review_count(html)
    except Exception:
        return None


def get_review_count(row):
    value = row.get("review_count")
    if pd.notna(value):
        return int(round(float(value)))

    course_id = str(row.get("course_id", ""))
    if course_id in review_cache:
        return review_cache[course_id]

    return None


def apply_filters(data):
    search = request.args.get("search", "").strip().lower()
    difficulty = request.args.get("difficulty", "").strip()
    course_type = request.args.get("type", "").strip()
    organization = request.args.get("organization", "").strip()
    duration = request.args.get("duration", "").strip()
    min_rating = request.args.get("min_rating", "").strip()

    if search:
        searchable = (
            data["title"].fillna("").astype(str) + " "
            + data["organization"].fillna("").astype(str) + " "
            + data["skills"].fillna("").astype(str)
        ).str.lower()
        data = data[searchable.str.contains(search, regex=False)]

    if difficulty:
        data = data[data["difficulty"].fillna("") == difficulty]

    if course_type:
        data = data[data["type"].fillna("") == course_type]

    if organization:
        data = data[data["organization"].fillna("") == organization]

    if duration:
        data = data[data["duration"].fillna("") == duration]

    if min_rating:
        try:
            data = data[data["ratings"] >= float(min_rating)]
        except ValueError:
            pass

    return data


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/summary")
def summary():
    data = apply_filters(df.copy())

    review_values = []
    for _, row in data.iterrows():
        count = get_review_count(row)
        if count is not None:
            review_values.append(count)

    summary_data = {
        "total_courses": int(len(data)),
        "organizations": int(data["organization"].nunique(dropna=True)),
        "avg_rating": round(float(data["ratings"].mean()), 2)
        if data["ratings"].notna().any() else None,
        "avg_skills": round(float(data["skill_count"].mean()), 1)
        if data["skill_count"].notna().any() else None,
        "total_reviews": int(sum(review_values)) if review_values else None,
        "courses_with_reviews": int(len(review_values)),
        "difficulty": data["difficulty"].fillna("Unknown").value_counts().to_dict(),
        "type": data["type"].fillna("Unknown").value_counts().to_dict(),
        "duration": data["duration"].fillna("Unknown").value_counts().to_dict(),
        "organizations_top": (
            data["organization"]
            .fillna("Unknown")
            .value_counts()
            .head(10)
            .to_dict()
        ),
        "ratings_distribution": (
            data["ratings"]
            .dropna()
            .round(1)
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }

    return jsonify(summary_data)


@app.route("/api/courses")
def courses():
    data = apply_filters(df.copy())

    columns = [
        "course_id", "title", "organization", "ratings",
        "review_count", "difficulty", "type", "duration",
        "skill_count", "course_url"
    ]
    columns = [c for c in columns if c in data.columns]

    data = data.sort_values(
        by=["ratings", "review_count"],
        ascending=[False, False],
        na_position="last"
    )

    records = clean_records(data[columns].head(100))

    # Add cached/live review values without changing the original CSV.
    for record in records:
        record["review_count_display"] = get_review_count(record)

    return jsonify({
        "count": int(len(data)),
        "courses": records
    })


@app.route("/api/course/<course_id>")
def course_detail(course_id):
    match = df[df["course_id"].astype(str) == str(course_id)]

    if match.empty:
        return jsonify({"error": "Course not found"}), 404

    record = clean_records(match.iloc[[0]])[0]
    record["review_count_display"] = get_review_count(record)
    return jsonify(record)


@app.route("/api/filters")
def filters():
    return jsonify({
        "difficulty": sorted([str(x) for x in df["difficulty"].dropna().unique()]),
        "type": sorted([str(x) for x in df["type"].dropna().unique()]),
        "organization": sorted([str(x) for x in df["organization"].dropna().unique()]),
        "duration": sorted([str(x) for x in df["duration"].dropna().unique()])
    })


@app.route("/api/reviews/refresh", methods=["POST"])
def refresh_reviews():
    """
    Fetch missing review counts from the course URLs.
    Existing CSV values are preserved. New values are cached locally.
    The dashboard can refresh only selected course IDs or all missing URLs.
    """
    payload = request.get_json(silent=True) or {}
    requested_ids = payload.get("course_ids")

    work = df.copy()
    if requested_ids:
        requested_ids = {str(x) for x in requested_ids}
        work = work[work["course_id"].astype(str).isin(requested_ids)]
    else:
        work = work[work["review_count"].isna()]

    updated = 0
    failed = 0

    for _, row in work.iterrows():
        course_id = str(row["course_id"])
        if pd.notna(row.get("review_count")):
            review_cache[course_id] = int(round(float(row["review_count"])))
            continue

        if course_id in review_cache:
            continue

        value = fetch_review_count(row.get("course_url"))
        if value is not None:
            review_cache[course_id] = int(value)
            updated += 1
        else:
            failed += 1

        # Small pause to avoid sending requests too quickly when many
        # URLs are refreshed.
        time.sleep(0.15)

    save_review_cache()

    return jsonify({
        "updated": updated,
        "failed": failed,
        "cached": len(review_cache)
    })


if __name__ == "__main__":
    app.run(debug=True)
