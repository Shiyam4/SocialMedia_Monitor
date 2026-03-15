import csv
import io
import os
import re
from collections import Counter
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, request, jsonify, make_response

import joblib

from preprocessing import clean_text

try:
    from langdetect import detect as detect_language
except ImportError:
    detect_language = None

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

app = Flask(__name__)

# Load trained model
model_path = os.path.join("model", "misinformation_model.pkl")
EVENTS_PATH = os.path.join("analytics", "events.csv")
EVENT_FIELDS = [
    "timestamp_utc",
    "result",
    "category",
    "source_language",
    "detected_language",
    "confidence",
    "template",
    "translated",
]
model = joblib.load(model_path)
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "ur": "Urdu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
}

CATEGORY_KEYWORDS = {
    "Health": [
        "vaccine", "covid", "coronavirus", "virus", "disease", "illness", "cancer", "diabetes",
        "symptom", "symptoms", "treatment", "cure", "medicine", "drug", "pill", "doctor", "hospital",
        "infection", "outbreak", "mask", "immune", "immunity", "health", "medical", "blood"
    ],
    "Technology": [
        "ai", "artificial intelligence", "machine learning", "robot", "robotics", "algorithm",
        "software", "hardware", "device", "smartphone", "phone", "laptop", "computer", "internet",
        "app", "application", "platform", "chip", "processor", "cyber", "hacker", "hack", "data",
        "privacy", "security", "blockchain", "crypto", "bitcoin", "cloud"
    ],
    "Scam": [
        "scam", "fraud", "phishing", "ponzi", "pyramid", "lottery", "jackpot", "giveaway",
        "free money", "investment", "guaranteed", "risk-free", "urgent", "limited time",
        "bank account", "credit card", "ssn", "password", "otp", "wire transfer", "gift card"
    ],
    "Politics": [
        "election", "vote", "voting", "ballot", "campaign", "president", "prime minister",
        "government", "senate", "congress", "parliament", "policy", "law", "bill", "minister",
        "party", "democrat", "republican", "bjp", "bsp", "aap", "mp", "mla", "governor"
    ],
    "Rumor": [
        "rumor", "rumour", "unverified", "alleged", "allegedly", "hearsay", "supposed",
        "leak", "leaked", "secret", "insider", "breaking", "viral", "trending", "claimed",
        "claim", "reportedly"
    ],
}

def categorize_text(text: str) -> str:
    if not text:
        return "Rumor"
    text_lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text_lower:
                score += 1
        if score > 0:
            scores[category] = score
    if not scores:
        return "Rumor"
    return max(scores, key=scores.get)


def make_template(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"http\S+|www\.\S+", " ", normalized)
    normalized = re.sub(r"\b(?:rs|inr|usd|eur|\$)\s?\d+[\d,]*\b", " <amount> ", normalized)
    normalized = re.sub(r"\b\d+[\d,]*\b", " <num> ", normalized)
    normalized = re.sub(r"[^a-z<>\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    words = normalized.split()
    return " ".join(words[:12])


def ensure_event_store() -> None:
    os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)
    if not os.path.exists(EVENTS_PATH):
        with open(EVENTS_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
            writer.writeheader()


def log_event(row: dict) -> None:
    ensure_event_store()
    with open(EVENTS_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        writer.writerow(row)


def load_events() -> list:
    if not os.path.exists(EVENTS_PATH):
        return []
    with open(EVENTS_PATH, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_ts(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _apply_date_filter(rows, start=None, end=None):
    if not (start or end):
        return rows
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None
    # if user provided a date without a time, treat end as inclusive by advancing one day
    if end_dt is not None and end_dt.time() == datetime.min.time():
        end_dt = end_dt + timedelta(days=1)
    # convert everything to naive UTC for comparison
    if start_dt is not None and start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt is not None and end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    filtered = []
    for r in rows:
        ts = _parse_ts(r.get("timestamp_utc", ""))
        if ts is None:
            continue
        # normalize ts to UTC naive
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc)
        ts_naive = ts.replace(tzinfo=None)
        if start_dt is not None:
            sd_naive = start_dt.astimezone(timezone.utc).replace(tzinfo=None)
            if ts_naive < sd_naive:
                continue
        if end_dt is not None:
            ed_naive = end_dt.astimezone(timezone.utc).replace(tzinfo=None)
            if ts_naive >= ed_naive:
                continue
        filtered.append(r)
    return filtered

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["post"]

    # 1. Detect language
    try:
        lang = detect_language(text) if detect_language else "unknown"
    except Exception:
        lang = "unknown"

    # 2. Translate to English if needed
    try:
        if lang != "en" and GoogleTranslator is not None:
            translated = GoogleTranslator(source="auto", target="en").translate(text)
        else:
            translated = text
    except Exception:
        translated = text

    # 3. Clean translated text
    cleaned = clean_text(translated)

    # 4. Make prediction
    prediction = model.predict([cleaned])[0]
    confidence = model.predict_proba([cleaned])[0].max() * 100

    result = "Misinformation" if prediction == 1 else "Not Misinformation"
    category = categorize_text(translated)
    source_language = LANGUAGE_NAMES.get(lang, lang.upper() if lang != "unknown" else "Unknown")

    log_event(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "category": category,
            "source_language": source_language,
            "detected_language": lang,
            "confidence": f"{confidence:.2f}",
            "template": make_template(translated),
            "translated": translated[:500],
        }
    )

    return render_template(
        "result.html",
        result=result,
        confidence=round(confidence, 2),
        category=category,
        original=text,
        translated=translated,
        detected_language=lang,
        source_language=source_language
    )


@app.route("/dashboard")
def dashboard():

    # allow optional ?start=YYYY-MM-DD&end=YYYY-MM-DD filters
    start = request.args.get("start")
    end = request.args.get("end")

    rows = load_events()
    rows = _apply_date_filter(rows, start, end)
    total = len(rows)

    if not rows:
        return render_template(
            "dashboard.html",
            total=0,
            misinfo_total=0,
            misinfo_rate=0,
            topic_rows=[],
            language_rows=[],
            template_rows=[],
            start=start,
            end=end,
        )

    misinfo_rows = [r for r in rows if r.get("result") == "Misinformation"]
    misinfo_total = len(misinfo_rows)
    misinfo_rate = round((misinfo_total / total) * 100, 2) if total else 0

    topic_counts = Counter(r.get("category", "Unknown") for r in misinfo_rows)
    topic_rows = [{"name": k, "count": v} for k, v in topic_counts.most_common(8)]

    language_counts = Counter(r.get("source_language", "Unknown") for r in rows)
    language_rows = [{"name": k, "count": v} for k, v in language_counts.most_common(10)]

    scam_rows = [r for r in misinfo_rows if r.get("category") == "Scam" and r.get("template")]
    if not scam_rows:
        scam_rows = [r for r in misinfo_rows if r.get("template")]

    template_counts = Counter(r.get("template", "") for r in scam_rows)
    template_rows = [{"name": k, "count": v} for k, v in template_counts.most_common(8) if k]

    return render_template(
        "dashboard.html",
        total=total,
        misinfo_total=misinfo_total,
        misinfo_rate=misinfo_rate,
        topic_rows=topic_rows,
        language_rows=language_rows,
        template_rows=template_rows,
        start=start,
        end=end,
    )

@app.route("/export")
def export():
    fmt = request.args.get("format", "csv").lower()
    start = request.args.get("start")
    end = request.args.get("end")
    rows = load_events()
    rows = _apply_date_filter(rows, start, end)
    if fmt == "json":
        return jsonify(rows)
    # default to csv
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=EVENT_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    resp = make_response(output.getvalue())
    resp.headers["Content-Disposition"] = "attachment; filename=events.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


if __name__ == "__main__":
    app.run(debug=True)
    
