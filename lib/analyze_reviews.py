from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_THEMES = [
    "product or service quality",
    "delay or turnaround time",
    "staff or support behavior",
    "incorrect order or fulfillment issue",
    "billing or pricing issue",
    "app or website issue",
    "communication issue",
    "unavailable product or service",
    "cleanliness or environment",
    "usability or workflow issue",
    "other",
]


ACTION_MAP = {
    "product or service quality": "Review quality control processes and consistency across the customer experience.",
    "delay or turnaround time": "Investigate response times, staffing coverage, and process bottlenecks.",
    "staff or support behavior": "Review customer-facing communication and coaching opportunities for staff or support teams.",
    "incorrect order or fulfillment issue": "Audit fulfillment steps and add a verification checkpoint before delivery or completion.",
    "billing or pricing issue": "Review billing accuracy, pricing clarity, and customer expectation setting.",
    "app or website issue": "Investigate platform reliability, broken flows, and user-reported friction points.",
    "communication issue": "Review how updates, expectations, and changes are communicated to customers.",
    "unavailable product or service": "Check inventory, service availability, and expectation-setting around availability.",
    "cleanliness or environment": "Review physical environment standards and maintenance routines.",
    "usability or workflow issue": "Investigate confusing steps, handoff gaps, or friction in the customer journey.",
    "other": "Review this feedback manually for a more specific next step.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze customer feedback into structured complaint insights.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--review-column", default="review_text", help="Column containing feedback text.")
    parser.add_argument("--rating-column", default="rating", help="Column containing rating if available.")
    parser.add_argument("--max-rating", type=float, default=None, help="Only analyze feedback with rating <= max-rating.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on rows to analyze.")
    parser.add_argument("--use-mock", action="store_true", help="Use heuristic classification instead of a real LLM.")
    return parser.parse_args()


def clean_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def validate_columns(df: pd.DataFrame, review_column: str) -> None:
    if review_column not in df.columns:
        raise ValueError(f"Missing review column '{review_column}'. Available columns: {list(df.columns)}")


def extract_first_sentence(text: str, max_words: int = 18) -> str:
    text = clean_text(text)
    if not text:
        return "No feedback text provided."

    sentence = re.split(r"[.!?]", text)[0].strip()
    words = sentence.split()
    return sentence if len(words) <= max_words else " ".join(words[:max_words]) + "..."


def normalize_theme_list(themes: Any) -> list[str]:
    if isinstance(themes, str):
        themes = [themes]

    if not isinstance(themes, list):
        return ["other"]

    cleaned = []
    for theme in themes:
        theme_str = str(theme).strip().lower()
        if theme_str in ALLOWED_THEMES and theme_str not in cleaned:
            cleaned.append(theme_str)

    return cleaned or ["other"]


def normalize_result(result: dict[str, Any], review_text: str) -> dict[str, Any]:
    themes = normalize_theme_list(result.get("themes", ["other"]))

    try:
        severity = int(result.get("severity", 3))
    except (TypeError, ValueError):
        severity = 3
    severity = max(1, min(5, severity))

    try:
        confidence = float(result.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    short_issue_summary = clean_text(result.get("short_issue_summary", "")) or extract_first_sentence(review_text)
    action_item = clean_text(result.get("action_item", "")) or ACTION_MAP.get(themes[0], ACTION_MAP["other"])

    return {
        "themes": themes,
        "severity": severity,
        "short_issue_summary": short_issue_summary,
        "action_item": action_item,
        "confidence": confidence,
    }


def build_classification_prompt(review_text: str, rating: Any = None, company: Any = None, business_unit: Any = None) -> str:
    context_parts = []
    if company is not None and str(company).strip():
        context_parts.append(f"Company: {company}")
    if business_unit is not None and str(business_unit).strip():
        context_parts.append(f"Business unit: {business_unit}")
    if rating is not None and str(rating).strip():
        context_parts.append(f"Rating: {rating}")

    context = "\n".join(context_parts) if context_parts else "No additional metadata provided."

    return f"""
You are analyzing customer feedback for operational and business insights.

Allowed complaint categories:
{", ".join(ALLOWED_THEMES)}

Classify the feedback into up to 2 complaint categories from the allowed list only.

Return valid JSON with exactly these keys:
- themes: list[str]
- severity: int from 1 to 5
- short_issue_summary: str
- action_item: str
- confidence: float from 0 to 1

Guidelines:
- Use only the allowed categories.
- If the feedback is mostly positive, mixed, or unclear, use ["other"].
- Keep action_item short, practical, and business-relevant.
- Do not include markdown or extra text.

Metadata:
{context}

Feedback:
\"\"\"{review_text}\"\"\"
""".strip()


def classify_feedback_mock(review_text: str, rating: Any = None) -> dict[str, Any]:
    text = review_text.lower()

    keyword_rules = {
        "product or service quality": ["bad quality", "poor quality", "broken", "defective", "disappointing", "didn't work"],
        "delay or turnaround time": ["slow", "late", "delay", "took forever", "waited", "long wait"],
        "staff or support behavior": ["rude", "unhelpful", "ignored", "attitude", "support was useless"],
        "incorrect order or fulfillment issue": ["wrong item", "wrong order", "missing item", "incorrect", "never arrived"],
        "billing or pricing issue": ["overcharged", "charged twice", "too expensive", "billing issue", "refund"],
        "app or website issue": ["app crashed", "website", "bug", "login", "checkout", "payment failed"],
        "communication issue": ["no update", "wasn't told", "unclear", "misleading", "poor communication"],
        "unavailable product or service": ["out of stock", "sold out", "unavailable", "not available"],
        "cleanliness or environment": ["dirty", "filthy", "messy", "unclean"],
        "usability or workflow issue": ["confusing", "hard to use", "too many steps", "workflow", "complicated"],
    }

    matched_themes = []
    for theme, keywords in keyword_rules.items():
        if any(keyword in text for keyword in keywords):
            matched_themes.append(theme)

    if not matched_themes:
        matched_themes = ["other"]

    matched_themes = matched_themes[:2]

    severity = 3
    if rating is not None and str(rating).strip():
        try:
            rating_value = float(rating)
            severity = max(1, min(5, int(round(6 - rating_value))))
        except (TypeError, ValueError):
            pass

    strong_negative_words = ["terrible", "awful", "worst", "horrible", "never again", "unacceptable"]
    if any(word in text for word in strong_negative_words):
        severity = min(5, severity + 1)

    confidence = 0.55 if matched_themes == ["other"] else min(0.9, 0.6 + 0.1 * len(matched_themes))

    return {
        "themes": matched_themes,
        "severity": severity,
        "short_issue_summary": extract_first_sentence(review_text),
        "action_item": ACTION_MAP.get(matched_themes[0], ACTION_MAP["other"]),
        "confidence": confidence,
    }


def classify_feedback_with_model(prompt: str) -> dict[str, Any]:
    raise NotImplementedError("Real LLM mode is not implemented yet. Use --use-mock for now.")


def analyze_reviews(
    input_path: str,
    output_path: str,
    review_column: str = "review_text",
    rating_column: str = "rating",
    max_rating: float | None = None,
    limit: int | None = None,
    use_mock: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    validate_columns(df, review_column)

    df = df.copy()
    df[review_column] = df[review_column].apply(clean_text)
    df = df[df[review_column] != ""]

    if max_rating is not None and rating_column in df.columns:
        numeric_ratings = pd.to_numeric(df[rating_column], errors="coerce")
        df = df[numeric_ratings <= max_rating]

    if limit is not None:
        df = df.head(limit)

    results = []
    total = len(df)
    print(f"Analyzing {total} feedback entries...")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        review_text = row.get(review_column, "")
        rating = row.get(rating_column) if rating_column in df.columns else None
        company = row.get("company")
        business_unit = row.get("business_unit")
        date = row.get("date")

        try:
            if use_mock:
                raw_result = classify_feedback_mock(review_text, rating)
            else:
                prompt = build_classification_prompt(
                    review_text=review_text,
                    rating=rating,
                    company=company,
                    business_unit=business_unit,
                )
                raw_result = classify_feedback_with_model(prompt)

            result = normalize_result(raw_result, review_text)

            results.append({
                "company": company,
                "business_unit": business_unit,
                "date": date,
                "rating": rating,
                "review_text": review_text,
                "themes": "; ".join(result["themes"]),
                "primary_theme": result["themes"][0],
                "severity": result["severity"],
                "short_issue_summary": result["short_issue_summary"],
                "action_item": result["action_item"],
                "confidence": result["confidence"],
                "analysis_status": "ok",
                "error": "",
            })

        except Exception as exc:
            results.append({
                "company": company,
                "business_unit": business_unit,
                "date": date,
                "rating": rating,
                "review_text": review_text,
                "themes": "other",
                "primary_theme": "other",
                "severity": "",
                "short_issue_summary": "",
                "action_item": "",
                "confidence": "",
                "analysis_status": "error",
                "error": str(exc),
            })

        if i % 10 == 0 or i == total:
            print(f"Processed {i}/{total}")

    output_df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Saved results to {output_path}")
    return output_df


def main() -> None:
    args = parse_args()
    analyze_reviews(
        input_path=args.input,
        output_path=args.output,
        review_column=args.review_column,
        rating_column=args.rating_column,
        max_rating=args.max_rating,
        limit=args.limit,
        use_mock=args.use_mock,
    )


if __name__ == "__main__":
    main()