from src.model import toxicity_classifier


def predict_comment(comment):
    results = toxicity_classifier(comment, top_k=None)

    category_scores = {}

    for result in results:
        category = result["label"]
        score = round(result["score"] * 100)

        category_scores[category] = score
    category_scores = dict(
        sorted(
        category_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
 )

    highest_category = max(
        category_scores,
        key=category_scores.get
    )

    risk_score = category_scores[highest_category]

    if risk_score >= 80:
        decision = "❌ Remove"
        risk_level = "High"
    elif risk_score >= 40:
        decision = "⚠️ Review"
        risk_level = "Medium"
    else:
        decision = "✅ Allow"
        risk_level = "Low"

    return {
        "decision": decision,
        "score": risk_score,
        "category": highest_category,
        "risk_level": risk_level,
        "category_scores": category_scores
    }