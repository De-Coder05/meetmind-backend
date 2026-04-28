import requests
import sys
from collections import defaultdict

API_URL = "http://localhost:8000/api/v1/tasks/feedback/export"

def evaluate_classifier():
    print("Fetching evaluation data from MeetMind API...")
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching feedback data: {e}")
        print("Make sure the MeetMind backend is running at http://localhost:8000")
        sys.exit(1)

    records = data.get("records", [])
    total = data.get("total", 0)

    if not records:
        print("No feedback records found. Submit corrections via the UI to generate evaluation data.")
        sys.exit(0)

    print(f"\n--- Evaluated on {total} samples ---\n")

    # Metrics
    # TP: predicted == actual == class
    # FP: predicted == class AND actual != class
    # FN: predicted != class AND actual == class
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    
    confused_pairs = defaultdict(int)
    
    classes = {"decision", "commitment", "discussion", "open_question"}

    for r in records:
        predicted = r["original_type"]  # What the classifier said
        actual = r["corrected_type"]    # What the human corrected it to
        
        classes.add(predicted)
        classes.add(actual)

        if predicted == actual:
            tp[predicted] += 1
        else:
            fp[predicted] += 1
            fn[actual] += 1
            # Track confusion: (Predicted -> Actual)
            pair_key = f"Predicted '{predicted}' but was actually '{actual}'"
            confused_pairs[pair_key] += 1

    # Print Precision and Recall per class
    print(f"{'CATEGORY':<20} | {'PRECISION':<12} | {'RECALL':<12} | {'F1 SCORE':<12}")
    print("-" * 65)

    overall_tp = 0
    overall_samples = len(records)

    for c in sorted(classes):
        _tp = tp[c]
        _fp = fp[c]
        _fn = fn[c]
        
        overall_tp += _tp
        
        precision = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        recall = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        p_str = f"{precision:.2f}" if (_tp + _fp) > 0 else "-"
        r_str = f"{recall:.2f}" if (_tp + _fn) > 0 else "-"
        f_str = f"{f1:.2f}" 

        print(f"{c.upper():<20} | {p_str:<12} | {r_str:<12} | {f_str:<12}")

    print("\n")
    accuracy = overall_tp / overall_samples if overall_samples > 0 else 0.0
    print(f"Overall Accuracy: {accuracy:.2%} ({overall_tp}/{overall_samples})\n")

    # Print most confused pairs
    if confused_pairs:
        print("Top Classification Confusions:")
        print("-" * 35)
        # Sort by count descending
        sorted_pairs = sorted(confused_pairs.items(), key=lambda x: -x[1])
        for pair, count in sorted_pairs[:5]:
            print(f"- {pair} ({count} instances)")
        print("\nRecommendation: Add more few-shot examples targeting the most confused pairs.")
    else:
        print("Perfect classification so far! No confused pairs detected.")


if __name__ == "__main__":
    evaluate_classifier()
