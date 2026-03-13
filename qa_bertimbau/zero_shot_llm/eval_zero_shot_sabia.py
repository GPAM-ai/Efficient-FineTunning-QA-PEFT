import json
import os
import evaluate
from datasets import load_dataset

PRED_FILE = "results/sabia_predictions.json"
OUTPUT_FILE = "results/sabia_zero_shot_metrics.json"

def load_dataset_for_eval():
    dataset = load_dataset(
        "json",
        data_files="../data/flat_squad-dev-v1.1.json",
        split="train"
    )
    return dataset

def main():
    metric = evaluate.load("squad")

    with open(PRED_FILE, encoding="utf-8") as f:
        predictions_raw = json.load(f)

    dataset = load_dataset_for_eval()

    predictions = []
    references = []

    for example in dataset:
        for qa in example["data"]:
            qid = qa["id"]
            ans = qa.get("answers", {})

            if not ans or not ans.get("text"):
                continue

            predictions.append({
                "id": qid,
                "prediction_text": predictions_raw.get(qid, "")
            })

            references.append({
                "id": qid,
                "answers": ans
            })

    if not predictions:
        print(" No valid predictions found.")
        return

    print(f"Evaluating {len(predictions)} samples.")
    results = metric.compute(
        predictions=predictions,
        references=references
    )

    print("\n RESULTADOS SABIÁ ZERO-SHOT")
    print(f"F1: {results['f1']:.2f}")
    print(f"EM: {results['exact_match']:.2f}")
    
    # Guardar resultados
    os.makedirs("results", exist_ok=True)
    output = {
        "model": "maritaca-ai/sabia-7b",
        "config": "zero-shot",
        "f1": results["f1"],
        "exact_match": results["exact_match"],
        "num_samples": len(predictions)
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Resultados guardados en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
