import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import evaluate  # ← nuevo paquete

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = "./lora_tucano_qa"
DATA_FILE = "../data/flat_squad-dev-v1.1.json"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Cargar tokenizer y modelo LoRA
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")
model.to(DEVICE)
model.eval()

# -----------------------------
# Cargar dataset
# -----------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

examples = []
for item in raw["data"]:
    context = item["context"]
    question = item["question"]
    answers = item.get("answers", {})
    if not answers or not answers.get("text"):
        continue
    examples.append({
        "id": item["id"],
        "context": context,
        "question": question,
        "answers": answers
    })

dataset = Dataset.from_list(examples)

# -----------------------------
# Preparar métricas
# -----------------------------
metric = evaluate.load("squad")  

# -----------------------------
# Generar predicciones
# -----------------------------
predictions = []
references = []

for example in dataset:
    input_text = f"Contexto: {example['context']}\nPergunta: {example['question']}\nResposta:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=MAX_OUTPUT_LENGTH)
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    predictions.append({"id": example["id"], "prediction_text": answer})
    references.append({"id": example["id"], "answers": example["answers"]})

# -----------------------------
# Guardar predicciones
# -----------------------------
os.makedirs("results", exist_ok=True)

pred_dict = {p["id"]: p["prediction_text"] for p in predictions}
with open("results/tucano_lora_predictions.json", "w", encoding="utf-8") as f:
    json.dump(pred_dict, f, ensure_ascii=False, indent=2)

print(f"Predicciones guardadas: {len(pred_dict)}")

# -----------------------------
# Calcular métricas F1 / EM
# -----------------------------
results = metric.compute(predictions=predictions, references=references)

print("\n RESULTADOS TUCANO + LoRA")
print(f"F1: {results['f1']:.2f}")
print(f"EM: {results['exact_match']:.2f}")

# Guardar métricas
metrics_output = {
    "model": "TucanoBR/tucano-1b1-instruct",
    "config": "LoRA",
    "f1": results["f1"],
    "exact_match": results["exact_match"],
    "num_samples": len(predictions)
}
with open("results/tucano_lora_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_output, f, ensure_ascii=False, indent=2)

print(f"Métricas guardadas en: results/tucano_lora_metrics.json")
