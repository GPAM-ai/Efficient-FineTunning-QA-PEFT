import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate

# -----------------------------
# Configuración
# -----------------------------
BASE_MODEL = "maritaca-ai/sabia-7b"
LORA_PATH = "./lora_sabia_qa"
DATA_FILE = "../data/flat_squad-dev-v1.1.json"

MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 64

# -----------------------------
# Métrica SQuAD
# -----------------------------
metric = evaluate.load("squad")

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Modelo base + LoRA (CORRECTO)
# -----------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# Detectamos device real del modelo
DEVICE = next(model.parameters()).device
print(" Modelo en:", DEVICE)

# -----------------------------
# Cargar dataset plano
# -----------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

examples = []
for item in raw["data"]:
    answers = item.get("answers", {})
    if not answers or not answers.get("text"):
        continue

    examples.append({
        "id": item["id"],
        "context": item["context"],
        "question": item["question"],
        "answers": {
            "text": [answers["text"][0]],
            "answer_start": [answers["answer_start"][0]]
        }
    })

print(f"Ejemplos cargados: {len(examples)}")

# -----------------------------
# Inferencia
# -----------------------------
predictions = []
references = []

for ex in examples:
    prompt = (
        f"Contexto: {ex['context']}\n"
        f"Pergunta: {ex['question']}\n"
        f"Resposta:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    # mover inputs al mismo device del modelo
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    predictions.append({
        "id": ex["id"],
        "prediction_text": answer
    })

    references.append({
        "id": ex["id"],
        "answers": ex["answers"]
    })

# -----------------------------
# Guardar predicciones
# -----------------------------
os.makedirs("results", exist_ok=True)

pred_dict = {p["id"]: p["prediction_text"] for p in predictions}
with open("results/sabia_lora_predictions.json", "w", encoding="utf-8") as f:
    json.dump(pred_dict, f, ensure_ascii=False, indent=2)

print(f"Predicciones guardadas: {len(pred_dict)}")

# -----------------------------
# Métricas
# -----------------------------
results = metric.compute(
    predictions=predictions,
    references=references
)

print("\n RESULTADOS SABIÁ + LoRA")
print(f"F1: {results['f1']:.2f}")
print(f"EM: {results['exact_match']:.2f}")

# Guardar métricas
metrics_output = {
    "model": "maritaca-ai/sabia-7b",
    "config": "LoRA",
    "f1": results["f1"],
    "exact_match": results["exact_match"],
    "num_samples": len(predictions)
}
with open("results/sabia_lora_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_output, f, ensure_ascii=False, indent=2)

print(f"Métricas guardadas en: results/sabia_lora_metrics.json")
