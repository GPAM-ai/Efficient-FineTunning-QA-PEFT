import json
import os
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import extractive_prompt

MODEL_NAME = "maritaca-ai/sabia-7b"
OUTPUT_FILE = "results/sabia_predictions.json"
MAX_NEW_TOKENS = 64  # Reducido - respuestas extractivas son cortas
MAX_INPUT_LENGTH = 512
BATCH_SIZE = 16  # Igual que Tucano para comparación justa

def load_validation_data():
    with open("../data/flat_squad-dev-v1.1.json", "r", encoding="utf-8") as f:
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
            "answers": answers
        })
    return examples

def main():
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Para generación
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()

    examples = load_validation_data()
    print(f"Loaded {len(examples)} samples")

    os.makedirs("results", exist_ok=True)

    # Preparar todos los prompts
    all_prompts = []
    all_ids = []
    for ex in examples:
        context = ex["context"][:2000]  # Limitar caracteres
        prompt = extractive_prompt(context, ex["question"])
        all_prompts.append(prompt)
        all_ids.append(ex["id"])

    # Benchmarking
    torch.cuda.reset_peak_memory_stats()
    mem_inicio = torch.cuda.memory_allocated() / (1024**2)  # MB

    start_time = time.time()

    predictions = {}
    num_batches = (len(all_prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(num_batches), desc="Generating Sabiá"):
        start_idx = i * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(all_prompts))
        
        batch_prompts = all_prompts[start_idx:end_idx]
        batch_ids = all_ids[start_idx:end_idx]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        for j, (output, qid) in enumerate(zip(outputs, batch_ids)):
            input_len = inputs["input_ids"][j].shape[0]
            generated = output[input_len:]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            answer = answer.split("\n")[0].strip()  # Primera línea
            predictions[qid] = answer

    end_time = time.time()

    if torch.cuda.is_available():
        mem_pico = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        mem_actual = torch.cuda.memory_allocated() / (1024**2)  # MB
        mem_neta = mem_actual - mem_inicio  # MB
    else:
        mem_pico = 0
        mem_neta = 0

    elapsed_minutes = (end_time - start_time) / 60

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Guardar métricas de benchmarking
    metrics = {
        "inference_time_seconds": round(end_time - start_time, 2),
        "inference_time_minutes": round(elapsed_minutes, 2),
        "mem_pico_mb": int(mem_pico),
        "mem_neta_mb": int(mem_neta),
        "num_samples": len(predictions)
    }
    with open("results/sabia_zero_shot_benchmarks.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {len(predictions)} predictions to {OUTPUT_FILE}")
    print(f"Benchmarks: {elapsed_minutes:.2f} min, {mem_pico:.0f} MB peak, {mem_neta:.0f} MB net")

if __name__ == "__main__":
    main()
