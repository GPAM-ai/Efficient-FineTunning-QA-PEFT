import json
import time
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
import torch

# -----------------------------
# Configuración general
# -----------------------------
MODEL_NAME = "TucanoBR/tucano-1b1-instruct"  # o "Sabiá"
OUTPUT_DIR = "./lora_tucano_qa"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 64
BATCH_SIZE = 2  # ajusta según VRAM
EPOCHS = 3
LR = 5e-5

# -----------------------------
# Cargar tokenizer y modelo
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # IMPORTANTE para padding

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,       
    device_map="auto"
)

# -----------------------------
# Configurar LoRA
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -----------------------------
# Preparar dataset QA extractiva
# -----------------------------
def flatten_qa_dataset(file_path="../data/flat_squad-dev-v1.1.json"):
    """Convierte dataset QA plano en Dataset para LoRA"""
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for item in raw["data"]:
        context = item["context"]
        question = item["question"]
        answers = item.get("answers", {})

        # saltar si no hay respuesta
        if not answers or not answers.get("text"):
            continue

        examples.append({
            "input_text": f"Contexto: {context}\nPergunta: {question}\nResposta:",
            "target_text": answers["text"][0]  # usar la primera respuesta
        })

    return Dataset.from_list(examples)

dataset = flatten_qa_dataset()

# Dividir en train/validation (90/10)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# -----------------------------
# Tokenizar dataset (prompt + target concatenado)
# Labels: -100 para el prompt, ids reales para respuesta
# -----------------------------
def preprocess(example):
    # Tokenizar respuesta primero para saber cuánto espacio reservar
    label_ids = tokenizer(
        example["target_text"],
        truncation=True,
        max_length=MAX_OUTPUT_LENGTH,
        add_special_tokens=False
    )["input_ids"]
    
    # Truncar input dejando espacio para la respuesta
    max_input = MAX_INPUT_LENGTH - len(label_ids) - 1  # -1 para EOS
    input_ids = tokenizer(
        example["input_text"],
        truncation=True,
        max_length=max_input,
        add_special_tokens=False
    )["input_ids"]

    # Concatenar: input + respuesta + EOS
    full_input_ids = input_ids + label_ids + [tokenizer.eos_token_id]
    
    # Labels: -100 para el input (no calcular loss), ids reales para respuesta
    labels = [-100] * len(input_ids) + label_ids + [tokenizer.eos_token_id]
    
    attention_mask = [1] * len(full_input_ids)

    return {
        "input_ids": full_input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

tokenized_train = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)

# -----------------------------
# Data collator
# -----------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding=True,
    return_tensors="pt"
)

# -----------------------------
# TrainingArguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    save_total_limit=2,
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator
)

# -----------------------------
# Entrenamiento
# -----------------------------
# Reset VRAM tracking y medir memoria inicial
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_inicio = torch.cuda.memory_allocated() / (1024**2)  # MB
else:
    mem_inicio = 0

start_time = time.time()
trainer.train()
end_time = time.time()

train_time = end_time - start_time

if torch.cuda.is_available():
    torch.cuda.synchronize()
    mem_pico = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    mem_actual = torch.cuda.memory_allocated() / (1024**2)  # MB
    mem_neta = mem_actual - mem_inicio
else:
    mem_pico = 0
    mem_neta = 0

# -----------------------------
# Guardar modelo LoRA entrenado
# -----------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Guardar métricas de entrenamiento
os.makedirs("results", exist_ok=True)
train_metrics = {
    "model": MODEL_NAME,
    "config": "LoRA",
    "train_time_seconds": train_time,
    "train_time_minutes": round(train_time / 60, 1),
    "mem_pico_mb": round(mem_pico, 0),
    "mem_neta_mb": round(mem_neta, 0),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR
}
with open("results/tucano_lora_train_metrics.json", "w", encoding="utf-8") as f:
    json.dump(train_metrics, f, ensure_ascii=False, indent=2)

print(f"\nLoRA Tucano guardado en {OUTPUT_DIR}")
print(f"Tiempo de entrenamiento: {train_time/60:.1f} minutos")
print(f"Mem Pico: {mem_pico:.0f} MB")
print(f"Mem Neta: {mem_neta:.0f} MB")
print(f"Métricas guardadas en: results/tucano_lora_train_metrics.json")
