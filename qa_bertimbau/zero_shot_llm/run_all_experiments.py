#!/usr/bin/env python
"""
Script para ejecutar todos los experimentos y generar tabla de resultados.

Ejecuta:
1. Zero-shot Tucano
2. Zero-shot Sabiá
3. Training LoRA Tucano (opcional)
4. Training LoRA Sabiá (opcional)
5. Evaluación de todos los modelos
6. Genera tabla final
"""

import subprocess
import json
import os
import sys
from datetime import datetime
import argparse

RESULTS_DIR = "results"

def run_script(script_name, description):
    """Ejecuta un script Python y retorna si fue exitoso."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"ERROR: {script_name} falló con código {result.returncode}")
        return False
    return True

def collect_results():
    """Recolecta todos los resultados de los JSONs generados."""
    results = []
    
    # Zero-shot Tucano
    try:
        with open(f"{RESULTS_DIR}/tucano_zero_shot_metrics.json") as f:
            data = json.load(f)
        # Intentar cargar benchmarks (opcional)
        tiempo_min = "-"
        mem_pico = "-"
        mem_neta = "-"
        try:
            with open(f"{RESULTS_DIR}/tucano_zero_shot_benchmarks.json") as f:
                bench = json.load(f)
                tiempo_min = bench.get("inference_time_minutes", "-")
                mem_pico = bench.get("mem_pico_mb", "-")
                mem_neta = bench.get("mem_neta_mb", "-")
        except FileNotFoundError:
            pass
        
        results.append({
            "Model": "Tucano 1B",
            "Config": "Zero-shot",
            "EM (%)": round(data.get("exact_match", 0), 2),
            "F1 (%)": round(data.get("f1", 0), 2),
            "Tiempo (min)": tiempo_min,
            "Mem Pico (MB)": mem_pico,
            "Mem Neta (MB)": mem_neta
        })
    except FileNotFoundError:
        print("  Advertencia: No se encontró tucano_zero_shot_metrics.json")
    
    # Zero-shot Sabiá
    try:
        with open(f"{RESULTS_DIR}/sabia_zero_shot_metrics.json") as f:
            data = json.load(f)
        # Intentar cargar benchmarks (opcional)
        tiempo_min = "-"
        mem_pico = "-"
        mem_neta = "-"
        try:
            with open(f"{RESULTS_DIR}/sabia_zero_shot_benchmarks.json") as f:
                bench = json.load(f)
                tiempo_min = bench.get("inference_time_minutes", "-")
                mem_pico = bench.get("mem_pico_mb", "-")
                mem_neta = bench.get("mem_neta_mb", "-")
        except FileNotFoundError:
            pass
        
        results.append({
            "Model": "Sabiá 7B",
            "Config": "Zero-shot",
            "EM (%)": round(data.get("exact_match", 0), 2),
            "F1 (%)": round(data.get("f1", 0), 2),
            "Tiempo (min)": tiempo_min,
            "Mem Pico (MB)": mem_pico,
            "Mem Neta (MB)": mem_neta
        })
    except FileNotFoundError:
        print("  Advertencia: No se encontró sabia_zero_shot_metrics.json")
    
    # LoRA Tucano
    try:
        with open(f"{RESULTS_DIR}/tucano_lora_train_metrics.json") as f:
            train_data = json.load(f)
        with open(f"{RESULTS_DIR}/tucano_lora_metrics.json") as f:
            eval_data = json.load(f)
        results.append({
            "Model": "Tucano 1B",
            "Config": "LoRA",
            "EM (%)": round(eval_data.get("exact_match", 0), 2),
            "F1 (%)": round(eval_data.get("f1", 0), 2),
            "Tiempo (min)": train_data.get("train_time_minutes", "-"),
            "Mem Pico (MB)": int(train_data.get("mem_pico_mb", 0)),
            "Mem Neta (MB)": int(train_data.get("mem_neta_mb", 0))
        })
    except FileNotFoundError as e:
        print(f"  Advertencia: {e}")
    
    # LoRA Sabiá
    try:
        with open(f"{RESULTS_DIR}/sabia_lora_train_metrics.json") as f:
            train_data = json.load(f)
        with open(f"{RESULTS_DIR}/sabia_lora_metrics.json") as f:
            eval_data = json.load(f)
        results.append({
            "Model": "Sabiá 7B",
            "Config": "LoRA",
            "EM (%)": round(eval_data.get("exact_match", 0), 2),
            "F1 (%)": round(eval_data.get("f1", 0), 2),
            "Tiempo (min)": train_data.get("train_time_minutes", "-"),
            "Mem Pico (MB)": int(train_data.get("mem_pico_mb", 0)),
            "Mem Neta (MB)": int(train_data.get("mem_neta_mb", 0))
        })
    except FileNotFoundError as e:
        print(f"  Advertencia: {e}")
    
    return results

def print_table(results):
    """Imprime tabla formateada y la retorna como string."""
    if not results:
        print("No hay resultados para mostrar.")
        return ""
    
    # Headers
    headers = ["Model", "Config", "EM (%)", "F1 (%)", "Tiempo (min)", "Mem Pico (MB)", "Mem Neta (MB)"]
    
    # Calcular anchos de columnas
    widths = {h: len(h) for h in headers}
    for row in results:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))
    
    # Construir tabla
    separator = "+" + "+".join("-" * (widths[h] + 2) for h in headers) + "+"
    header_row = "|" + "|".join(f" {h:^{widths[h]}} " for h in headers) + "|"
    
    lines = []
    lines.append(separator)
    lines.append(header_row)
    lines.append(separator)
    
    for row in results:
        row_str = "|" + "|".join(f" {str(row.get(h, '')):^{widths[h]}} " for h in headers) + "|"
        lines.append(row_str)
    
    lines.append(separator)
    
    table_str = "\n".join(lines)
    print("\n" + table_str)
    
    return table_str

def save_results_csv(results):
    """Guarda resultados en CSV."""
    if not results:
        return
    
    headers = ["Model", "Config", "EM (%)", "F1 (%)", "Tiempo (min)", "Mem Pico (MB)", "Mem Neta (MB)"]
    
    with open(f"{RESULTS_DIR}/final_results.csv", "w") as f:
        f.write(",".join(headers) + "\n")
        for row in results:
            f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")
    
    print(f"\nResultados guardados en: {RESULTS_DIR}/final_results.csv")

def main():
    parser = argparse.ArgumentParser(description="Ejecutar experimentos de QA")
    parser.add_argument("--skip-training", action="store_true", 
                        help="Salta el entrenamiento LoRA (solo eval de existentes)")
    parser.add_argument("--only-zero-shot", action="store_true",
                        help="Solo ejecuta Zero-shot (sin LoRA)")
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"  EJECUTANDO EXPERIMENTOS")
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.skip_training:
        print(f"  Modo: Zero-shot + LoRA (eval solo)")
    elif args.only_zero_shot:
        print(f"  Modo: Solo Zero-shot")
    else:
        print(f"  Modo: Completo (Zero-shot + LoRA entrenamiento)")
    print(f"{'#'*60}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Lista de experimentos a ejecutar
    experiments = [
        # Zero-shot primero (más rápido para validar)
        ("zero_shot_tucano.py", "Zero-shot Tucano (inferencia)"),
        ("eval_zero_shot_tucano.py", "Evaluación Zero-shot Tucano"),
        
        ("zero_shot_sabia.py", "Zero-shot Sabiá (inferencia)"),
        ("eval_zero_shot_sabia.py", "Evaluación Zero-shot Sabiá"),
        
        # Luego LoRA training (si no está en --only-zero-shot)
        ("train_lora_tucano.py", "Training LoRA Tucano"),
        ("eval_lora_tucano.py", "Evaluación LoRA Tucano"),
        
        ("train_lora_sabia.py", "Training LoRA Sabiá"),
        ("eval_lora_sabia.py", "Evaluación LoRA Sabiá"),
    ]
    
    # Filtrar experimentos según opciones
    if args.only_zero_shot:
        # Solo Zero-shot
        experiments = experiments[:4]
    elif args.skip_training:
        # Zero-shot + LoRA eval (sin training)
        experiments = experiments[:4] + [
            ("eval_lora_tucano.py", "Evaluación LoRA Tucano (EXISTENTES)"),
            ("eval_lora_sabia.py", "Evaluación LoRA Sabiá (EXISTENTES)"),
        ]
    
    # Ejecutar cada experimento
    for script, description in experiments:
        if not os.path.exists(script):
            print(f"  SALTANDO: {script} no existe")
            continue
        
        success = run_script(script, description)
        if not success:
            print(f"  Continuando a pesar del error en {script}...")
    
    # Recolectar y mostrar resultados
    print(f"\n{'#'*60}")
    print(f"  RESULTADOS FINALES")
    print(f"{'#'*60}")
    
    results = collect_results()
    table_str = print_table(results)
    save_results_csv(results)
    
    # Guardar también como JSON
    with open(f"{RESULTS_DIR}/final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Guardar tabla en TXT
    with open(f"{RESULTS_DIR}/final_results.txt", "w") as f:
        f.write(f"RESULTADOS - QA Extractivo (SQuAD PT-BR)\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        f.write(table_str)
        f.write("\n")
    
    print(f"Tabla guardada en: {RESULTS_DIR}/final_results.txt")
    print(f"\nExperimentos completados!")

if __name__ == "__main__":
    main()
