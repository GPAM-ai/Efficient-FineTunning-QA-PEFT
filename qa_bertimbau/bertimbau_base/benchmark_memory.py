#!/usr/bin/env python
"""
Script para medir el pico de memoria GPU de los 5 métodos de entrenamiento.
Ejecuta cada método con 1 epoch y reporta el uso máximo de memoria.
"""

import subprocess
import os
import time
import re
import shutil

# Configuración
SCRIPTS = [
    ("Full Fine-tuning", "main.py"),
    ("LoRA", "main_lora.py"),
    ("DoRA", "main_dora.py"),
    ("QLoRA (4-bit)", "main_qlora.py"),
    ("QDoRA (4-bit)", "main_qdora.py"),
]

BASE_DIR = "/data/nina/Efficient-FineTunning-QA-PEFT/qa_bertimbau/bertimbau_base/results"
GPU_ID = "0"
NUM_EPOCHS = 1


def modify_epochs(script_path, num_epochs):
    """Modifica el número de epochs en un script temporalmente."""
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Guardar backup
    backup_path = script_path + ".bak"
    shutil.copy(script_path, backup_path)
    
    # Modificar num_train_epochs
    modified = re.sub(
        r'num_train_epochs\s*=\s*\d+',
        f'num_train_epochs={num_epochs}',
        content
    )
    
    with open(script_path, 'w') as f:
        f.write(modified)
    
    return backup_path


def restore_script(script_path, backup_path):
    """Restaura el script original desde el backup."""
    if os.path.exists(backup_path):
        shutil.move(backup_path, script_path)


def get_gpu_memory_mb():
    """Obtiene la memoria GPU usada actual en MB usando nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"--id={GPU_ID}"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
        return 0


def run_benchmark(script_name, method_name):
    """Ejecuta un script y mide el pico de memoria GPU."""
    script_path = os.path.join(BASE_DIR, script_name)
    
    print(f"\n{'='*60}")
    print(f"Ejecutando: {method_name} ({script_name})")
    print('='*60)
    
    # Modificar a 1 epoch temporalmente
    backup_path = modify_epochs(script_path, NUM_EPOCHS)
    
    try:
        # Limpiar memoria GPU antes de empezar
        subprocess.run(["python", "-c", "import torch; torch.cuda.empty_cache()"], 
                       capture_output=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": GPU_ID})
        time.sleep(2)
        
        initial_memory = get_gpu_memory_mb()
        peak_memory = initial_memory
        
        # Iniciar el proceso de entrenamiento
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = GPU_ID
        
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=BASE_DIR
        )
        
        # Monitorear memoria mientras corre
        start_time = time.time()
        last_output = ""
        
        while process.poll() is None:
            current_memory = get_gpu_memory_mb()
            if current_memory > peak_memory:
                peak_memory = current_memory
            
            # Leer output para mostrar progreso
            try:
                line = process.stdout.readline()
                if line:
                    last_output = line.strip()
                    if "it/s" in line or "Epoch" in line or "loss" in line.lower():
                        print(f"\r  {last_output[:80]}", end="", flush=True)
            except:
                pass
            
            time.sleep(0.5)
        
        # Obtener código de salida y output final
        stdout, _ = process.communicate()
        elapsed_time = time.time() - start_time
        
        # Calcular memoria neta usada
        net_memory = peak_memory - initial_memory
        
        print(f"\n  Tiempo: {elapsed_time/60:.1f} min")
        print(f"  Memoria inicial: {initial_memory} MB")
        print(f"  Memoria pico: {peak_memory} MB")
        print(f"  Memoria neta usada: {net_memory} MB")
        
        if process.returncode != 0:
            print(f"  Error (código: {process.returncode})")
            # Mostrar últimas líneas del error
            if stdout:
                lines = stdout.strip().split('\n')
                for line in lines[-10:]:
                    print(f"    {line}")
        
        return {
            "method": method_name,
            "script": script_name,
            "peak_memory_mb": peak_memory,
            "net_memory_mb": net_memory,
            "time_min": elapsed_time / 60,
            "success": process.returncode == 0
        }
    
    finally:
        # Siempre restaurar el script original
        restore_script(script_path, backup_path)


def main():
    print("\n" + "="*60)
    print("BENCHMARK DE MEMORIA GPU - BERTimbau Base QA")
    print("="*60)
    print(f"GPU: {GPU_ID}")
    print(f"Directorio: {BASE_DIR}")
    print(f"Epochs: 1")
    print("="*60)
    
    results = []
    
    for method_name, script_name in SCRIPTS:
        try:
            result = run_benchmark(script_name, method_name)
            results.append(result)
        except Exception as e:
            print(f"Error ejecutando {method_name}: {e}")
            results.append({
                "method": method_name,
                "script": script_name,
                "peak_memory_mb": 0,
                "net_memory_mb": 0,
                "time_min": 0,
                "success": False
            })
        
        # Esperar entre ejecuciones para liberar memoria
        print("\nLiberando memoria...")
        subprocess.run(["python", "-c", "import torch; torch.cuda.empty_cache()"], 
                       capture_output=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": GPU_ID})
        time.sleep(5)
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    print(f"{'Método':<20} {'Mem Pico (MB)':<15} {'Mem Neta (MB)':<15} {'Tiempo (min)':<12} {'Estado'}")
    print("-"*60)
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{r['method']:<20} {r['peak_memory_mb']:<15} {r['net_memory_mb']:<15} {r['time_min']:<12.1f} {status}")
    
    # Guardar resultados en archivo
    with open(os.path.join(BASE_DIR, "benchmark_results.txt"), "w") as f:
        f.write("BENCHMARK DE MEMORIA GPU - BERTimbau Base QA\n")
        f.write("="*60 + "\n")
        f.write(f"{'Método':<20} {'Mem Pico (MB)':<15} {'Mem Neta (MB)':<15} {'Tiempo (min)':<12} {'Estado'}\n")
        f.write("-"*60 + "\n")
        for r in results:
            status = "OK" if r["success"] else "ERROR"
            f.write(f"{r['method']:<20} {r['peak_memory_mb']:<15} {r['net_memory_mb']:<15} {r['time_min']:<12.1f} {status}\n")
    
    print(f"\nResultados guardados en: {BASE_DIR}/benchmark_results.txt")


if __name__ == "__main__":
    main()
