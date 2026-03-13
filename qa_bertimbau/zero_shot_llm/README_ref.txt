
# Descrição e organização da pasta zero_shot_llm

Esta pasta contém scripts e subpastas para treinamento, inferência e avaliação de modelos LoRA e zero-shot, principalmente para os modelos Sabiá e Tucano.

## Estrutura e propósito dos arquivos principais

- **train_lora_sabia.py / train_lora_tucano.py**: Treinam modelos LoRA para Sabiá e Tucano.
- **eval_lora_sabia.py / eval_lora_tucano.py**: Avaliam modelos LoRA para Sabiá e Tucano.
- **eval_zero_shot_sabia.py / eval_zero_shot_tucano.py**: Avaliam modelos em modo zero-shot.
- **zero_shot_sabia.py / zero_shot_tucano.py**: Scripts para inferência zero-shot.
- **lora_inference.py**: Realiza inferência usando modelos LoRA.
- **prepare_lora_data.py**: Prepara dados para LoRA.
- **preprocessing.py**: Processa dados antes de treinamento/avaliação.
- **prompts.py**: Define os prompts usados nos modelos.

## Subpastas
- **lora_sabia_qa/** e **lora_tucano_qa/**: Dados ou resultados de QA para LoRA Sabiá e Tucano.
- **sabia-lora/**: Checkpoints, configurações ou resultados de LoRA Sabiá.
- **results/**: Resultados de avaliações ou inferências (por exemplo, sabia_predictions.json, tucano_predictions.json).


Este arquivo serve como referência rápida para entender a organização e o propósito dos scripts e pastas. Se precisar de detalhes sobre algum arquivo ou processo, revise o script correspondente ou consulte o responsável pelo projeto.
