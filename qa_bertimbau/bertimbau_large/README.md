# bertimbau_large

Esta pasta contém arquivos e scripts relacionados ao modelo Bertimbau Large.

## Estrutura e significado dos arquivos

- **main.py**: Script principal para executar o modelo Bertimbau Large.
- **main_dora.py**: Versão do pipeline que usa a abordagem DoRA (experimentos específicos).
- **main_lora.py**: Pipeline para treinar/avaliar usando LoRA.
- **main_predict.py**: Script para rodar predições com modelos treinados.
- **main_qdora.py**: Variante de DoRA com configuração Q (experimentos específicos).
- **main_qlora.py**: Variante de LoRA com configuração Q (experimentos específicos).
- **preprocessing.py**: Funções para processar e limpar os dados antes do treinamento.
- **postprocessing.py**: Funções para processar saídas e gerar formatos finais de resultados.
- **analysis_bertimbau_large.ipynb**: Notebook para análise de resultados e métricas.

### Pastas

- **figures/**: Gráficos e visualizações geradas a partir dos resultados.
- **results/**: Resultados das avaliações e predições (JSON, CSV, etc.).

