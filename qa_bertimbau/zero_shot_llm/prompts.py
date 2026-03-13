def extractive_prompt(context: str, question: str) -> str:
    return f"""Responda exatamente com um trecho literal do contexto.
Não explique, não resuma, não invente palavras.
Se a resposta não estiver explicitamente no contexto, responda com vazio.

Contexto:
{context}

Pergunta:
{question}

Resposta:"""
