# Testy porównawcze: Grover + LLM

## Cel
Ocena wpływu:
- użycia Grovera (vs. klasyczny algorytm),
- różnych modeli LLM (llama-3-8b, mixtral-8x7b, phi-3.5),
- obecności kontekstu (LLM ma top 3 konteksty, top 1 lub nie ma żadnych),
- czasu generowania odpowiedzi.

## Metryki
- Word overlap (między z kontekstami i bez),
- Cosine similarity (między z kontekstami/bez i odpowiedziami "idealnymi")
- Czas generacji (s),
- Trafność względem idealnych odpowiedzi (heurystycznie).

## Wyniki

