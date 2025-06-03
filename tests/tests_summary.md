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
- Czas generacji (s).

## Przeprowadzanie testów
Kod, za pomocą którego przeprowadzono testy dostępny w pliku evaluation/tests_runner.py.

## Wyniki
Pełne wyniki (metryki i odpowiedzi dla każdego sprawdzanego wariantu) dostępne w pliku evaluation/test_results.csv (oraz evaluation/test_results.json).

## Podsumowanie wyników
Podsumowanie wszystkich metryk dla każdego modelu i wariantu:
![Podsumowanie wszystkich metryk dla każdego modelu i wariantu.](.\evaluation\results_images\all_metrics_summary.png)

Tabela porównująca modele i warianty pod względem średniej cosine_similarity do odpowiedzi "idealnej":
![Tabela porównująca modele i warianty pod względem średniej cosine_similarity do odpowiedzi "idealnej".](.\evaluation\results_images\cosine_summary_table.png)

Wykres wartości średnich cosine_similarity do "idealnych" odpowiedzi:
![Wykres wartości średnich cosine_similarity do "idealnych" odpowiedzi.](.\evaluation\results_images\cosine_summary_plot.png)

Tabela porównująca średni czas generowania odpowiedzi (z kontekstem i bez) dla wszystkich 3 modeli w zależnosci od wariantu:
![Średni czas generowania odpowiedzi (z kontekstem i bez) dla wszystkich 3 modeli.](.\evaluation\results_images\time_summary_table.png)

Wykres porównujący średni czas generowania odpowiedzi (z kontekstem i bez) dla wszystkich 3 modeli w zależnosci od wariantu:
![Wykres wartości średnich czasu generowania odpowiedzi (z kontekstem i bez) dla wszystkich 3 modeli.](.\evaluation\results_images\time_summary_plot.png)

# Sprawdzenie działania GUI

## Strona startowa

![Strona startowa aplikacji.](.\evaluation\GUI_images\home_page.png)

## Ekran ładowania odpowiedzi

![Ekran ładowania odpowiedzi.](.\evaluation\GUI_images\loading_page.png)

## Odpowiedzi na pytanie zadane przez użytkownika

![Odpowiedzi na pytanie.](.\evaluation\GUI_images\answer_page.png)

![Odpowiedzi dalsze.](.\evaluation\GUI_images\answer_page2.png)

Ze zwiniętymi odpowiedziami (można rozwinąć odpowiedzi dla wybranego modelu):

![Strona ze zwiniętymi odpowiedziami.](.\evaluation\GUI_images\hidden_answers.png)

3 najlepsze konteksty znalezione przez algorytm Grovera:

![Top 3 konteksty](.\evaluation\GUI_images\top3.png)

10 najlepszych kontekstów znalezionych przez klasyczny algorytm:

![Top 10 kontekstów 1.](.\evaluation\GUI_images\top4.png)
![Top 10 kontekstów 2.](.\evaluation\GUI_images\top8.png)
![Top 10 kontekstów 3.](.\evaluation\GUI_images\top10.png)