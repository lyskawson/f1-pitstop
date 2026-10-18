# Predykcja pit stopów F1 — zakres projektu

## O projekcie

Budujemy sieci neuronowe, które na podstawie danych z kolejnych okrążeń przewidują, czy kierowca F1 zjedzie do boksu na następnym okrążeniu. Wyniki pokazujemy w interaktywnym dashboardzie.

- **Dane:** prawdziwe okrążenia z FastF1, sezony 2022–2025.
- **Modele:** prosty baseline i trzy sieci — MLP, GRU/LSTM, 1D-CNN/TCN.
- **Ocena:** uczymy na sezonach 2022–2024, testujemy na 2025.
- **Demo:** dashboard w Streamlit — prawdopodobieństwo zjazdu okrążenie po okrążeniu na tle prawdziwych pit stopów.

## Podział ról

Każdy ma swój obszar i trenuje jedną sieć.

- **A — dane:** pobranie i przygotowanie danych z FastF1; sieć MLP.
- **B — modele:** trening i porównanie modeli; sieć GRU/LSTM.
- **C — demo:** dashboard i prezentacja; sieć 1D-CNN/TCN.

## Harmonogram

Demo gotowe przed świętami, prezentacja w połowie stycznia (19.10.2026–17.01.2027).

| Data | Kamień milowy |
| --- | --- |
| 25.10.2026 | Start |
| 8.11.2026 | Dane gotowe |
| 6.12.2026 | Modele wytrenowane |
| 20.12.2026 | Demo gotowe |
| 21.12.2026–3.01.2027 | Przerwa świąteczna |
| 17.01.2027 | Prezentacja |

Najpierw dane — od nich zależy reszta, więc powinny być gotowe na początku listopada.

## Źródła

- [FastF1 — dokumentacja danych okrążeń](https://docs.fastf1.dev/api_reference/timing_data.html)
- [Sasikumar, Leema, Balakrishnan (2025). Data-driven pit stop decision support for Formula 1 using deep learning models](https://doi.org/10.3389/frai.2025.1673148)
