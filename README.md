# Dokumentacja Systemu Wizji Komputerowej ASO River

Opracowane rozwiązania z zakresu wizji komputerowej dedykowane są do detekcji przeszkód rzecznych oraz semantycznej segmentacji wody. Badania i ewaluacja zostały przeprowadzone na zbiorze danych [LaRS (Lake and River Scene)](https://lars-dataset.github.io/). Pracę zrealizowano w ramach przedmiotu ASO (semestr 26L).

---

### Skład zespołu

- **Zofia Czyżewska**
- **Mateusz Ogniewski**
- **Mateusz Wawrzyniak**

Szczegółowy opis założeń projektowych, architektury systemu oraz uzyskanych wyników pomiarowych zamieszczono w sprawozdaniu [Etap 1](etap1.md).

---

![Zarys projektu](vii.png)

## Opis struktury i architektury systemu

Niniejsze repozytorium zawiera kompletny, modułowy potok obliczeniowy (pipeline) przetwarzania obrazów, na który składają się następujące elementy:

1. **Segmentacja semantyczna (`segmentation/`)**: Segmentator bazujący na architekturze sieci neuronowej U-Net, realizujący klasyfikację pikseli na trzy klasy: Woda, Przeszkoda oraz Niebo. Dodatkowo zaimplementowano referencyjny, tradycyjny algorytm nienadzorowany bazujący na metodzie przesunięcia średniej (_Mean-Shift_).
2. **Detekcja obiektów (`detection/`)**: Douczone modele detekcyjne Faster R-CNN (z siecią bazową ResNet-50 FPN) oraz Fast R-CNN (z siecią bazową ResNet-18) realizujące zadanie lokalizowania przeszkód wodnych w postaci prostokątów otaczających.
3. **Zintegrowany potok End-to-End (`main.py`)**: Konsoliduje moduły segmentacji i detekcji. Wykryte ramki otaczające są poddawane filtracji przestrzennej z wykorzystaniem dylatowanej maski segmentacji wody. Pozwala to na redukcję liczby błędnych detekcji (false positives) na obszarach lądowych, niebie oraz linii brzegowej.

---

## Wymagania systemowe i instalacja

- **Wersja Pythona**: `3.11` lub `3.12` (Uwaga: wersje `3.13+` nie są kompatybilne ze względu na strukturę paczek PyTorch)
- **Menedżer zależności**: [uv](https://docs.astral.sh/uv/) (wysoce zalecany ze względu na szybkość działania i powtarzalność środowiska)
- **Akceleracja GPU**: Karta NVIDIA wspierająca technologię CUDA jest używana automatycznie, jeśli jest dostępna. W przeciwnym razie system bezawaryjnie przełącza się na pracę na procesorze CPU.

Z poziomu głównego folderu projektu zainstaluj wszystkie wymagane zależności przy użyciu `uv`:

```bash
uv sync
```

### Konfiguracja zmiennych środowiskowych (Opcjonalnie)

W przypadku chęci monitorowania procesu uczenia oraz wersjonowania eksperymentów w usłudze Weights & Biases (W&B), należy utworzyć plik `.env` w katalogu głównym projektu i zdefiniować klucz dostępu API:

```bash
WANDB_API_KEY=wprowadz_klucz_api_wandb
```

### Struktura katalogu danych

System wymaga zachowania poniższej struktury dla zbioru danych LaRS v1.0.0:

```
data/lars_v1.0.0/
  images/
    train/images/
    val/images/
    test/images/
  annotations/
    train/
    val/
    test/
```

---

## Uruchamianie zintegrowanego potoku (`main.py`)

Skrypt [main.py](main.py) odpowiada za jednoczesne wywoływanie modeli segmentacji oraz detekcji.

```bash
uv run main.py --mode [pipeline|metrics|test] [argumenty]
```

### Tryby pracy programu

#### 1. Potok interaktywny (`--mode pipeline`)

Wizualizuje predykcje modeli na zbiorze walidacyjnym w czasie rzeczywistym. Segmentacja wody nakładana jest w postaci półprzezroczystej warstwy barwnej o definiowalnej palecie (BGR), z jednoczesnym nanoszeniem zweryfikowanych ramek otaczających zebranych z detektora.

```bash
uv run main.py --mode pipeline --detector-kind faster_rcnn --detector-score-thresh 0.35
```

#### 2. Ewaluacja konsolowa metryk (`--mode metrics`)

Uruchamia proces weryfikacji ilościowej w trybie bezokienkowym. Dokonuje zbiorczego wyznaczenia metryki Water IoU, Mean Bounding Box IoU oraz parametrów statystycznych Precision, Recall i F1-score.

```bash
uv run main.py --mode metrics --batch-size 8 --detector-water-filter
```

#### 3. Przetwarzanie zbiorów testowych (`--mode test`)

Dokonuje predykcji na obrazach zewnętrznych znajdujących się w dedykowanym folderze, po czym zapisuje uzyskane wyniki detekcji zgodnie ze specyfikacją formatu COCO w pliku (`test_data/detections.json`).

```bash
uv run main.py --mode test --test-data-dir test_data --div 1
```

### Obsługa interfejsu graficznego

- **Dowolny klawisz** (poza poniższymi): Wyświetlenie kolejnego przetworzonego obrazu.
- **`q` lub `Esc`**: Wyjście z programu i zamknięcie struktur okiennych interfejsu OpenCV.

---

## Izolowany moduł detekcji obiektów (`detection/`)

Procesy uczenia, strojenia progów oraz analizy jednostkowej modeli detekcyjnych realizowane są przez skrypt [detection/detection_main.py](detection/detection_main.py).

```bash
uv run detection/detection_main.py --mode [show|train|tune_thresh|sample] [argumenty]
```

### Tryby działania

- **Trenowanie detektora**: Dostrajanie (fine-tuning) wag sieci detekcyjnych na zbiorze treningowym LaRS.
  ```bash
  uv run detection/detection_main.py --mode train --epochs 20 --batch-size 4
  ```
- **Przeszukiwanie progu odrzucenia (Grid Sweep)**: Analizuje predykcje na zbiorze walidacyjnym w celu znalezienia optymalnego progu pewności detektora, dającego najwyższy globalny wynik F1.
  ```bash
  uv run detection/detection_main.py --mode tune_thresh
  ```
- **Interaktywny podgląd detekcji**: Wyświetla surowe predykcje modelu detektora na obrazach walidacyjnych.
  ```bash
  uv run detection/detection_main.py --mode show --checkpoint lars_faster_rcnn.pth
  ```
- **Wizualizacja etykiet referencyjnych (Ground-Truth)**: Wyświetla rzeczywiste ramki annotacji referencyjnych nałożone na surowe zdjęcia.
  ```bash
  uv run detection/detection_main.py --mode sample
  ```

### Wybór modelu Fast R-CNN (Selective Search)

Aby uruchomić i ocenić model Fast R-CNN wykorzystujący algorytm Selective Search do propozycji regionów:

```bash
uv run detection/main_with_fastrcnn.py --mode metrics
```

---

## Samodzielny potok segmentacji (`segmentation/`)

### Podejście głębokie (Deep Learning - `unet_main.py`)

Służy do trenowania i walidacji struktur U-Net dla semantycznej segmentacji obrazów.

- **Trening**:
  ```bash
  uv run segmentation/unet_main.py --mode train --batch-size 16 --epochs 50 --features 16 32 64 128
  ```
- **Walidacja wskaźnika mIoU**:
  ```bash
  uv run segmentation/unet_main.py --mode mean_iou --model-name best.pth
  ```
- **Wizualizacja predykcji**:
  ```bash
  uv run segmentation/unet_main.py --mode show --model-name best.pth
  ```

### Podejście bez nadzoru (Unsupervised C.V. - `means_shift_main.py`)

Nienadzorowana tradycyjna segmentacja przy użyciu filtrowania przestrzenno-kolorystycznego Mean-Shift, po którym następuje algorytm grupowania K-Średnich (K-Means).

- **Interaktywny podgląd**:
  ```bash
  uv run segmentation/means_shift_main.py --mode show --sp 32 --sr 128
  ```
- **Optymalizacja parametrów grupowania**:
  ```bash
  uv run segmentation/means_shift_main.py --mode optimize
  ```

---

## Wykaz dostępnych parametrów CLI

### Argumenty zintegrowanego potoku (`main.py`)

| Właściwość                   | Domyślnie                      | Opis                                                                           |
| ---------------------------- | ------------------------------ | ------------------------------------------------------------------------------ |
| `--mode`                     | `pipeline`                     | Wybór trybu pracy: `pipeline`, `metrics`, `test`                               |
| `--div`                      | `1`                            | Współczynnik skalowania obrazu w dół (np. `2` -> dwukrotne zmniejszenie)       |
| `--seg-model-path`           | `segmentation/models/best.pth` | Ścieżka do zapisanego pliku checkpointu modelu U-Net                           |
| `--detector-kind`            | `faster_rcnn`                  | Wybór architektury detektora: `faster_rcnn`, `fast_rcnn`                       |
| `--detector-checkpoint`      | `lars_faster_rcnn.pth`         | Nazwa pliku z wagami detektora wewnątrz `detection/models/`                    |
| `--detector-score-thresh`    | `0.35`                         | Próg pewności detekcji (confidence threshold)                                  |
| `--detector-water-filter`    | `True`                         | Filtrowanie obiektów leżących poza obszarem z dylatowaną maską wody            |
| `--detector-water-dilate-px` | `30`                           | Rozmiar dylatacji (poszerzenia) maski wody przy filtrowaniu rzek (w pikselach) |
| `--batch-size`               | `8`                            | Wielkość partii (batch size) podczas bezokienkowej ewaluacji metryk            |
| `--pipeline-fps`             | `2`                            | Częstotliwość klatek na sekundę podczas odtwarzania interaktywnego             |

### Argumenty modułu segmentacji opartego na głębokim uczeniu (`segmentation/unet_main.py`)

| Właściwość        | Domyślnie             | Opis                                                          |
| ----------------- | --------------------- | ------------------------------------------------------------- |
| `--mode`          | `show`                | Wybór trybu pracy: `train`, `show`, `sample`, `mean_iou`      |
| `--model-name`    | `best.pth`            | Nazwa pliku z wagami modelu (wew. katalogu `--models-dir`)    |
| `--models-dir`    | `segmentation/models` | Katalog przechowujący checkpointy wag modelu                  |
| `--preload`       | `None`                | Opcjonalne wczytanie wag początkowych do douczania            |
| `--features`      | `16 16 32 32 64 128`  | Definicja rozmiarów warstw filtrów dla sieci U-Net            |
| `--hsv`           | wejście wyłączone     | Konwersja przestrzeni barwnej odczytywanych obrazów do HSV    |
| `--div`           | `1`                   | Współczynnik skalowania obrazu (np. `2` -> pół-rozdzielczość) |
| `--batch-size`    | `8`                   | Rozmiar wsadu (batch size) do treningu i ewaluacji            |
| `--epochs`        | `40`                  | Całkowita liczba epok treningowych                            |
| `--lr`            | `3e-4`                | Początkowy współczynnik uczenia optimizer                     |
| `--augment`       | włączone              | Włączenie augmentacji danych treningowych                     |
| `--weighted-loss` | włączone              | Stosowanie ważonej funkcji straty uwzględniającej klasy       |

### Argumenty modułu tradycyjnej segmentacji (`segmentation/means_shift_main.py`)

| Właściwość | Domyślnie | Opis                                                        |
| ---------- | --------- | ----------------------------------------------------------- |
| `--mode`   | `show`    | Wybór trybu pracy: `show`, `mean_iou`, `optimize`           |
| `--sp`     | `32`      | Promień okna przestrzennego dla filtru Mean-Shift           |
| `--sr`     | `128`     | Promień okna kolorystycznego dla filtru Mean-Shift          |
| `--dim`    | `256`     | Maksymalny wymiar obrazu roboczego (szybkość przetwarzania) |

### Argumenty narzędzia detekcji (`detection/detection_main.py`)

| Właściwość       | Domyślnie              | Opis                                                        |
| ---------------- | ---------------------- | ----------------------------------------------------------- |
| `--mode`         | `show`                 | Wybór trybu pracy: `show`, `train`, `tune_thresh`, `sample` |
| `--checkpoint`   | `lars_faster_rcnn.pth` | Ścieżka/Nazwa pliku z wagami wewnątrz `detection/models/`   |
| `--backbone`     | `resnet50`             | Sieć bazowa: `resnet50`, `mobilenetv3_large`                |
| `--epochs`       | `20`                   | Liczba epok douczania                                       |
| `--batch-size`   | `4`                    | Wielkość partii treningowej (batch size)                    |
| `--score-thresh` | `0.35`                 | Próg pewności detekcji wyjściowej                           |

---

## Rozwiązywanie problemów

- **FileNotFoundError: Checkpoint not found**:  
  Upewnij się, że pliki z wyuczonymi wagami sieci znajdują się odpowiednio w katalogach `detection/models/` lub `segmentation/models/` bądź podaj ich pełne, bezwzględne ścieżki systemowe.
- **Folder not found: data/lars_v1.0.0**:  
  Pobierz i rozpakuj zbiór danych LaRS, upewniając się, że ścieżki i nazwy katalogów odpowiadają dokładnie schematowi przedstawionemu w punkcie [Struktura zbioru danych](#struktura-zbioru-danych).
- **Rozsynchronizowany Cache / Przesunięte Etykiety**:  
  Skasuj cały wygenerowany folder podręczny `data/lars_v1.0.0/cached/` i uruchom skrypt ponownie, aby zainicjować ponowne generowanie i wyrównanie tabel danych.
- **Błąd GUI QT / OpenCV w systemach Linux**:  
  Upewnij się, że serwer wyświetlania X11/XWayland jest poprawnie postawiony. W środowiskach opartych na Waylandzie eksport zmiennej środowiskowej `export QT_QPA_PLATFORM=xcb` przed uruchomieniem skrótu zazwyczaj przywraca poprawną obsługę okien.
