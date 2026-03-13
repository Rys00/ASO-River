# Dokumentacja Projektu ASO - Etap I

## 1. Wprowadzenie i Kontekst Projektu

#### 1.1 Kontekst Problemu

Pojazdy wodne poruszające się po rzekach napotykają liczne trudności związane z oceną otoczenia, zwłaszcza w warunkach np. dużego ruchu. Załoga musi nieustannie monitorować sytuację, identyfikować przeszkody, rozpoznawać znaki nawigacyjne i podejmować szybkie decyzje, co jest obciążające i podatne na błędy.

Problem jest istotny, ponieważ błędna ocena otoczenia może prowadzić do kolizji a nawet zagrożenia życia, a także utrudnia sprawne prowadzenie jednostki. Współczesne systemy wsparcia są często ograniczone, nie zapewniają pełnej autonomii i wymagają stałej uwagi człowieka.

Główne wyzwania to:

- Automatyczna detekcja i klasyfikacja przeszkód na wodzie.
- Rozpoznawanie i interpretacja znaków nawigacyjnych.
- Segmentacja obrazu w trudnych warunkach pogodowych i oświetleniowych.

#### 1.2 Cel Projektu

Zaprojektowanie systemu do autonomicznej ewaluacji otoczenia pojazdu wodnego znajdującego się na rzece w celu wspomagania załogi pojazdu. W skład tejże ewaluacji będzie wchodzić analiza obrazu z kamer umieszczonych na pokładzie okrętu, a dokładniej:

- Segmentacja semantyczna obrazu: Oddzielenie lustra wody od brzegu, nieba i roślinności.
- Detekcję przeszkód: Identyfikację i lokalizację obiektów na wodzie (np. łodzie, boje, pływające konary, rośliny, inni użytkownicy).
- Wykrywanie i klasyfikację znaków: Identyfikacja znaczenia poszczególnych znaków znajdujących się na obecnym szklaku żeglownym pojazdu.
- Nie użycie YOLO.

## 2. Możliwe Zastosowania

#### 2.1 Obszary Zastosowań

System może znaleźć zastosowanie w:

- Żegludze śródlądowej (statki, promy, barki)
- Jednostkach rekreacyjnych (jachty, łodzie motorowe)
- Ratownictwie wodnym i patrolach policyjnych
- Transportach towarowych na rzekach
- Automatyzacji i wsparciu autonomicznych pojazdów wodnych

#### 2.2 Możliwe korzyści

Wdrożenie rozwiązania przyniesie:

- Zwiększenie bezpieczeństwa żeglugi poprzez szybką detekcję przeszkód i zagrożeń
- Redukcję ryzyka kolizji i wypadków
- Oszczędność czasu i zmniejszenie obciążenia załogi
- Zwiększenie precyzji nawigacji i interpretacji znaków
- Możliwość pracy w trudnych warunkach pogodowych i przy ograniczonej widoczności

## 3. Przegląd Rozwiązań (Stan Wiedzy)

#### 3.1 Przegląd Istniejących Metod

W literaturze podejście do percepcji sceny wodnej z kamer najczęściej dzieli się na trzy uzupełniające się zadania:

1. Segmentację semantyczną sceny (woda / brzeg / niebo / roślinność),
2. Detekcję przeszkód i obiektów pływających w obszarze wody
3. Wykrywanie i klasyfikację znaków nawigacyjnych.

Spotykane rozwiązania to najczęściej:

- Segmentacja semantyczna: historycznie stosowano metody klasyczne (progowanie barw, cechy tekstury, detekcja linii horyzontu), jednak w praktyce dominują sieci głębokie typu encoder–decoder (np. U-Net oraz rodziny DeepLab/PSPNet). W środowisku wodnym istotnym problemem są odbicia, fale i kilwater, przez co modele „drogowe” generują liczne fałszywe detekcje wody/nie-wody. Przykładowo, architektura WaSR (Water Segmentation and Refinement) proponuje segmentację wody z dodatkowym „oczyszczaniem” wyniku, wykorzystując encoder (ResNet101 z konwolucjami atrous) oraz dekoder fuzujący cechy z informacją inercyjną (IMU), co poprawia działanie w niejednoznacznych warunkach (np. mgła na horyzoncie) i redukuje FP, zwiększając F1.
- Segmentacja wsparta wiedzą dziedzinową: obok „czystego” uczenia głębokiego pojawiają się metody, które jawnie wprowadzają wiedzę o naturze powierzchni wody (nieregularność, dynamiczność). Przykładem jest podejście knowledge-driven, w którym zaprojektowany detektor cech specyficznych dla wody (LToF) działa jako składnik probabilistyczny (funkcja wiarygodności) łączony w ramie bayesowskiej z segmentacją U-Net, co ma zmniejszać zależność od samej ilości danych i poprawiać odporność między zbiorami.

- Detekcja przeszkód/obiektów: dla obiektów na wodzie (łodzie, boje, konary, roślinność, człowiek w wodzie) stosuje się detektory obiektów: dwuetapowe (np. Faster R-CNN – zwykle dokładniejsze, wolniejsze) oraz jednoetapowe (np. SSD/RetinaNet – zwykle szybsze). W praktycznych systemach często łączy się segmentację wody (maska obszaru zainteresowania) z detekcją obiektów, aby ograniczać liczbę fałszywych alarmów poza akwenem. W pracy arXiv dotyczącej detekcji obiektów pływających w rzekach i jeziorach porównano między innymi SSD i Faster R-CNN (oraz YOLOv5) pod kątem skuteczności w rozpoznawaniu „debris”, pokazując typowy kompromis: prędkość działania vs. precyzja.

- Wykrywanie i klasyfikacja znaków: najczęściej jest to detekcja obiektów (znak jako obiekt) połączona z klasyfikacją znaczenia. W literaturze dla znaków nawigacyjnych na wodach śródlądowych popularne są modele z rodziny YOLO (ze względu na szybkość). Artykuł w TransNav analizował wiele wersji YOLO (1–12) dla detekcji znaków śródlądowych i wskazał, że „nowsze” nie zawsze są najlepsze: przy kryterium wydajności detekcji wyróżniał się YOLOv4, a przy kryterium dokładności lokalizacji – między innymi YOLOv7 (dokładność pozioma) i YOLOv10 (dokładność pionowa); zwrócono też uwagę, że starsze wersje mogą ograniczać liczbę fałszywych detekcji. Alternatywnie (zwłaszcza gdy nie zakłada się użycia YOLO) rozważa się detektory dwuetapowe lub architektury oparte o transformatory (np. podejścia typu DETR) oraz pipeline „detekcja → klasyfikacja”, co bywa korzystne dla małych obiektów i złożonych klas.

#### 3.2 Wady i Zalety

Najważniejsze mocne strony i ograniczenia analizowanych podejść (w kontekście rzek i kamer pokładowych) to:

- Metody głębokie (segmentacja/detekcja) zapewniają wysoką skuteczność, ale wymagają dużych, dobrze oznaczonych danych oraz są wrażliwe na zmianę domeny: inne rzeki, inny kolor wody, pora dnia, mgła, deszcz, odblaski i kilwater.
- Segmentacja wody istotnie redukuje przestrzeń poszukiwań dla detekcji przeszkód, jednak błędy segmentacji propagują się dalej (np. „dziury” w masce wody mogą ukryć przeszkodę; błędna klasyfikacja odbić może generować FP).
- Podejścia specyficzne dla środowiska wodnego (np. WaSR) zmniejszają liczbę fałszywych alarmów dzięki lepszemu modelowaniu wody i/lub fuzji z IMU, ale zwiększają złożoność systemu (integracja dodatkowych danych, strojenie, koszty obliczeń i utrzymania).
- Podejścia knowledge-driven mogą poprawiać odporność na braki danych i zmienność sceny, ale wymagają poprawnie dobranych założeń/heurystyk (np. jakie cechy są „typowe” dla wody) i nadal mogą tracić skuteczność w nietypowych warunkach.
- Detekcja znaków nawigacyjnych jest trudna ze względu na mały rozmiar obiektów w kadrze, częściowe zasłonięcia (gałęzie/roślinność), rozmycie ruchu oraz duże podobieństwo klas. Zastosowanie bardzo szybkich detektorów ułatwia działanie w czasie rzeczywistym, ale często odbywa się kosztem stabilności lokalizacji i odporności na FP/zmianę oświetlenia.

Powyższe ograniczenia uzasadniają potrzebę projektowania rozwiązania dedykowanego scenom wodnym (rzeka) oraz zwracają uwagę na dobór metod, które będą możliwe do uruchomienia w czasie rzeczywistym na docelowym sprzęcie, a jednocześnie będą odporne na typowe zjawiska dla środowiska wodnego.

## 4. Wstępna Propozycja Rozwiązania

4.1 Architektura Systemu
Ogólny schemat proponowanego rozwiązania.

4.2 Narzędzia i Technologie
Wstępny wybór technologii, bibliotek i języków programowania.Przykład: Python, OpenCV, biblioteka PyTorch/TensorFlow (do ML), ROS (do robotyki/integracji).

## 5. Analiza dostępnych zbiorów danych

5.1 Możliwe źródła danych do trenowania i testowania. Czy będą to dane syntetyczne, publiczne czy zbierane własnoręcznie?
5.2 Analiza poszczególnych zbiorów pod względem np. liczebności przykładów, poprawności formatu oznaczeń, itp.

## 6. Harmonogram Wstępny i Kamienie Milowe

| Kamień milowy                     | Szacunkowa data ukończenia |
| --------------------------------- | -------------------------- |
| Dokumentacja wstępna              | 17.03.2026                 |
| Działajaca segmentacja            | 12.04.2026 (26 dni)        |
| Działające wykrywanie obiektów    | 08.05.2026 (26 dni)        |
| Działająca identyfikacja obiektów | 03.06.2026 (26 dni)        |
| Dokumentacja końcowa              | 16.06.2026 (13 dni)        |
