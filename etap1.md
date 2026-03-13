# Dokumentacja Projektu ASO - Etap I

## Wprowadzenie i Kontekst Projektu

#### 1.2 Kontekst Problemu

Pojazdy wodne poruszające się po rzekach napotykają liczne trudności związane z oceną otoczenia, zwłaszcza w warunkach np. dużego ruchu. Załoga musi nieustannie monitorować sytuację, identyfikować przeszkody, rozpoznawać znaki nawigacyjne i podejmować szybkie decyzje, co jest obciążające i podatne na błędy.

Problem jest istotny, ponieważ błędna ocena otoczenia może prowadzić do kolizji a nawet zagrożenia życia, a także utrudnia sprawne prowadzenie jednostki. Współczesne systemy wsparcia są często ograniczone, nie zapewniają pełnej autonomii i wymagają stałej uwagi człowieka.

Główne wyzwania to:

- Automatyczna detekcja i klasyfikacja przeszkód na wodzie.
- Rozpoznawanie i interpretacja znaków nawigacyjnych.
- Segmentacja obrazu w trudnych warunkach pogodowych i oświetleniowych.

#### 1.3 Cel Projektu

Zaprojektowanie systemu do autonomicznej ewaluacji otoczenia pojazdu wodnego znajdującego się na rzece w celu wspomagania załogi pojazdu. W skład tejże ewaluacji będzie wchodzić analiza obrazu z kamer umieszczonych na pokładzie okrętu, a dokładniej:

- Segmentacja semantyczna obrazu: Oddzielenie lustra wody od brzegu, nieba i roślinności.
- Detekcję przeszkód: Identyfikację i lokalizację obiektów na wodzie (np. łodzie, boje, pływające konary, rośliny, inni użytkownicy).
- Wykrywanie i klasyfikację znaków: Identyfikacja znaczenia poszczególnych znaków znajdujących się na obecnym szklaku żeglownym pojazdu.

## Możliwe Zastosowania

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

## Przegląd Rozwiązań (Stan Wiedzy)

3.1 Przegląd Istniejących Metod
Krótki opis kluczowych technik i algorytmów stosowanych obecnie do rozwiązania danego problemu.

3.2 Wady i ZaletyWskazanie mocnych i słabych stron przeanalizowanych metod. Dlaczego nie są one idealne dla danego kontekstu projektu?

## Wstępna Propozycja Rozwiązania

4.1 Architektura Systemu
Ogólny schemat proponowanego rozwiązania.

4.2 Narzędzia i Technologie
Wstępny wybór technologii, bibliotek i języków programowania.Przykład: Python, OpenCV, biblioteka PyTorch/TensorFlow (do ML), ROS (do robotyki/integracji).

## Analiza dostępnych zbiorów danych

5.1 Możliwe źródła danych do trenowania i testowania. Czy będą to dane syntetyczne, publiczne czy zbierane własnoręcznie?
5.2 Analiza poszczególnych zbiorów pod względem np. liczebności przykładów, poprawności formatu oznaczeń, itp.

## Harmonogram Wstępny i Kamienie Milowe
