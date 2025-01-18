1. Charakterystyka oprogramowania

a. Nazwa skrócona:
Porównywarka modeli

b. Nazwa pełna:
Analiza porównawcza wybranych modeli uczenia maszynowego

c. Krótki opis:
Aplikacja wykorzystuje trzy różne zbiory danych związane z medycyną, aby wspierać
diagnozowanie chorób. Po udzieleniu odpowiedzi na pytania przez użytkownika, program
analizuje informacje i prezentuje prawdopodobieństwo wystąpienia danej choroby (raka
płuc, cukrzycy, choroby serca). Wyniki są obliczane w oparciu o trzy wybrane modele
uczenia maszynowego: regresję logistyczną, drzewo decyzyjne oraz las losowy.

2. Prawa autorskie

a. Autorzy:
- Adam Wrzałek
- Bartosz Deptuła
- Mikołaj Mazur

b. Warunki licencyjne do oprogramowania wytworzonego przez grupę

MIT License
Copyright (c) [2025] [Adam Wrzałek, Bartosz Deptuła, Mikołaj Mazur]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

3. Specyfikacja wymagań

![image](https://github.com/user-attachments/assets/e0713495-57dd-4773-9178-292193e354d8)

4. Architektura systemu/oprogramowania
Aplikacja opiera się na architekturze klient-serwer z trzema głównymi komponentami:
frontendem (HTML, CSS), backendem (Flask Framework) oraz modelami uczenia
maszynowego (las losowy, regresja logistyczna, drzewo decyzyjne).
Jako główny język programowania został wykorzystany Python.
Frontend umożliwia użytkownikowi interakcję z systemem poprzez formularze
wprowadzania danych, wyniki predykcji oraz historię poprzednich analiz.
Backend przetwarza dane użytkownika, obsługuje logikę aplikacji oraz komunikuje się z
modelami uczenia maszynowego, które są dynamicznie wczytywane na podstawie
wybranego zbioru danych (np. choroby serca, cukrzyca, rak płuc).
Dane wejściowe są odpowiednio skalowane, a wyniki predykcji są zwracane i zapisywane.
Aplikacja korzysta z bazy danych do przechowywania użytkowników, wprowadzonych
danych oraz wyników predykcji poprzez wykorzystanie SQLalchemy.
Architektura aplikacji wspiera filtrowanie wyników (np. wyświetlanie historii na
podstawie zbioru danych), zarządzanie predykcjami (np. usuwanie wybranych analiz), a
także prezentację wyników w sposób przejrzysty i zrozumiały dla użytkownika.
Dzięki modułowej budowie, system można łatwo rozszerzać o nowe modele predykcyjne lub
dodatkowe funkcjonalności.

Biblioteki użyte w aplikacji:

![image](https://github.com/user-attachments/assets/87bd5022-271e-4412-afe5-c51c304dc916)

5. Scenariusze testowe

a. Rejestrowanie użytkownika

![image](https://github.com/user-attachments/assets/3ca881f2-adeb-4f5a-80bb-cd61ff797916)

b. Logowanie użytkownika

![image](https://github.com/user-attachments/assets/4bd0b056-b21a-4bab-9e0c-aa36713e11fa)

c. Przeprowadzenie predykcji dla chorób serca

![image](https://github.com/user-attachments/assets/c5bb2a3a-e4c6-4d0f-a077-50ccd9789e79)

d. Przeprowadzenie predykcji dla cukrzycy

![image](https://github.com/user-attachments/assets/a457c7f6-fc67-46ed-80dc-05f19be92857)

e. Przeprowadzenie predykcji dla raka płuc

![image](https://github.com/user-attachments/assets/4ff8a158-6015-4d42-bc0f-7aa7563510c9)

f. Zapisywanie elementów w historii predykcji

![image](https://github.com/user-attachments/assets/548a218b-b2a1-43d8-a638-e063a8be7253)

g. Sprawdzenie czy inni użytkownicy nie mają dostępu do danych

![image](https://github.com/user-attachments/assets/4d0dfde3-fe25-4829-b533-737f36ffdedb)
