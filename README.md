Wymagane: PyCharmCommunity Edition (najnowsza dostępna wersja)

Przygotowanie środowiska (po uruchomieniu aplikacji PyCharm):
1. File > New Project
1.1. Location: \"Twoja_nazwa"
1.2. > Create

2. Stwórz i aktywuj virtualne środowisko
2.1. W terminalu: 
python -m venv venv
2.2. # Windows:
venv\Scripts\activate
2.3. # Linux/Mac:
source venv\bin\activate

3. Zainstaluj wymagane pakiety
3.1. pip install flask pandas numpy scikit-learn joblib
3.2. Poczekaj aż ponownie pojawi się wiersz polecenia.

4. (Github) W celu pobrania aplikacji należy kliknąć:
4.1. "<> Code" > Download ZIP
4.2. Aplikacje umieść w folderze o ścieżce podanej obok nazwy twojego projektu
4.3. Wypakuj aplikację (Zip/WinRar)

5. Kliknij dwukrotnie w flask-app.py w folderze ML_app
5.1. Po prawej stronie w górnym rogu kliknij ikonę Run
5.2. W zakładce "Run" kliknij w link znajdujący się obok "Running on _"
5.3. Możesz używać aplikacji
