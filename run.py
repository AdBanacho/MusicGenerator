from proces import run

if __name__ == '__main__':
    dlugosc = 2000  # ~9sek
    blok = 20       # ~0.09sek jesli blok = 0, brak cyklicznosci i dzwieki sa usuwane losowo
    epoki = 20

    run(epoki=epoki, blok=blok, dlugosc=dlugosc, DL=False, ML=False, create_data=True)
