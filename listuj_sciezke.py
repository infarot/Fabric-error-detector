from os import walk
def listuj_sciezke(katalog):
    for (dirpath, katalogi, pliki) in walk(katalog):
        break
    return [katalogi, pliki]
