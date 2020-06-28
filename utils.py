import yaml
import unicodedata
import codecs

def savex(w):
    fileW = codecs.open("trash/trash.txt", "w", "utf-8")
    fileW.write(str(w))
    fileW.close()

def loadx():
    with open("trash/trash.txt", "rb") as f:
        words = f.read().decode("UTF-8")
    words = yaml.load(words,Loader=yaml.FullLoader)
    if words is None:
        return {}
    return words