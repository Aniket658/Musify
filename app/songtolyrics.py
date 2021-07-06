import json
from lyrics_extractor import SongLyrics
import re
import csv


extract_lyrics = SongLyrics('AIzaSyCM-tiCW5V75fQIAnySF2EMUCnP9MJOP_g','0930e3ce67fe2b3cc')

def getlyrics(song, genre):
    s = extract_lyrics.get_lyrics(song)['lyrics']
    s = s.replace(".", "")
    s = s.replace(",", "")
    s = re.sub(r'\n+', " ", s)
    s = re.sub(r'\[(.*?)\]', "", s)
    s = re.sub(r'\s+', " ", s)
    s = re.sub(r'[0-9]+', " ", s)
    lis = []
    header = []
    file = open(f'C:/Users/sai/Desktop/data/{genre}.csv', 'a', encoding="utf-8", newline='')
    with file:
        writer = csv.writer(file)
        lis.append(song)
        lis.append(s)
        writer.writerow(lis)





