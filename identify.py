import requests

def identify_piece(audio_path, api_key):
    with open(audio_path, 'rb') as f:
        result = requests.post(
            'https://api.audd.io/',
            data={'api_token': api_key, 'return': 'timecode'},
            files={'file': f}
        )
    
    data = result.json()
    print(data)
    
    if data.get('status') == 'success' and data.get('result'):
        piece = data['result']
        print(f"\nPiece identified!")
        print(f"Title: {piece.get('title')}")
        print(f"Artist: {piece.get('artist')}")
    else:
        print("\nCould not identify piece")

api_key = "325c5724964620fc81ee7a56ba971908"
audio_path = r"C:\Users\finla\Downloads\Cantata ''Jesu, der du meine Seele'', BWV. 78 - 2. Wir eilen mit schwachen (1).mp3"
identify_piece(audio_path, api_key)

import urllib.parse

search_term = urllib.parse.quote("Bach BWV 78 SATB")
print(f"\nSearch for score at:")
print(f"https://musescore.com/sheetmusic?text={search_term}")