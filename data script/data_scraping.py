import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd

def extract_paragraphs_to_csv(url, output_csv):
    try:
        # Ambil konten dari URL
        response = requests.get(url)
        response.raise_for_status()  # Raise error jika ada masalah pada HTTP

        # Parsing HTML dengan BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Ambil semua tag <p>
        paragraphs = soup.find_all('p')

        # Ekstrak teks dari tiap paragraf dan buang paragraf kosong
        text_paragraphs = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]

        # Simpan ke CSV (1 paragraf per baris)
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for para in text_paragraphs:
                writer.writerow([para])
    
        print(f"Berhasil mengekstrak {len(text_paragraphs)} paragraf dan menyimpannya ke {output_csv}")
    
    except Exception as e:
        print(f"Gagal memproses URL: {e}")


df = pd.read_csv('../data/berita_regulasi_id.csv')


i = 0
for link in df['Link']:
    i+=1
    url = link
    output_file = "berita_{}.csv".format(i)
    extract_paragraphs_to_csv(url, output_file)
# url = "https://rakasukmaasri.com/tren-terbaru-dalam-regulasi-impor-di-indonesia-2025/"
