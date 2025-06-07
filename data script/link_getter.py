import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import quote

def scrape_google_news(query, num_results=10, lang="en"):

    query = quote(query)
    
    url = f"https://www.google.com/search?q={query}&tbm=nws&hl={lang}"
    
    # Headers untuk menghindari pemblokiran oleh Google
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": f"{lang},en-US;q=0.9,en;q=0.8"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML response dengan BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Mencari semua elemen berita
        news_results = []
        news_divs = soup.find_all("div", class_="SoaBEf")
        
        for div in news_divs[:num_results]:
            try:
                # Mengekstrak judul, link, sumber, dan deskripsi
                title_element = div.find("div", class_="n0jPhd")
                title = title_element.text.strip() if title_element else "Tidak ada judul"
                
                link_element = div.find("a")
                link = link_element["href"] if link_element else ""
                
                # publish_element = div.find("div", class_="OSrXXb")
                # time_element = publish_element.find("span")
                # time = time_element.text.strip() if time_element else "Tidak ada waktu"
                
                description_element = div.find("div", class_="GI74Re")
                description = description_element.text.strip() if description_element else "Tidak ada deskripsi"
                
                # Mencari dan mengekstrak waktu publikasi
                # source_element = div.find("span")
                # source = source_element.text.strip() if source else "Tidak ada sumber"
                
                news_results.append({
                    "Judul": title,
                    "Deskripsi": description,
                    "Link": link
                })
            except Exception as e:
                print(f"Error saat mengekstrak data: {e}")
        
        # Membuat DataFrame dari hasil
        df = pd.DataFrame(news_results)
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error saat melakukan request: {e}")
        return pd.DataFrame()

def main():
    print("===== PENCARI BERITA DARI GOOGLE =====")
    while True:
        # Menerima input dari pengguna
        query = input("\nMasukkan kata kunci pencarian berita (ketik 'keluar' untuk berhenti): ")
        
        if query.lower() == "keluar":
            print("Terima kasih telah menggunakan program ini.")
            break
        
        try:
            num_results = int(input("Jumlah hasil yang diinginkan (default: 10): ") or 10)
        except ValueError:
            num_results = 10
            print("Input tidak valid, menggunakan nilai default: 10")
        
        lang = input("Bahasa hasil pencarian (id/en, default: id): ") or "id"
        
        print(f"\nMencari berita tentang '{query}'...")
        
        # Melakukan scraping
        start_time = time.time()
        results = scrape_google_news(query, num_results, lang)
        end_time = time.time()
        
        # Menampilkan hasil
        if not results.empty:
            print(f"\nBerhasil menemukan {len(results)} berita ({end_time - start_time:.2f} detik)")
            
            # Menampilkan hasil di konsol dengan format yang rapi
            for i, row in results.iterrows():
                print(f"\n{i+1}. {row['Judul']}")
                print(f"   {row['Deskripsi']}")
                print(f"   Link: {row['Link']}")
            
            # Menyimpan hasil ke file CSV
            save_option = input("\nApakah Anda ingin menyimpan hasil pencarian ke file CSV? (y/n): ")
            if save_option.lower() == 'y':
                # Menggunakan re.sub di luar f-string untuk menghindari masalah dengan backslash
                clean_query = re.sub(r'[^\w]', '_', query)
                filename = f"berita_{clean_query}_{lang}.csv"
                results.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"Hasil berhasil disimpan ke file {filename}")
        else:
            print("Tidak ada hasil yang ditemukan atau terjadi error.")

if __name__ == "__main__":
    main()