import re
import string
import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TextPreprocessor:
    def __init__(self, language='indonesian'):

        self.language = language
        
        # Initialize Sastrawi untuk bahasa Indonesia
        if language == 'indonesian':
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            
            stop_factory = StopWordRemoverFactory()
            self.stopword_remover = stop_factory.create_stop_word_remover()
            
            # Stopwords bahasa Indonesia
            self.stopwords = stop_factory.get_stop_words()
        else:
            # Untuk bahasa Inggris (opsional)
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer
            import nltk
            
            try:
                nltk.download('stopwords', quiet=True)
                self.stopwords = set(stopwords.words('english'))
                self.stemmer = PorterStemmer()
            except:
                self.stopwords = set()
                self.stemmer = None
    
    def basic_cleaning(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text
    
    def remove_urls_emails(self, text: str) -> str:
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text
    
    def remove_numbers_symbols(self, text: str, keep_punctuation: bool = True) -> str:
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        if not keep_punctuation:
            # Remove all punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
        else:
            # Keep basic punctuation (.,!?) and remove others
            text = re.sub(r'[^\w\s.,!?]', '', text)
        
        return text
    
    def normalize_repeated_chars(self, text: str) -> str:
        # Normalize repeated characters (more than 2 occurrences)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        if self.language == 'indonesian':
            # Menggunakan Sastrawi
            text = self.stopword_remover.remove(text)
        else:
            # Untuk bahasa Inggris
            words = text.split()
            text = ' '.join([word for word in words if word.lower() not in self.stopwords])
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        return text.split()
    
    def stemming(self, text: str) -> str:
        if self.language == 'indonesian' and self.stemmer:
            return self.stemmer.stem(text)
        elif self.language == 'english' and self.stemmer:
            words = text.split()
            stemmed_words = [self.stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
        
        return text
    
    def preprocess_for_bert(self, text: str, 
                          remove_stopwords: bool = False,
                          apply_stemming: bool = False,
                          max_length: int = None) -> str:
        # Basic cleaning
        text = self.basic_cleaning(text)
        
        # Remove URLs and emails
        text = self.remove_urls_emails(text)
        
        # Normalize repeated characters
        text = self.normalize_repeated_chars(text)
        
        # Remove numbers and excessive symbols (keep basic punctuation for BERT)
        text = self.remove_numbers_symbols(text, keep_punctuation=True)
        
        # Remove stopwords (opsional untuk BERT)
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Stemming (opsional untuk BERT)
        if apply_stemming:
            text = self.stemming(text)
        
        # Truncate if too long
        if max_length:
            words = text.split()
            if len(words) > max_length:
                text = ' '.join(words[:max_length])
        
        # Final cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def batch_preprocess(self, texts: List[str], **kwargs) -> List[str]:
        return [self.preprocess_for_bert(text, **kwargs) for text in texts]
    
    def compute_tfidf(self, texts: List[str], max_features: int = 5000) -> tuple:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # unigram dan bigram
            min_df=2,  # minimal muncul di 2 dokumen
            max_df=0.95  # maksimal muncul di 95% dokumen
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        return tfidf_matrix, feature_names, vectorizer
    
    def get_text_statistics(self, texts: List[str]) -> dict:

        if not texts:
            return {}
        
        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]
        
        stats = {
            'total_texts': len(texts),
            'avg_words': np.mean(word_counts),
            'max_words': np.max(word_counts),
            'min_words': np.min(word_counts),
            'avg_chars': np.mean(char_counts),
            'max_chars': np.max(char_counts),
            'min_chars': np.min(char_counts)
        }
        
        return stats

# Contoh penggunaan
def example_usage():
    # Inisialisasi preprocessor
    preprocessor = TextPreprocessor(language='indonesian')
    
    sample_texts = []

    df = pd.read_csv('news_dataset1.csv')  # Jika tidak ada header
    # Jika ada header, hapus parameter `header=None`

    # Ambil semua isi dari kolom pertama
    
    for txt in df['description']:
        sample_texts.append(txt)


    # Cetak hasilnya
    print("=== Original Texts ===")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\n=== Preprocessed Texts (untuk BERT) ===")
    # Preprocessing untuk BERT (tanpa stemming dan stopword removal)
    processed_texts = preprocessor.batch_preprocess(
        sample_texts,
        remove_stopwords=False,  # Biasanya tidak perlu untuk BERT
        apply_stemming=False,    # Biasanya tidak perlu untuk BERT
        max_length=128           # Sesuaikan dengan max_length BERT
    )
    
    for i, text in enumerate(processed_texts, 1):
        print(f"{i}. {text}")
    
    print("\n=== Text Statistics ===")
    stats = preprocessor.get_text_statistics(processed_texts)
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    print("\n=== TF-IDF Analysis ===")
    try:
        tfidf_matrix, feature_names, vectorizer = preprocessor.compute_tfidf(processed_texts)
        print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
        print(f"Top Features: {feature_names[:30]}")
    except:
        print('error')
    
    
    return processed_texts, preprocessor

# Fungsi utility untuk DataFrame
def preprocess_dataframe(df: pd.DataFrame, 
                        text_column: str,
                        language: str = 'indonesian',
                        **preprocess_kwargs) -> pd.DataFrame:
    preprocessor = TextPreprocessor(language=language)
    
    # Copy dataframe
    df_processed = df.copy()
    
    # Preprocess text column
    df_processed[f'{text_column}_processed'] = preprocessor.batch_preprocess(
        df[text_column].fillna('').tolist(),
        **preprocess_kwargs
    )
    
    return df_processed

if __name__ == "__main__":
    # Jalankan contoh penggunaan
    processed_texts, preprocessor = example_usage()