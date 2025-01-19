import re
import nltk
from unidecode import unidecode
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import spacy


nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# Pré-processamento de textos em português
nlp = spacy.load('pt_core_news_sm')
def pre_processamento_texto(texto, stemming=False):
    texto = re.sub(r'<.*?>', '', texto) # Remove tags HTML
    texto = unidecode(texto) # Remove acentuação preservando a letra
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto) # Remove caracteres especiais
    texto = re.sub(r'http\S+', '', texto) # Remove URLs
    texto = re.sub(r'\s+', ' ', texto) # Remove espaços em branco extras
    texto = texto.strip() # Remove espaços em branco no início e no final
    texto = texto.lower() # Converte para minúsculas
    texto = re.sub(r'\d+', '', texto) # Remove números
    tokens = word_tokenize(texto) # converte o texto para tokens de palavras
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]
    
    if stemming:
        stemmer = RSLPStemmer()
        tokens_sl = [stemmer.stem(token) for token in tokens]
    else:
        tokens_sl = [token.lemma_ for token in nlp(' '.join(tokens))]

    return " ".join(tokens_sl)

# Pré-processamento de textos em inglês
def text_preprocessing(texto, stemming=False):
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.strip()
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    tokens = word_tokenize(texto)
    stop_words = set(stopwords.words('english')) # english
    tokens = [token for token in tokens if token not in stop_words]
    
    if stemming:
        stemmer = PorterStemmer()
        tokens_sl = [stemmer.stem(token) for token in tokens]
    else:
        lemmatizer = WordNetLemmatizer()
        tokens_sl = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens_sl)
