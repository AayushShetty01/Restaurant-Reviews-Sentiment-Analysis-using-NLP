import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['cleaned_review'] = df['review'].apply(clean_text)
    return df
