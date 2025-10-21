import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import warnings
from scipy.stats import pearsonr
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\ASUS\Desktop\Ted Talks Recommendation System\tedx_datase.csv')
print(df.head())

splitted = df['posted'].str.split(' ', expand=True)
df['year'] = splitted[2].astype('int')
df['month'] = splitted[1]
df['year'].value_counts().plot.bar()
plt.show()

df['details'] = df['title'] + ' ' + df['details']

df = df[['main_speaker', 'details']]
df.dropna(inplace=True)

data = df.copy()

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = [word.lower() for word in str(text).split() if word.lower() not in stop_words]
    return " ".join(imp_words)

df['details'] = df['details'].apply(remove_stopwords)

def cleaning_punctuations(text):
    signal = str.maketrans('', '', string.punctuation)
    return text.translate(signal)

df['details'] = df['details'].apply(cleaning_punctuations)

details_corpus = " ".join(df['details'])
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=1000, width=800, height=400).generate(details_corpus)
plt.axis('off')
plt.imshow(wc)
plt.show()

vectorizer = TfidfVectorizer(analyzer='word')
vectorizer.fit(df['details'])

def get_similarities(talk_content, data=df):
    if isinstance(talk_content, str):
        talk_content = [talk_content]

    talk_array1 = vectorizer.transform(talk_content).toarray()
    sim, pea = [], []

    for idx, row in data.iterrows():
        talk_array2 = vectorizer.transform([row['details']]).toarray()

        cos_sim = cosine_similarity(talk_array1, talk_array2)[0][0]

        try:
            pea_sim = pearsonr(talk_array1.squeeze(), talk_array2.squeeze())[0]
        except Exception:
            pea_sim = 0

        sim.append(cos_sim)
        pea.append(pea_sim)

    return sim, pea

def recommend_talks(talk_content, data=data):
    data = data.copy()
    data['cos_sim'], data['pea_sim'] = get_similarities(talk_content)
    data.sort_values(by=['cos_sim', 'pea_sim'], ascending=[False, False], inplace=True)

    print("\nRecommended TED Talks:\n")
    print(data[['main_speaker', 'details']].head())

talk_content = "Time Management and working hard to become successful in life"
recommend_talks(talk_content)

talk_content = "Climate change and impact on the health. How can we change this world by reducing carbon footprints?"
recommend_talks(talk_content)
