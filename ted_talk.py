# TED Talks Recommender - Streamlit App
# Save this file as `ted_talks_recommender_streamlit.py` and run:
#    streamlit run ted_talks_recommender_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
import warnings
from scipy.stats import pearsonr
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK stopwords are available
nltk.download('stopwords')

warnings.filterwarnings('ignore')

st.set_page_config(page_title='TED Talks Recommender', layout='wide')

# ----------------------- Helpers -----------------------
@st.cache_data
def load_data(uploaded_file, local_path=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if local_path is None:
            st.error('No dataset provided. Upload a CSV or provide a local path in the code.')
            return None
        df = pd.read_csv(local_path)
    return df

@st.cache_data
def preprocess_df(df_raw):
    df = df_raw.copy()
    # Try to parse 'posted' column into year/month if present
    if 'posted' in df.columns:
        try:
            splitted = df['posted'].astype(str).str.split(' ', expand=True)
            df['year'] = splitted[2].astype('int')
            df['month'] = splitted[1]
        except Exception:
            pass

    # create details column if possible
    if 'title' in df.columns and 'details' in df.columns:
        df['details'] = df['title'].fillna('') + ' ' + df['details'].fillna('')
    elif 'title' in df.columns and 'details' not in df.columns:
        df['details'] = df['title'].fillna('')
    elif 'details' in df.columns:
        df['details'] = df['details'].fillna('')
    else:
        st.error('Dataset needs at least a `details` or `title` column')
        return None

    # Keep only useful columns
    keep_cols = []
    if 'main_speaker' in df.columns:
        keep_cols.append('main_speaker')
    keep_cols.append('details')
    df = df[keep_cols].dropna()

    # text cleaning
    stop_words = set(stopwords.words('english'))

    def remove_stopwords(text):
        return ' '.join([w.lower() for w in str(text).split() if w.lower() not in stop_words])

    def cleaning_punctuations(text):
        signal = str.maketrans('', '', string.punctuation)
        return str(text).translate(signal)

    df['details'] = df['details'].apply(remove_stopwords).apply(cleaning_punctuations)
    return df

@st.cache_data
def train_vectorizer(corpus_series):
    vectorizer = TfidfVectorizer(analyzer='word')
    vectorizer.fit(corpus_series)
    return vectorizer

def get_similarities(talk_content, vectorizer, data_df):
    if isinstance(talk_content, str):
        talk_content = [talk_content]

    talk_array1 = vectorizer.transform(talk_content).toarray()
    sim, pea = [], []

    for idx, row in data_df.iterrows():
        talk_array2 = vectorizer.transform([row['details']]).toarray()

        cos_sim = cosine_similarity(talk_array1, talk_array2)[0][0]

        try:
            pea_sim = pearsonr(talk_array1.squeeze(), talk_array2.squeeze())[0]
        except Exception:
            pea_sim = 0

        sim.append(cos_sim)
        pea.append(pea_sim)

    return sim, pea

# ----------------------- UI -----------------------
st.title('🎤 TED Talks Recommender')
st.markdown('Type a short description or paste a paragraph and the app will return similar TED talks from your dataset.')

with st.sidebar:
    st.header('Data')
    uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
    st.write('Or place your CSV path into the code and reload the app to use local file fallback.')
    n_recs = st.slider('Number of recommendations', 1, 10, 5)
    show_wordcloud = st.checkbox('Show wordcloud', value=True)
    show_year_plot = st.checkbox('Show year distribution (if available)', value=True)

# load data
local_default_path = r'C:\Users\ASUS\Desktop\Ted Talks Recommendation System\tedx_datase.csv'
df_raw = load_data(uploaded_file, local_default_path)
if df_raw is None:
    st.stop()

st.write('Data preview:')
st.dataframe(df_raw.head())

# preprocess
with st.spinner('Preprocessing dataset...'):
    df = preprocess_df(df_raw)
    if df is None:
        st.stop()

# visualizations
if show_year_plot and 'year' in df_raw.columns:
    plt.figure(figsize=(8, 4))
    df_raw['posted'].astype(str).str.split(' ', expand=True)[2].astype(float).value_counts().sort_index().plot.bar()
    plt.title('Talks per Year')
    plt.tight_layout()
    st.pyplot(plt)

if show_wordcloud:
    details_corpus = ' '.join(df['details'].tolist())
    wc = WordCloud(max_words=500, width=800, height=400).generate(details_corpus)
    plt.figure(figsize=(10, 4))
    plt.imshow(wc)
    plt.axis('off')
    st.pyplot(plt)

# train vectorizer
vectorizer = train_vectorizer(df['details'])

# recommendation box
st.subheader('Get recommendations')
user_input = st.text_area('Enter talk description / keywords', height=150, placeholder='e.g. Climate change and its impact on health...')
if st.button('Recommend'):
    if not user_input or str(user_input).strip() == '':
        st.warning('Please enter a description to get recommendations.')
    else:
        with st.spinner('Computing similarities...'):
            cos_sim, pea_sim = get_similarities(user_input, vectorizer, df)
            results = df.copy()
            results['cos_sim'] = cos_sim
            results['pea_sim'] = pea_sim
            results.sort_values(by=['cos_sim', 'pea_sim'], ascending=[False, False], inplace=True)

            st.success('Top recommendations:')
            display_cols = ['main_speaker', 'details'] if 'main_speaker' in results.columns else ['details']
            st.table(results[display_cols].head(n_recs).reset_index(drop=True))

            st.markdown('---')
            st.write('Full top 10 with similarity scores:')
            st.dataframe(results[['cos_sim', 'pea_sim'] + display_cols].head(10).reset_index(drop=True))

st.markdown('\n---\n')
st.write('App built with Streamlit. If you want a Flask or React version, tell me and I will provide that.')

# Footer / instructions
st.caption('Tip: For best results, provide a dataset with `title`, `details` and `main_speaker` columns. Preprocessing removes common English stopwords.')
