import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def matrix():    
    df_corr = pd.read_csv('ML.csv')
    # Crear el vectorizador TF-IDF
    vectorizer = TfidfVectorizer()
    # Aplicar el vectorizador a la columna 'listed_in' para obtener la matriz de características
    tfidf_matrix = vectorizer.fit_transform(df_corr['categorias'])
    # Calcular la matriz de similitud de coseno
    cosine_sim1 = cosine_similarity(tfidf_matrix)
    return cosine_sim1