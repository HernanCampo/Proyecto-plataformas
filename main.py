#Importacion de las librerias que voy a utilizar.
import pandas as pd
from fastapi import FastAPI
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Leo el Dataframe que voy a utilizar en las funciones
df = pd.read_csv('plataformas.csv',usecols=['id', 'type', 'title', 'cast', 'country','release_year', 'rating','listed_in', 'duration_int', 'duration_type', 'score'])
df_corr = pd.read_csv('ML.csv')

app = FastAPI()

@app.get('/get_max_duration/{anio}/{plataforma}/{dtype}')
def get_max_duration(anio: int, plataforma: str, dtype: str):
    
    # Me aseguro que si el usuario escribe en mayusculas se pase a minusculas.
    dtype = dtype.lower()
    platform = plataforma.lower()[0] #Aclaro indice en 0 para que tome la primer letra

    # Filtro por año de lanzamiento, plataforma aclarando que solo tenga en cuenta la primer letra de la columna id y que el tipo sea 'movie'.
    data_filtrada = df[(df['release_year'] == anio) & (df['id'].str.startswith(platform)) & (df['duration_type'] == dtype) & (df['type'] == 'movie')]
    
    # Busco la duración máxima dentro del conjunto de datos filtrados
    max_duration = data_filtrada['duration_int'].max()
    # Busco la fila que contiene la duración máxima dentro del conjunto de datos filtrados
    max_duration_row = data_filtrada.loc[data_filtrada['duration_int'] == max_duration]    
       
    if not max_duration_row.empty:
        # Si se encontró una fila con la duración máxima, se obtiene el nombre de la
        # película correspondiente (es decir, la película con la duración máxima)
        # extrayendo el valor de la columna 'title' de la fila.
        movie_name = max_duration_row['title'].iloc[0]
        return {'pelicula': movie_name}
    else:
        return {'No se encontraron peliculas que cumplan con los criterios de busqueda.'}
        

@app.get('/get_score_count/{plataforma}/{scored}/{anio}')
def get_score_count(plataforma: str, scored: float, year: int):
      
    # Me aseguro que si el usuario escribe en mayus se pase a minusculas, en el caso de plataform solo me interesa la primer letra.
    platform = plataforma.lower()[0]#Aclaro indice en 0 para que tome la primer letra
        
    # Filtro por año de lanzamiento, plataforma aclarando que solo tenga en cuenta la primer letra de la columna id y que el tipo sea 'movie'.   
    data_filtrada = df[(df['id'].str.startswith(platform)) & (df['release_year'] == year) & (df['type'] == 'movie')]
    
    # Cuento la cantidad de películas que tienen el rating deseado
    count = data_filtrada[data_filtrada['score'] > scored]['id'].count()
    
    return {'plataforma': plataforma,
            'cantidad': int(count),
            'anio': year,
            'score': scored}
        
    
@app.get('/get_count_platform/{plataforma}')
def get_count_platform(plataforma: str):
        
    # Me aseguro que si el usuario escribe en mayus se pase a minusculas, en el caso de plataforma solo me interesa la primer letra.
    platform = plataforma.lower()[0]#Aclaro indice en 0 para que tome la primer letra
    
    # Realizo el filtro para que cuente del dataframe solo la plataforma de la columna id y movie de la columna type.
    data_filtrada = df[(df['id'].str.startswith(platform)) & (df['type'] == 'movie')].shape[0]    

    return {'plataforma': plataforma,
            'peliculas': data_filtrada}
    
    
@app.get('/get_actor/{plataforma}/{anio}')
def get_actor(plataforma: str, anio: int):    
    
    # Me aseguro que si el usuario escribe en mayus se pase a minusculas, en el caso de plataform solo me interesa la primer letra.
    platform = plataforma.lower()[0]#Aclaro indice en 0 para que tome la primer letra
    
    # Realizo el filtro para que cuente del dataframe solo la plataforma de la columna id y el añio.
    data_filtrada = df[(df['id'].str.startswith(platform)) & (df['release_year'] == anio)]   
 
    # En este paso, se utiliza la función dropna() de pandas para eliminar los valores NaN de la columna "cast" del DataFrame. 
    # Luego utilizo el método apply() para aplicar la función str() a cada valor de la columna "cast", convirtiendo los valores a cadenas de caracteres. 
    # Finalmente, se utiliza el método str.split(',') para dividir cada cadena de caracteres en una lista de actores utilizando la coma como separador
    # y un .strip para que extraiga cualquier valor que sea tipo digito(No enteros si no un .digit). 
    # El resultado es una serie de pandas que contiene una lista de actores para cada fila del DataFrame original.
    actores_por_fila = data_filtrada['cast'].dropna().apply(lambda x: [i.strip() for i in x.split(',') if not i.strip().isdigit()])


    # Cuento la cantidad de veces que aparece cada actor en todas las filas, utilizando la clase Counter de Python.
    contador_actores = Counter()
    for actores in actores_por_fila:
        # Actualizamos el contador de actores con un nuevo conjunto de actores.
        # Si el contador ya tenía un registro para un actor específico, se incrementa su
        # valor en 1. De lo contrario, se crea un nuevo registro para ese actor con un
        # valor inicial de 1.
        contador_actores.update(actores)

    # Encuentro el actor que aparece más veces utilizando la funcion most common devolviendo una lista de tuplas donde cada tupla contiene un actor 
    # y la cantidad de veces que aparece en todas las filas del DataFrame.
    actor_mas_repetido = contador_actores.most_common(1)# El 1 indica que se muestre 1 solo valor.
    
    if actor_mas_repetido:
        # se asigna [0][0] para indicar el actor que mas veces aparece
        actor_mas_repetido = actor_mas_repetido[0][0]
    else:
        return {'plataforma': plataforma,
                'anio': anio,
                'actor': "No hay datos disponibles",
                'apariciones': "No hay datos disponibles"}

    # Muestro el actor que aparece más veces y la cantidad de veces que aparece
    cantidad_actor_mas_repetido = contador_actores[actor_mas_repetido]
    return {'plataforma': plataforma,
            'anio': anio,
            'actor': actor_mas_repetido,
            'apariciones': cantidad_actor_mas_repetido}
    
    
@app.get('/prod_per_county/{tipo}/{pais}/{anio}')
def prod_per_county(tipo: str, pais: str, anio: int):   
    
    # Realizo un filtro para que busque por tipo, año de lanzamiento y que el campo "country" contenga el país.
    data_filtrada = df[(df['type'] == tipo) & (df['release_year'] == anio) & (df['country'].str.contains(pais))]
    
    # Separar los países en una lista y contarlos individualmente
    count_por_pais = {}
    for countries in data_filtrada['country']:
        for country in countries.split(','):
            if country.strip() not in count_por_pais:
                count_por_pais[country.strip()] = 0
            count_por_pais[country.strip()] += 1
    
    # Obtener el recuento de películas para el país especificado
    respuesta = count_por_pais.get(pais, 0)
    
    return {'pais': pais,
            'anio': anio,            
            'peliculas': respuesta}
    
        
@app.get('/get_contents/{rating}')   
def get_contents(rating: str):       
    
    #Realizo un .value_counts de rating en base al rating que pongan.    
    respuesta =  df['rating'].value_counts()[rating]

    return {'rating': rating,
            'contenido': int(respuesta)}

@app.get('/get_recommendationA/{title}')
def get_recommendationA(title: str):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['title'])
    indices = pd.Series(df.index, index=df['title'])
    idx = indices[title]

    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    respuesta = df['title'].iloc[movie_indices].tolist()
    return {'recomendacion': respuesta}


@app.get('/get_recommendationB/{title}')
def get_recommendationB(title: str):    
    # Crear el vectorizador TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_corr['categorias'])
    
    # Obtener el índice de la película que coincide con el título proporcionado
    idx = df_corr.index[df_corr['title'] == title.lower()].tolist()[0]
    
    # Calcular la matriz de similitud de coseno
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    
    # Obtener las puntuaciones de similitud de coseno de la película con todas las demás películas
    sim_scores = list(enumerate(cosine_sim[0]))
    
    # Ordenar las películas según las puntuaciones de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obtener las cinco películas más similares, excluyendo la película de entrada
    sim_scores = [i for i in sim_scores if i[0] != idx]
    sim_scores = sim_scores[:5]
    
    # Obtener los títulos de las cinco películas más similares
    recomendaciones = df_corr.iloc[[i[0] for i in sim_scores]]['title'].tolist()
    
    # Devolver las recomendaciones en el formato especificado
    return {'recomendacion': recomendaciones} 