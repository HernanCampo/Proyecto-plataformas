# <h1 align=center> **Proyecto-plataformas individual Nº1** </h1>

# <h1 align=center>**`Data Engineering`**</h1>

## Introducción:

Mi nombre es Hernán Campodónico, estoy realizando una presentacion del proyecto que se realiza en los labs de Data Science de SoyHenry. 
Este proyecto busca situarnos en el rol de un Data Engineer con conocimientos en Machine Learning.
Para realizarlo se nos entregan 4 datasets de distintas plataformas de streaming, amazon, disney, hulu, netflix y 8 dataset que contienen el rating/score que ciertos usuarios realizaron.

## Objetivos: 

Realizar un trabajo de ETL y EDA sobre los datasets recibidos, luego generar funciones con diferentes endpoints que se consumiran en una FastAPI y luego realizar el Deploy en Render.

Transformaciones(ETL):

- Generar columna id: Cada id se compondrá de la primera letra del nombre de la plataforma, seguido del show_id ya presente en los datasets.

- Los valores nulos del campo rating deberán reemplazarse por el string “G”.

- Las fechas deberán tener el formato AAAA-mm-dd.

- Los campos de texto deberán estar en minúsculas.

- La columna duration debe convertirse en dos: duration_int y duration_type.

Desarrollo API: 

Crear 7 funciones

- Película con mayor duración según año, plataforma y tipo de duración. La función se llama get_max_duration.

- Cantidad de películas según plataforma, con un puntaje mayor a XX en determinado año. La función se llama get_score_count.

- Cantidad de películas según plataforma. La función se llama get_count_platform.

- Actor que más se repite según plataforma y año. La función se llama get_actor.

- La cantidad de contenidos/productos (todo lo disponible en streaming) que se publicó por país y año. La función se llama prod_per_county.

- La cantidad total de contenidos/productos según el rating de audiencia dado. La función debe llamarse get_contents.

- Junto al modelo de ML crear un sistema de recomendacion de peliculas llamado get_recomendation.

Deployment:

- Se debe realizar el deploymente de la FastAPI sobre alguna plataforma, en mi caso utilice render. https://dashboard.render.com/
- Esta herramienta me parecio facil, efectiva para lo solicitado. Me hubiese gustado poder entregar otras versiones de las funciones ya que me veia limitado por el espacio que ofrece render de manera gratuita.


## Contenidos del Repositorio:

+ En el notebook `ETL.ipynb` se encuentra el código comentado paso por paso, explicando las decisiones tomadas a la hora de encarar este proyecto;

+ En el notebook `EDA.ipynb` se encuentra un pequeño analisis del dataset generado previamente por el ETL. En el mismo vemos outliers, posibles registros mal cargados y registros anomalos.

+ En el archivo `main.py` se encuentran todas las funciones configuradas de la API listas para hacer que se instancien los decoradores de la API para luego hacer el deploy.

+ En el notebook `ML(coseno).ipynb` se encuentra un pequeño ETL y EDA extra para poder configurar mi sistema de recomendacion de la manera mas optima. 

+ En los archivos CSV `plataformas.csv` y `ML.csv` se encuentran los DataSet trabajados o Dataset finales que se usan para la implementacion de las funciones de la API.

+ En el notebook `FuncionesApi.ipynb` se encuentra una copia de las funciones de API solo para poder realizar pruebas de manera mas rapida.

+ El archivo `requirements.txt` en el cual se encuentra las librerias utilizadas para el deploy.

+ El archivo `.gitignore` se creo para cuando realice la carga de archivos en GitHub no me cargue todos los archivos si no solo los que quiero.

## Herramientas utilizadas:

+ VSC.

+ Python.

+ Render.

+ FastAPI.

## Librerias utilizadas:

- pandas - matplotlib - seaborn - uvicorn - scikit-learn (TfidfVectorizer - cosine_similarity) 

## Links:

+ Drive con los dataset: https://drive.google.com/drive/u/0/folders/1T02IMQyVbaf2iPDh_gmwh885s1RNNgIQ

+ Video Explicativo de la API: 

+ Deploy de la API en Render: https://fastapi-qljo.onrender.com/docs