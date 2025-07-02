"""


TFG Matemáticas - Machine Learning

@author: talarza
"""
#Importamos pandas, biblioteca para manipulación de datos
import pandas as pd

# Cargar el archivo Excel
file_path = "C:/Users/talarza/OneDrive - Universidad Rey Juan Carlos/TFG Mates/1-s2.0-S0302283824024084-mmc2.xlsx"
excel = pd.ExcelFile(file_path)

# Mostrar las hojas disponibles en el archivo
excel.sheet_names

# Cargar los datos de la hoja "Table S6"
tabla_s6 = excel.parse("Table S6")

# Mostrar las primeras filas para entender la estructura
tabla_s6.head()

#Limpiamos los datos
# Extraer los nombres de las columnas correctos (fila con índice 2) ya que los demas son metadatos
columnas = tabla_s6.iloc[2, :].values

# Extraer los datos eliminando las primeras tres filas y reiniciando el índice
tabla_s6_cleaned = tabla_s6.iloc[3:, :].reset_index(drop=True)

# Asignar los nombres de las columnas correctos
tabla_s6_cleaned.columns = columnas

# Eliminar la columna 'Gene' si existe y ponerla como índice
if 'Gene' in tabla_s6_cleaned.columns:
    tabla_s6_cleaned = tabla_s6_cleaned.set_index('Gene')

# Convertir a valores numéricos (omitimos 'Annotation' completamente)
df_genes = tabla_s6_cleaned.apply(pd.to_numeric, errors='coerce')
df_genes = tabla_s6_cleaned.drop(columns=['Annotation'], errors='ignore')
df_genes.to_excel("C:/Users/talarza/OneDrive - Universidad Rey Juan Carlos/TFG Mates/tabla_genes_cleaned.xlsx")


#Aplicar K-Means con Diferentes Valores de Clusters
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# Definir el rango de clusters a probar (de 2 a 10)
k_values = range(2, 11)

#Para medir como de bien agrupados están segun el metodo del codo
inertia_values = []

#Graficar clusters
fig, axes = plt.subplots(3, 3, figsize=(18, 12)) 

#Para medir como de bien agrupados están segun el Silhouette Score
silhouette_scores = [] 

# Crear un DataFrame para almacenar los clusters de cada paciente en diferentes valores de k
df_all_clusters = pd.DataFrame(index=df_genes.columns)

#En principio no es necesario escalar los datos al estar en el mismo rango (expresados en log2)
# Aplicar K-Means para distintos valores de k

'''
K-MEANS

1. Se eligen k puntos aleatorios como centroides iniciales
2. Asignación de clusters:
    - Para cada punto, se calcula la distancia de cada centroide
    - Se asigna el punto al cluster cuyo centroide está más cerca.
3. Reajuste de Centroides:
    - Cuando todos han sido asignados, se recalcula la posición de cada centroide como el promedio de los puntos que pertenece a ese cluster.
4. Repetición del proceso
    - Se repite 2 y 3 hasta que los centroides dejan de moverse o hasta alcanzar un numero maximo de interaciones.
    
DISTANCIAS
El algoritmo usa la distancia euclidiana entre cada paciente (columnas) y cada centroide del cluster.
    - Fila: representan genes
    - Columna: representan pacientes
    - Utilizaremos K-Means para agrupar los pacientes en cluster segun sus genes
    
'''

for i, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #random_state -> Fija los centroides iniciales para obtener siempre los mismos clusters
    #n_init=10 -> el algoritmo ejecuta k-means 10 veces con diferentes centroides iniciales (valor recomendado por buen equilibrio entre velocidad y estabilidad)
    #Transponemos para agrupar pacientes porque queremos ver la distancia entre pacientes
    clusters = kmeans.fit_predict(df_genes.T) 
    
    
    # Agregar los clusters a la tabla con el número de k como columna
    df_all_clusters[f"K={k}"] = clusters
    df_all_clusters.to_excel("C:/Users/talarza/OneDrive - Universidad Rey Juan Carlos/TFG Mates/tabla_clustering_pacientes.xlsx")
    
    #Metrica de inercia para el Metodo del codo
    inertia_values.append(kmeans.inertia_)
    
    '''
    VISUALIZAR DISTRIBUCION DE LOS CLUSTERS
    1. Reducir la dimensionalidad con PCA (Análisis de COmponentes Principales) para poder graficarlo en 2D
        PCA crea nuevas variables (PC1 y PC2) que combinan la informacion entre los genes
    2. Graficar en 2D
    3. Colorear los clusters
    '''

    
    #aplicar PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_genes.T)
    
    # Crear DataFrame con los resultados
    df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=df_genes.columns)
    df_pca["Cluster"] = clusters #Agregar informacion de los clusters
    
    '''
    METRICA DE SILHOUETTE
    - Calcula como de separados está los clusters
    - Valor cercano a 1: buenos
    - Valor cercano a 0: solapamiento
    
    1.Calcular cohesion a(i): distancia promediao entre i y todos los demas puntos en su mismo cluster
     a(i) = 1/(|c|-1)* sumatorio d(i,j)con jeC y j!=i, donde:
         - C es el cluster al que pertenece i
         - d(i,j) es la distancia euclidiana entre i y j
    2. Calcula la separación b(i): distancia promedio entre i y todos los puntos del cluster mas cercano C':
     b(i)= min 1/|C'|* sumatorio  d(i,j)con jeC' y C'!=C, donde:
         - Se mide la distancia del punto i a todos los clusters distintos al suyo
         - Se selecciona el cluster mas cercano C'
    3. Silhouette Score para i:
        s(i)=[(b(i)-a(i)] / [max(a(i),b(i))]
        - s(i) cercano a 1: bien agrupado
        - s(i) cercano a 0: limite entre clusters
        - s(i) negativo: deberia estara en otro cluster
    4. Silhouetter Score Global: promedio de s(i) para todos los puntos
        S = 1/N * sumatorio (s(i)) de i=1 hasta N
        - N es el numero total de puntos del dataset
        - Mayor S, mayor calidad del clustering
    '''

    # Calcular Silhouette Score
    silhouette_avg = silhouette_score(df_genes.T, clusters)
    silhouette_scores.append((k, silhouette_avg)) #guardar metrica
    
    # Posicionar cada gráfico en la cuadrícula de 3x3
    row, col = divmod(i, 3)  # Calculamos fila y columna para la cuadrícula
    ax = axes[row, col]  

    # Graficar
    #plt.subplot(3, 3, i)
    for cluster in df_pca["Cluster"].unique(): #recorremos cada cluster y graficamos los pacientes de él
        subset = df_pca[df_pca["Cluster"] == cluster]
        ax.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster}", alpha=0.7) #dibujamos los puntos (pacientes)
    
    '''
    REPRESENTACION DE LOS EJES
    EJE X (PC1):
        - Representa la direccion de mayor variabilidad de los datos
        - Cuanto mas alejados esten dos puntos en el eje X, mas diferentes en cuanto a expresion genetica
    EJE Y (PC2):
        - Segunda direccion de mayor variabilidad
        
    '''
    
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.set_title(f"K-Means con k={k} (Silhouette={silhouette_avg:.2f})", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True)

plt.tight_layout(pad=3.0)
plt.show()

# Mostrar los Silhouette Scores
silhouette_scores





"""
METODO DEL CODO

Inercia en K-Means: métrica que mide qué tan bien agrupados están los datos dentro de sus clusters.
Matemáticamente, la inercia se calcula como:
Sumatorio desde i=0 hasta i=n de la distancia al cuadrado entre el punto_i y el centro del cluster.

Donde:
- Cada punto del cluster se compara con su centroide.
- La distancia se eleva al cuadrado para evitar valores negativos.
- La suma total representa la dispersión de los puntos en sus respectivos clusters.

Interpretaciones:
- Inercia baja-> significa que los puntos están muy bien agrupados.
- Inercia alta-> los puntos están más dispersos y el clustering es menos efectivo.
Queremos minimizar la inercia, pero sin sobreajustar el modelo.

El Método del Codo es una técnica para encontrar el número óptimo de clusters en K-Means.
1. Calculamos la inercia para distintos valores de k (número de clusters).
2. Graficamos la inercia vs.k.
3. Buscamos el "codo" en la curva:
    - Al principio, la inercia baja mucho al aumentar k.
    - Luego, la reducción de inercia se hace más lenta.
    - El punto donde la curva se "dobla" es el mejor k.
"""

#Graficar el método del codo
#Añadir columnas para ver cuanto acierto en "Annotation"
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker='o', linestyle='-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo para Selección de k en K-Means')
plt.grid(True)
plt.show()

# Extraer los datos desde la columna C en adelante (índice 2 en pandas)
tabla_s6_clusters = tabla_s6.iloc[0:3, 2:].T  # Transponer para reorganizar

# Renombrar columnas
tabla_s6_clusters.columns = ["Cluster", "Score", "Patient"]

# Eliminar la fila intermedia (B-L Score)
tabla_s6_clusters = tabla_s6_clusters.drop(columns=["Score"])

# Resetear el índice para una mejor presentación
tabla_s6_clusters = tabla_s6_clusters.reset_index(drop=True)

# Crear un diccionario para mapear cada cluster a un número único
cluster_mapping = {cluster: idx for idx, cluster in enumerate(tabla_s6_clusters["Cluster"].unique(), start=0)}

# Asignar los números a la columna Cluster
tabla_s6_clusters["Cluster_Estudio"] = tabla_s6_clusters["Cluster"].map(cluster_mapping)

# Pasar a excel
tabla_s6_clusters.to_excel("C:/Users/talarza/OneDrive - Universidad Rey Juan Carlos/TFG Mates/tabla_s6_clusters_hechos.xlsx")

#Tabla con los clusters k=6
cluster_k6_ruta = "C:/Users/talarza/OneDrive - Universidad Rey Juan Carlos/TFG Mates/clusters_pacientes_k6.csv"
cluster_k6 = pd.read_csv(cluster_k6_ruta)
cluster_k6.head()
# Renombrar la columna de pacientes en la nueva tabla
cluster_k6.rename(columns={"Unnamed: 0": "Patient", "Cluster": "Cluster_k6"}, inplace=True)

# Fusionar ambas tablas en base a la columna "Patient"
fusion_clusters = tabla_s6_clusters.merge(cluster_k6, on="Patient", how="inner")

# Comparar si los clusters coinciden
fusion_clusters["Match"] = fusion_clusters["Cluster_Estudio"] == fusion_clusters["Cluster_k6"]
fusion_clusters.to_excel("C:/Users/talarza/OneDrive - Universidad Rey Juan Carlos/TFG Mates/tabla_s6_fusion.xlsx")

#Contar coincidencias y discrepancias
match_counts = fusion_clusters["Match"].value_counts()

# Mostrar los resultados
match_counts
#32 coincidencias y 66 no, hay un 33% de coincidencia

'''
Ver cómo de bueno son los clusters del estudio y comparar los resultados con los obtenidos con mis clusters.
'''

# Calcular Silhouette Score para cada método
silhouette_cluster_estudio = silhouette_score(df_genes.T, fusion_clusters["Cluster_Estudio"])
silhouette_cluster_estudio
#0.12
silhouette_cluster_k6 = silhouette_score(df_genes.T, fusion_clusters["Cluster_k6"])
silhouette_cluster_k6
#0.17 -> mejor
#k=3 tiene el mejor, 0.23

# Función para calcular el índice de Jaccard de dos particiones de clusters
def jaccard_index_manual(cluster_labels1, cluster_labels2, patients):
    """
    Calcula el índice de Jaccard entre dos agrupaciones de pacientes.
    
    cluster_labels1: Lista con los clusters del primer método.
    cluster_labels2: Lista con los clusters del segundo método.
    patients: Lista de identificadores de pacientes.
    
    Retorna el índice de Jaccard promedio sobre todos los clusters.
    """
    # Obtener los conjuntos únicos de clusters en cada método
    unique_clusters1 = set(cluster_labels1)
    unique_clusters2 = set(cluster_labels2)

    # Crear diccionarios de pacientes agrupados por cluster
    cluster_groups1 = {cluster: set() for cluster in unique_clusters1}
    cluster_groups2 = {cluster: set() for cluster in unique_clusters2}

    # Llenar los diccionarios con los pacientes asignados a cada cluster
    for patient, cluster1, cluster2 in zip(patients, cluster_labels1, cluster_labels2):
        cluster_groups1[cluster1].add(patient)
        cluster_groups2[cluster2].add(patient)

    # Calcular el índice de Jaccard para cada cluster en el primer método
    jaccard_scores = []
    for cluster1, patients1 in cluster_groups1.items():
        best_jaccard = 0  # Para almacenar la mejor coincidencia

        for cluster2, patients2 in cluster_groups2.items():
            # Calcular intersección y unión
            intersection = len(patients1 & patients2)
            union = len(patients1 | patients2)

            # Evitar divisiones por cero
            if union > 0:
                jaccard = intersection / union
                best_jaccard = max(best_jaccard, jaccard)  # Tomamos el mejor match

        jaccard_scores.append(best_jaccard)

    # Devolver el promedio de Jaccard
    return sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0

# Aplicar la función manual al dataset
jaccard_manual = jaccard_index_manual(fusion_clusters["Cluster_Estudio"], fusion_clusters["Cluster_k6"], fusion_clusters["Patient"])

# Mostrar el resultado
jaccard_manual

'''
El Índice de Jaccard calculado manualmente es 0.6463.

De promedio, hay un 64.63% de coincidencia entre los conjuntos de pacientes 
asignados a los mismos clusters en ambos métodos. 
Esto sugiere que los dos métodos de clusterización tienen una similitud moderada.

Jaccard da 0.6463 y compara los grupos de pacientes como conjuntos, no etiqueta por etiqueta. 
Esto permite que los clusters se consideren similares aunque tengan nombres diferentes.

Ejemplo:

Si un método asigna {A, B, C} al Cluster 1 y otro método los pone en Cluster 3,
la comparación con true o false los vería como diferentes.
Pero el Jaccard basado en grupos (0.6463) detecta que los mismos pacientes están juntos y lo considera más similar.
En otras palabras, 0.33 mide qué tan exactamente coinciden los nombres de los clusters, 
mientras que 0.6463 mide qué tan similares son las estructuras de agrupación.
'''


'''
Clustering Jerárquico Aglomerativo (Agglomerative Clustering)
Ventajas frente al k-means:
    - No necesita inicialización aleatoria.
    - Es interpretativo con dendrogramas.
    -Puede funcionar mejor cuando los clusters no tienen forma esférica.
    Inicialización:

1. Cada dato (paciente) empieza siendo su propio cluster.
2. Cálculo de distancias: Se calcula la distancia entre todos los clusters.
3. Fusión de clusters: Se encuentran los dos clusters más cercanos y se fusionan.
4. Actualización de distancias: Se recalculan las distancias entre el nuevo cluster y los demás.
5. Repetición: Se repiten los pasos 3 y 4 hasta que todos los puntos estén en un solo cluster, o hasta que se llegue a un número deseado de clusters.

Tipos de distancia para unir clusters:
    - Single: Mínima distancia entre puntos de dos clusters
    -Complete: Máxima distancia entre puntos de dos clusters
    -Average: Promedio de distancias entre todos los puntos
    -Ward:	Aumento mínima en la varianza total al fusionar clusters
    

'''



from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Transponer para que cada paciente sea una fila
X = df_genes.T

#Quiero probar con todos los metodos de linkage
linkage_methods = ["single", "complete", "average", "ward"]

from sklearn.decomposition import PCA

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
resultados_comparacion = []

for idx, method in enumerate(linkage_methods):
    clustering = AgglomerativeClustering(n_clusters=6, linkage=method, affinity='euclidean')
    labels = clustering.fit_predict(X)
    n_clusters_detected = len(set(labels))

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    # Cálculo de métricas
    silhouette = silhouette_score(X, labels)
    jaccard_estudio = jaccard_index_manual(fusion_clusters["Cluster_Estudio"], labels, fusion_clusters["Patient"])
    jaccard_kmeans = jaccard_index_manual(fusion_clusters["Cluster_k6"], labels, fusion_clusters["Patient"])

    resultados_comparacion.append({
        "Método": method.capitalize(),
        "Silhouette Score": round(silhouette, 4),
        "Jaccard con clusters del estudio": round(jaccard_estudio, 4),
        "Jaccard con clusters del k-means con k=6": round(jaccard_kmeans, 4)
    })

    # Graficar correctamente con el PCA que corresponde a X
    ax = axes[idx]
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap="tab10", alpha=0.8)
    ax.set_title(f"Linkage: {method.capitalize()} ({n_clusters_detected} clusters)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)

plt.tight_layout()
plt.show()


df_resultados_comparacion_agg = pd.DataFrame(resultados_comparacion)
#tools.display_dataframe_to_user(name="Comparación de Métodos de Linkage", dataframe=df_resultados)

'''
Recordar:
    -Silhoutte score k-means con k=6: 0.17
    -Silhoutte score clusters originales: 0.12
        
CONCLUSIONES Clustering Jerárquico Aglomerativo - distancia euclidiana
1. Single Linkage:
    - Single: Mínima distancia entre puntos de dos clusters
    - Tiende a formar agrupaciones poco definidas y alargadas.
    - Visualmente, la estructura no sugiere clusters bien definidos.
    - Confirmado con su Silhouette Score negativo (-0.0014) y el peor Jaccard con el estudio (0.1655).

2. Complete Linkage:
    -Complete: Máxima distancia entre puntos de dos clusters
    - Produce clusters más compactos y separados.
    - La visualización muestra separación razonable entre grupos.
    - No ofrece el mejor Silhouette Score (0.1660) de todos los métodos jerárquicos, aunque ligeramente por debajo de Average.
    - Jaccard con el estudio: 0.4981, lo que indica una similitud moderada con los grupos del estudio.

3. Average Linkage:
    -Average: Promedio de distancias entre todos los puntos
    - Comportamiento intermedio entre Single y Complete.
    - Genera clusters algo más definidos que Single, pero menos compactos que Complete.
    - Silhouette Score: 0.1664 (ligeramente superior a Complete) y Jaccard: 0.4243.

4. Ward Linkage:
    -Ward:Aumento mínima en la varianza total al fusionar clusters
    - El método más consistente con los resultados del estudio original.
    - Visualmente muestra clusters bien agrupados y balanceados.
    - Aunque su Silhouette Score (0.1348) es inferior al de Complete/Average, su Jaccard con los clusters del estudio es el mayor (0.5188).
    - También se acerca más al resultado de K-Means (Jaccard con K-Means = 0.6750).
'''
#Clusters del mejor metodo -> complete

best_method = 'complete'

agg = AgglomerativeClustering(
        n_clusters=6, linkage=best_method, affinity='euclidean'
)

labels_agg = agg.fit_predict(X)              # X = df_genes.T

cluster_agg_complete = (
    pd.DataFrame({"Patient": df_genes.columns, 
                  "Cluster_Agg": labels_agg})
)





'''
    PROBAR EFICACIA DE LOS CLUSTERS EN FARMACOS
    
	Paso 1 · Extraer la columna de Gemcitabina de “Table S11”
'''


import pandas as pd
from pathlib import Path
#file_path = "C:/Users/talarza/OneDrive - Universidad Rey Juan Carlos/TFG Mates/1-s2.0-S0302283824024084-mmc2.xlsx"

xls        = pd.ExcelFile(file_path)
# --- Hoja de resultados de fármacos ---
tabla_s11  = xls.parse("Table S11", skiprows=1)  
tabla_s11  = tabla_s11.rename(columns={"Sample": "Patient"})

# Nos quedamos con Gemcitabina (viabilidad %) 
sensibilidad_gem = tabla_s11[["Patient", "Gemcitabine"]].dropna()

'''
Paso 2 · Unir con los clústeres existentes
'''
def merge_drug_cluster(drug_df, cluster_df, cluster_col):
    """
    Une la columna del fármaco con la asignación de clústeres.
    - drug_df: contiene ['Patient', '<Drug>']
    - cluster_df: contiene ['Patient', cluster_col]
    """
    merged = (
        drug_df.merge(cluster_df, on="Patient", how="inner")
               .dropna(subset=[cluster_col])
    )
    return merged

df_agg_complete_gem   = merge_drug_cluster(sensibilidad_gem, cluster_agg_complete, "Cluster_Agg")
df_k6_gem    = merge_drug_cluster(sensibilidad_gem, cluster_k6, "Cluster_k6")
df_estudio_gem  = merge_drug_cluster(sensibilidad_gem, tabla_s6_clusters, "Cluster_Estudio")


#Comprueba rápidamente cuántos pacientes de los 65 de S11 quedan con datos de Gemcitabina y un clúster asignado:
print(df_k6_gem.Cluster_k6.value_counts())
print(df_estudio_gem.Cluster_Estudio.value_counts())
	
'''
	Paso 3 · Explorar y visualizar
'''

'''
n – número de pacientes en el clúster

mean – media de viabilidad (cuanto mas bajo mejor funciona el farmaco)

std – desviación estándar absoluta

CV – dispersión relativa (std/mean); adimensional, por eso permite comparar clústeres con medias distintas 
    para ensayos biológicos: 
            -CV < 0,15 ➜ excelente; 
            -0,15–0,30 ➜ aceptable; 
            -> 0,30 ➜ la variabilidad empieza a comprometer la utilidad predictiva
'''
def summarise_within_cluster(df, drug_col, cluster_col):
    """
    Devuelve: n, media, SD y CV por clúster.
    """
    summary = (df.groupby(cluster_col)[drug_col]
                 .agg(n='count',
                      mean='mean',
                      std='std')
                 .assign(CV=lambda x: x['std'] / x['mean']))
    return summary

stats_k6  = summarise_within_cluster(df_k6_gem, "Gemcitabine", "Cluster_k6")
stats_k6 = stats_k6.reset_index()
print(stats_k6)
'''
CONLUSIONES K-MEANS CON K=6

Cluster 0: bastante heterogeneo, CV=0.44
Cluster 1: esta en el limite con CV=30
Cluster 2: muy homogeneo, CV=0.19 y de media un 82% de resistencia
Cluster 3: muy heterogeneo CV=0.47 y la muestra es muy pequeña
Cluster 4: igual al 3 pero mas heterogeneo
Cluster 5: heterogeneo

el cluster 0,3,4 y 5 funcionan muy mal


'''
#Varianza global k-means
'''
INCLUIR EN LA MEMORIA LA FORMULA DE N^2 (ETA-CUADRADO)

Sirve para cuantificar qué parte de la variabilidad explica el clustering
'''

var_total_k6 = df_k6_gem['Gemcitabine'].var(ddof=1)
var_within_k6 = sum(
    g['Gemcitabine'].var(ddof=1) * len(g)
    for _, g in df_k6_gem.groupby('Cluster_k6')
) / len(df_k6_gem)           # varianza ponderada dentro de clúster

eta2_k6 = 1 - var_within_k6 / var_total_k6
print(f"η² intraclúster = {eta2_k6:.2%}")

'''
η² intraclúster = 20.77%

Una quinta parte de la variabilidad de la respuesta a Gemcitabina 
se debe a diferencias sistemáticas entre clústeres; 
el resto (≈ 79 %) está dentro de ellos.

El agrupamiento K-means (k = 6) reduce la varianza global de la respuesta a Gemcitabina en un 20,8 % (η² = 0,208), 
valor que se considera un efecto medio. 

El clúster 2 concentra la mayor parte de la coherencia interna (CV = 0,19), 
mientras que los clústeres 0, 3, 4 y 5 permanecen heterogéneos. 

Por tanto, aunque el modelo captura una parte sustancial de la señal, 
existe margen de mejora mediante sub-clustering o ajustes en k.
'''

# ------------------------------------------------------------------
# 3. FIGURA 1  –  Media ± SD por clúster KMEANS 
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(stats_k6["Cluster_k6"].astype(str),        # eje x: etiquetas de clúster
        stats_k6["mean"],                          # altura de la barra = media
        yerr=stats_k6["std"],                      # “T” = desviación estándar
        capsize=4)                                 # puntitas de las barras de error
plt.ylabel("Media de viabilidad (%)")
plt.xlabel("Clúster K-means (k = 6)")
plt.title("Gemcitabine: media ± SD por clúster")
plt.figtext(0.02, 0.92, f"η² intraclúster = {eta2_k6*100:.2f}%", 
            ha="left", va="center", fontsize=9)    # añade η² en la esquina
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4. FIGURA 2  –  Coeficiente de variación (CV) por clúster KMEANS
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(stats_k6["Cluster_k6"].astype(str), stats_k6["CV"])
plt.ylabel("Coeficiente de variación (CV)")
plt.xlabel("Clúster K-means (k = 6)")
plt.title("Gemcitabine: CV por clúster")
plt.tight_layout()
plt.show()

'''
AGGLOMERATIVE CLUSTERING CON LINKAGE COMPLETE
'''

stats_agg_complete  = summarise_within_cluster(df_agg_complete_gem, "Gemcitabine", "Cluster_Agg")
stats_agg_complete = stats_agg_complete.reset_index()
print(stats_agg_complete)
'''
CONLUSIONES AGGLOMERATIVE CLUSTERING CON LINKAGE COMPLETE

Muy heterogeneos menos el cluster 4 que es consistente y tiene una resistencia clara

Los demas funcionan muy mal

'''

var_total_agg_complete = df_agg_complete_gem['Gemcitabine'].var(ddof=1)
var_within_agg_complete = sum(
    g['Gemcitabine'].var(ddof=1) * len(g)
    for _, g in df_agg_complete_gem.groupby('Cluster_Agg')
) / len(df_agg_complete_gem)           # varianza ponderada dentro de clúster

eta2_agg_complete = 1 - var_within_agg_complete / var_total_agg_complete
print(f"η² intraclúster = {eta2_agg_complete:.2%}")

'''
η² intraclúster = 27.62%

A partir del 26% mas o menos el cluster es relevante, es mejor que k-means

“El agrupamiento aglomerativo explica el 27,6 % de la variabilidad global en la respuesta a Gemcitabina (η² = 0,276), 
lo que se considera un efecto grande. 

Esto significa que, al incorporar la etiqueta de clúster, 
reducimos cerca de un tercio del ruido biológico observador-dependiente, 
demostrando la relevancia clínica de la estratificación. 

Sin embargo, la varianza residual (> 70 %) indica que ciertos clústeres continúan mezclando subpoblaciones
 y deben refinarse o validarse con más datos.”
'''

# ------------------------------------------------------------------
# 3. FIGURA 1  –  Media ± SD por clúster AGGLOMERATIVE CLUSTERING CON LINKAGE COMPLETE 
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(stats_agg_complete["Cluster_Agg"].astype(str),        # eje x: etiquetas de clúster
        stats_agg_complete["mean"],                          # altura de la barra = media
        yerr=stats_agg_complete["std"],                      # “T” = desviación estándar
        capsize=4)                                 # puntitas de las barras de error
plt.ylabel("Media de viabilidad (%)")
plt.xlabel("Clúster Aglomerativo (linkage = Complete)")
plt.title("Gemcitabine: media ± SD por clúster")
plt.figtext(0.02, 0.92, f"η² intraclúster = {eta2_agg_complete*100:.2f}%", 
            ha="left", va="center", fontsize=9)    # añade η² en la esquina
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4. FIGURA 2  –  Coeficiente de variación (CV) por clúster KMEANS
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(stats_agg_complete["Cluster_Agg"].astype(str), stats_agg_complete["CV"])
plt.ylabel("Coeficiente de variación (CV)")
plt.xlabel("Clúster Aglomerativo (linkage = Complete)")
plt.title("Gemcitabine: CV por clúster")
plt.tight_layout()
plt.show()



'''
CLUSTERING DEL ESTUDIO
'''

stats_estudio  = summarise_within_cluster(df_estudio_gem, "Gemcitabine", "Cluster_Estudio")
stats_estudio = stats_estudio.reset_index()
print(stats_estudio)
'''
CONLUSIONES CLUSTERING DEL ESTUDIO

Únicamente el clúster 2 muestra un CV < 0 ,30 → respuesta interna estable.

Los clústeres 3 y 1 son especialmente variables (CV ≈ 0,38–0,64).

'''

var_total_estudio = df_estudio_gem['Gemcitabine'].var(ddof=1)
var_within_estudio = sum(
    g['Gemcitabine'].var(ddof=1) * len(g)
    for _, g in df_estudio_gem.groupby('Cluster_Estudio')
) / len(df_estudio_gem)           # varianza ponderada dentro de clúster

eta2_estudio = 1 - var_within_estudio / var_total_estudio
print(f"η² intraclúster = {eta2_estudio:.2%}")

'''
η² intraclúster = 25.78%
A partir del 26% mas o menos el cluster es relevante, es mejor que k-means

“Con las etiquetas clínicas originales, la proporción de varianza explicada por los clústeres es η² = 25,8 %, 
lo que corresponde a un efecto medio–alto. 

El valor mejora respecto al modelo K-means (20,8 %) pero es inferior al alcanzado con el agrupamiento aglomerativo (27,6 %).”
'''

# ------------------------------------------------------------------
# 3. FIGURA 1  –  Media ± SD por clúster CLUSTERING DEL ESTUDIO 
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(stats_estudio["Cluster_Estudio"].astype(str),        # eje x: etiquetas de clúster
        stats_estudio["mean"],                          # altura de la barra = media
        yerr=stats_estudio["std"],                      # “T” = desviación estándar
        capsize=4)                                 # puntitas de las barras de error
plt.ylabel("Media de viabilidad (%)")
plt.xlabel("Clúster Estudio")
plt.title("Gemcitabine: media ± SD por clúster")
plt.figtext(0.02, 0.92, f"η² intraclúster = {eta2_estudio*100:.2f}%", 
            ha="left", va="center", fontsize=9)    # añade η² en la esquina
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4. FIGURA 2  –  Coeficiente de variación (CV) por clúster CLUSTERING DEL ESTUDIO 
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(stats_estudio["Cluster_Estudio"].astype(str), stats_estudio["CV"])
plt.ylabel("Coeficiente de variación (CV)")
plt.xlabel("Clúster Estudio")
plt.title("Gemcitabine: CV por clúster")
plt.tight_layout()
plt.show()

'''
COMPARACION DE METODOS
'''

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

# ------------------------------------------------------------------
# 1 · recopilar los resultados
# ------------------------------------------------------------------
eta2_dict = {
    "K-means (k=6)"    : eta2_k6,      
    "Aglomerativo"     : eta2_agg_complete,      
    "Clúster (estudio)": eta2_estudio       
}

eta2_dict

# DataFrames con CV por clúster 

cvs = pd.concat([
    stats_k6.assign(Método="K-means (k=6)")  [["CV","Método"]],
    stats_agg_complete.assign(Método="Aglomerativo (Linkage: Complete)")  [["CV","Método"]],
    stats_estudio.assign(Método="Clúster (estudio)")[["CV","Método"]]
])

cvs

# ------------------------------------------------------------------
# 2 · FIGURA A  ─  η² por método
# ------------------------------------------------------------------
plt.figure(figsize=(5,3))
plt.bar(eta2_dict.keys(), [v*100 for v in eta2_dict.values()])
plt.ylabel("η² intraclúster (%)")
plt.title("Varianza explicada por cada método")
plt.ylim(0,50)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3 · FIGURA B  ─  distribución de CV
# ------------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="Método", y="CV", data=cvs, width=0.5, showfliers=False)
sns.stripplot(x="Método", y="CV", data=cvs, color=".25", size=4)
plt.axhline(0.30, ls="--", lw=1)      # umbral “clúster coherente”
plt.ylabel("Coeficiente de variación (CV)")
plt.title("Dispersión interna de cada clúster")
plt.xticks(rotation=90)               # <── aquí
plt.tight_layout()
plt.show()


'''
“El método aglomerativo alcanza el η² más alto (27,6 %), por delante del esquema clínico original (25,8 %) y del K-means (20,8 %). 
Sin embargo, todos los métodos comparten un patrón: solo un clúster por agrupación cae por debajo del umbral CV ≤ 0,30 (línea discontinua). 
Esto indica que, aunque el aglomerativo explica más varianza global, la mayor parte de los clústeres siguen siendo heterogéneos y requieren refinamiento.”
'''
