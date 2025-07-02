# TFG_Matematicas
Desarrollo del TFG de Matemáticas de Thatyana Alarza Souza basado en el "Análisis de subtipos de cáncer de vejiga utilizando Machine Learning".

En este Trabajo Fin de Grado se investiga la estratificación molecular de 97 organoides derivados de pacientes con cáncer de vejiga mediante técnicas de clustering no supervisado y su relación con la respuesta ex vivo al medicamento gemcitabina. Esto se desarrolla a partir de la matriz de expresión génica log_2 publicada en el estudio "Integrative Drug Screening" a partir del cual se construye una red de similitud entre pacientes y se comparan dos algoritmos: K-means y  Agglomerative Clustering. 

La coherencia interna se evalúa con la métrica de Silhouette y el número óptimo de grupos se determina combinando el método del codo y la proyección PCA. La calidad externa se contrasta mediante el índice de Jaccard frente a la partición clínica original. Finalmente, se integra la viabilidad de los PDOs frente a gemcitabina y se cuantifica la varianza explicada y el coeficiente de variación (CV) dentro de cada clúster. 
