# -*- coding: utf-8 -*-
"""Pokemon_K_MEANS_IRIS_A126.ipynb"""

# IMPORTA BIBLIOTECAS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import seaborn as sb
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. CARREGAR O DATASET
# -------------------------------------------------------
df = pd.read_csv('pokemon_gen1.csv')

# VISUALIZAR O DATAFRAME - MOSTRAR AS PRIMEIRAS LINHAS
df

# -------------------------------------------------------
# 2. DEFININDO O K
# -------------------------------------------------------
k = 4  # Fraco / Médio / Forte / Lendário

# -------------------------------------------------------
# 3. TRANSFORMAR RÓTULOS: FAIXA DE FORÇA BASEADA NO TOTAL
# -------------------------------------------------------
# O CSV JÁ TEM A COLUNA TOTAL, É SÓ USAR!
df['Faixa_Forca'] = pd.qcut(df['Total'], q=k, labels=False, duplicates='drop')
y_true = df['Faixa_Forca'].values

# VISUALIZAR A TRANSFORMAÇÃO
y_true

# VISUALIZAR O DATAFRAME ORIGINAL COM FAIXA DE FORÇA
sb.pairplot(df, hue='Faixa_Forca')

# -------------------------------------------------------
# 4. PREPARAR OS DADOS NUMÉRICOS E NORMALIZAR
# -------------------------------------------------------
X = df.select_dtypes(include=[np.number]).values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)  # substitui X pela versão normalizada

# -------------------------------------------------------
# 5. TREINAMENTO COM KMEANS
# -------------------------------------------------------
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)
clusters = kmeans.fit_predict(X)
print(clusters)

# EXIBIR OS RÓTULOS GERADOS
kmeans.labels_

# CRIANDO UMA NOVA COLUNA NO DATAFRAME COM OS CLUSTERS
df['k_classes'] = kmeans.labels_

# EXIBE AS PRIMEIRAS LINHAS COM A NOVA COLUNA
df.head(151)

sb.pairplot(df, hue='k_classes')
sb.pairplot(df, hue='Faixa_Forca')

# -------------------------------------------------------
# 6. MATRIZ DE CONFUSÃO E ACURÁCIA
# -------------------------------------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

conf_matrix = confusion_matrix(y_true, clusters)
row_ind, col_ind = linear_sum_assignment(-conf_matrix)

clusters_mapeados = np.zeros_like(clusters)
for i, j in zip(row_ind, col_ind):
    clusters_mapeados[clusters == i] = j

acuracia = accuracy_score(y_true, clusters_mapeados)
print(f"Acurácia: {acuracia * 100:.2f}%")

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Reds')
plt.title("Matriz de Confusão")
plt.show()

# -------------------------------------------------------
# 7. MÉTODO DO COTOVELO - ENCONTRAR O MELHOR K
# -------------------------------------------------------
inercias = []

for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inercias.append(kmeans.inertia_)

plt.plot(range(1, 15), inercias, marker='o')
plt.xlabel('Num de clusters')
plt.ylabel('Inércias')
plt.title('Método Cotovelo')
plt.show()
print(inercias)

# -------------------------------------------------------
# 8. ÍNDICE DE SILHUETA
# -------------------------------------------------------
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.plot(range(2, 15), silhouette_scores, marker='o')
plt.xlabel('Num de clusters')
plt.ylabel('Índice de Silhueta')
plt.title('Índice de Silhueta por K')
plt.show()
print(silhouette_scores)