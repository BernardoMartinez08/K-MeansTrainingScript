import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os
import time


# Función para cargar datos desde un archivo CSV
def load_data(file_path):
    print(f'\n\n\nCargando datos desde {file_path}')
    data = pd.read_csv(file_path)
    return data[['x', 'y']].values


# Función para entrenar K-means
def train_kmeans(data, k):
    print(f'\n\nEntrenando K-means con k = {k}')
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)
    end_time = time.time()
    print(f'Tiempo de entrenamiento para k = {k}: {end_time - start_time:.2f} segundos')
    return kmeans


# Función para guardar el modelo entrenado
def save_model(kmeans, model_dir, k):
    print(f'Guardando modelo en {model_dir}')
    model_filename = f'{model_dir}/kmeans_k{k}.joblib'
    joblib.dump(kmeans, model_filename)
    print(f'Modelo guardado en {model_filename}')


# Función para cargar un modelo entrenado
def load_model(model_dir, k):
    model_filename = f'{model_dir}/kmeans_k{k}.joblib'
    print(f'Cargando modelo desde {model_filename}')
    return joblib.load(model_filename)


# Función para graficar los resultados
def plot_kmeans(data, kmeans, k, sample_percentage=0.1):
    print(f'Graficando resultados para k = {k}')
    start_time = time.time()

    # Muestrear aleatoriamente un subconjunto de los datos
    sample_size = int(len(data) * sample_percentage)
    sample_indices = np.random.choice(len(data), sample_size, replace=False)
    data_sample = data[sample_indices]

    y_kmeans = kmeans.predict(data_sample)
    centers = kmeans.cluster_centers_

    plt.scatter(data_sample[:, 0], data_sample[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
    plt.title(f'K = {k}')
    plt.xlabel('X')
    plt.ylabel('Y')

    end_time = time.time()
    print(f'Tiempo de graficación para k = {k}: {end_time - start_time:.2f} segundos')


def plot_inertia_and_silhouette(data, dataset_name, k_max, sample_percentage=0.1):
    # Muestrear aleatoriamente un subconjunto de los datos
    sample_size = int(len(data) * sample_percentage)
    sample_indices = np.random.choice(len(data), sample_size, replace=False)
    data_sample = data[sample_indices]

    inertias = []
    silhouette_scores = []

    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data_sample)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data_sample, kmeans.labels_))

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Número de Clusters (K)')
    ax1.set_ylabel('Inercia', color=color)
    ax1.plot(range(2, k_max + 1), inertias, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Coeficiente de Silueta', color=color)
    ax2.plot(range(2, k_max + 1), silhouette_scores, 'o-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Inercia y Coeficiente de Silueta para {dataset_name}')
    plt.show()


# Main script
def main(data_path, model_dir, plt_dir, k_max):
    datasets = {
        'Dataset 1': f'{data_path}/datos_1.csv',
        'Dataset 2': f'{data_path}/datos_2.csv',
        'Dataset 3': f'{data_path}/datos_3.csv',
    }

    # Aplicar K-means y generar gráficos para cada dataset
    for name, file_path in datasets.items():
        data = load_data(file_path)

        plt.figure(figsize=(15, 10))

        model_path = f'{model_dir}/{name}'
        plt_path = f'{plt_dir}/{name}'

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(plt_path, exist_ok=True)

        for k in range(1, k_max + 1):
            kmeans = train_kmeans(data, k)
            save_model(kmeans, model_path, k)
            plt.subplot(2, 3, k)
            plot_kmeans(data, kmeans, k)

        print(f'\n\n\nMostrando resultados de K-means para {name}')
        plt.suptitle(f'Resultados de K-means para {name}')
        print(f'Guardando gráfico en {plt_path}/kmeans_results.png')
        plt.savefig(f'{plt_path}/kmeans_results.png')
        print(f'Gráfico guardado en {plt_path}/kmeans_results.png')
        plt.show()
        plt.close()

        # Aplicar la función a cada dataset
        #plot_inertia_and_silhouette(data, name, k_max)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: python kmeans_training_script.py <data_dir> <models_dir> <plts_dir> <k_max>')
        sys.exit(1)

    data_path = sys.argv[1]
    model_dir = sys.argv[2]
    plt_dir = sys.argv[3]
    k_max = int(sys.argv[4])
    main(data_path, model_dir, plt_dir, k_max)
