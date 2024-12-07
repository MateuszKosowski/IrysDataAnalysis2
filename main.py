import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

# Wczytaj plik csv
file = './Dane/data2.csv'

# Wczytaj dane z pliku CSV
# dane to obiekt typu DataFrame
data = pd.read_csv(file, header=None, sep=',')
data.columns = ['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka']

# Tworzymy nową figurę i ustawiamy jej rozmiar
fig, axs = plt.subplots(3, 2, figsize=(10, 18))

# Tworzymy własną mapę kolorów
color_array = ['red', 'green', 'blue']
custom_cmap = ListedColormap(color_array)

# Tworzymy macierz danych i normalizujemy ją
matrix = np.array(data)
matrix_norm = np.array((data - data.min()) / (data.max() - data.min()))

# Tworzymy obiekt kMeans
# n_clusters - liczba klastrów
# max_iter - maksymalna liczba iteracji
kmeans = KMeans(n_clusters=3, init='random', max_iter=20)

# Uczymy model
kmeans.fit(matrix_norm)

# Przewidujemy przynależność do klastra
y_kmeans = kmeans.predict(matrix_norm)

# Odwracamy normalizację
centers = kmeans.cluster_centers_

# Odwracamy normalizację centroidów
centers_original = np.zeros_like(centers)
for i in range(4):
    centers_original[:, i] = centers[:, i] * (matrix[:, i].max() - matrix[:, i].min()) + matrix[:, i].min()

def k_means():

    # Rysujemy wykres, gdzie punkty są kolorowane na podstawie przynależności do klastra, ale bez wypełnienia
    axs[0, 0].scatter(matrix[:, 0], matrix[:, 1], facecolors='none', edgecolors=[color_array[j] for j in y_kmeans], s=50)
    axs[0, 0].set_xlabel('Długość działki kielicha [cm]')
    axs[0, 0].set_ylabel('Szerokość działki kielicha [cm]')

    axs[0, 1].scatter(matrix[:, 0], matrix[:, 2], facecolors='none', edgecolors=[color_array[j] for j in y_kmeans], s=50)
    axs[0, 1].set_xlabel('Długość działki kielicha [cm]')
    axs[0, 1].set_ylabel('Długość płatka [cm]')

    axs[1, 0].scatter(matrix[:, 0], matrix[:, 3], facecolors='none', edgecolors=[color_array[j] for j in y_kmeans], s=50)
    axs[1, 0].set_xlabel('Długość działki kielicha [cm]')
    axs[1, 0].set_ylabel('Szerokość płatka [cm]')

    axs[1, 1].scatter(matrix[:, 1], matrix[:, 2], facecolors='none', edgecolors=[color_array[j] for j in y_kmeans], s=50)
    axs[1, 1].set_xlabel('Szerokość działki kielicha [cm]')
    axs[1, 1].set_ylabel('Długość płatka [cm]')

    axs[2, 0].scatter(matrix[:, 1], matrix[:, 3], facecolors='none', edgecolors=[color_array[j] for j in y_kmeans], s=50)
    axs[2, 0].set_xlabel('Szerokość działki kielicha [cm]')
    axs[2, 0].set_ylabel('Szerokość płatka [cm]')

    axs[2, 1].scatter(matrix[:, 2], matrix[:, 3], facecolors='none', edgecolors=[color_array[j] for j in y_kmeans], s=50)
    axs[2, 1].set_xlabel('Długość płatka [cm]')
    axs[2, 1].set_ylabel('Szerokość płatka [cm]')

    # Rysujemy centra klastrów na wykresie w oryginalnej skali
    axs[0, 0].scatter(centers_original[:, 0], centers_original[:, 1], c=color_array[:], s=100, alpha=1.0, marker='D', edgecolors="k")
    axs[0, 1].scatter(centers_original[:, 0], centers_original[:, 2], c=color_array[:], s=100, alpha=1.0, marker='D', edgecolors="k")
    axs[1, 0].scatter(centers_original[:, 0], centers_original[:, 3], c=color_array[:], s=100, alpha=1.0, marker='D', edgecolors="k")
    axs[1, 1].scatter(centers_original[:, 1], centers_original[:, 2], c=color_array[:], s=100, alpha=1.0, marker='D', edgecolors="k")
    axs[2, 0].scatter(centers_original[:, 1], centers_original[:, 3], c=color_array[:], s=100, alpha=1.0, marker='D', edgecolors="k")
    axs[2, 1].scatter(centers_original[:, 2], centers_original[:, 3], c=color_array[:], s=100, alpha=1.0, marker='D', edgecolors="k")


k_means()

plt.subplots_adjust(wspace=0.3, hspace=0.3)

def k_means_wcss():

    iter_array = []
    wcss_array = []

    print('\n-----------------')
    print('WCSS')
    for k in range(2, 11):
        kmeans1 = KMeans(n_clusters=k, init='random', max_iter=100)
        kmeans1.fit(matrix_norm)
        iter_array.append(kmeans1.n_iter_)
        wcss_array.append(kmeans1.inertia_)
        print(f'k={k} wcss={kmeans1.inertia_} iter={kmeans1.n_iter_}')

    fig1 = plt.figure(figsize=(6,4))
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(range(2, 11), wcss_array)
    ax.scatter(range(2, 11), wcss_array)
    ax.set_xlabel('k')
    ax.set_ylabel('WCSS')
    ax.set_title('Wykres zależności WCSS od współczynnika k')

k_means_wcss()

# Ustawienie odstępów między wykresami
plt.subplots_adjust(wspace=0.3, hspace=0.6)

# Wyświetlenie wykresów
plt.show()
