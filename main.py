import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageColor import colormap
from matplotlib.colors import ListedColormap

from sklearn.cluster import KMeans


# Wczytaj plik csv
file = './Dane/data2.csv'

# Wczytaj dane z pliku CSV
# dane to obiekt typu DataFrame
data = pd.read_csv(file, header=None, sep=',')
data.columns = ['Dlugosc kielicha', 'Szerokosc kielicha', 'Dlugosc platka', 'Szerokosc platka']

# Tworzymy nową figurę i ustawiamy jej rozmiar
fig, axs = plt.subplots(3, 2, figsize=(8, 14))

# Tworzymy własną mapę kolorów
color_array = ['red', 'green', 'blue']
custom_cmap = ListedColormap(color_array)

# TODO: KOSMETYKA - rozmiary, kolory, opisy, kształty, itp.
def k_means(x, y, feature_data1, feature_data2, desc1, desc2):
    """
    Wykonuje algorytm k-średnich na danych i wyświetla wyniki na wykresie.
    :param x: współrzędna x wykresu
    :param y: współrzędna y wykresu
    :param feature_data1: dane numeryczne dla danej cechy 1.
    :param feature_data2: dane numeryczne dla danej cechy 2.
    :param desc1: opis danej cechy 1.
    :param desc2: opis danej cechy 2.
    :return: None
    """
    # Normalizujemy dane
    feature_data_norm_1 = (feature_data1 - feature_data1.min()) / (feature_data1.max() - feature_data1.min())
    feature_data_norm_2 = (feature_data2 - feature_data2.min()) / (feature_data2.max() - feature_data2.min())

    # Tworzymy macierz danych
    # zip - łączy dwie listy w jedną
    # list - tworzy listę z obiektu zip
    matrix = np.array(list(zip(feature_data1, feature_data2)))
    matrix_norm = np.array(list(zip(feature_data_norm_1, feature_data_norm_2)))

    # Tworzymy obiekt kMeans
    # n_clusters - liczba klastrów
    # max_iter - maksymalna liczba iteracji
    kmeans = KMeans(n_clusters=3, init='random', max_iter=20)

    # Uczymy model
    kmeans.fit(matrix_norm)

    # Przewidujemy przynależność do klastra
    y_kmeans = kmeans.predict(matrix_norm)

    # Rysujemy wykres, gdzie punkty są kolorowane na podstawie przynależności do klastra, ale bez wypełnienia
    axs[x, y].scatter(matrix[:, 0], matrix[:, 1], c=y_kmeans, s=50, cmap=custom_cmap, edgecolors='k',)

    # Odwracamy normalizację
    centers = kmeans.cluster_centers_

    # Odwracamy normalizację centroidów
    centers_original = np.zeros_like(centers)
    centers_original[:, 0] = centers[:, 0] * (feature_data1.max() - feature_data1.min()) + feature_data1.min()
    centers_original[:, 1] = centers[:, 1] * (feature_data2.max() - feature_data2.min()) + feature_data2.min()

    # Rysujemy centra klastrów na wykresie w oryginalnej skali
    axs[x, y].scatter(centers_original[:, 0], centers_original[:, 1], c=color_array[:], s=200, alpha=1.0)

    # Ustawiamy tytuł wykresu
    axs[x, y].set_title(f'{desc1} vs {desc2}')

    # Ustawiamy etykiety osi
    axs[x, y].set_xlabel(desc1)
    axs[x, y].set_ylabel(desc2)

k_means(0, 0, data['Dlugosc kielicha'], data['Szerokosc kielicha'], 'Dlugosc kielicha', 'Szerokosc kielicha')
k_means(0, 1, data['Dlugosc kielicha'], data['Dlugosc platka'], 'Dlugosc kielicha', 'Dlugosc platka')
k_means(1, 0, data['Dlugosc kielicha'], data['Szerokosc platka'], 'Dlugosc kielicha', 'Szerokosc platka')
k_means(1, 1, data['Szerokosc kielicha'], data['Dlugosc platka'], 'Szerokosc kielicha', 'Dlugosc platka')
k_means(2, 0, data['Szerokosc kielicha'], data['Szerokosc platka'], 'Szerokosc kielicha', 'Szerokosc platka')
k_means(2, 1, data['Dlugosc platka'], data['Szerokosc platka'], 'Dlugosc platka', 'Szerokosc platka')


def k_means_wcss(x, y, feature_data1, feature_data2, desc1, desc2):
    """
    Wykonuje algorytm k-średnich na danych i wyświetla wyniki na wykresie.
    :param x: współrzędna x wykresu
    :param y: współrzędna y wykresu
    :param feature_data1: dane numeryczne dla danej cechy 1.
    :param feature_data2: dane numeryczne dla danej cechy 2.
    :param desc1: opis danej cechy 1.
    :param desc2: opis danej cechy 2.
    :return: None
    """

    feature_data_norm_1 = (feature_data1 - feature_data1.min()) / (feature_data1.max() - feature_data1.min())
    feature_data_norm_2 = (feature_data2 - feature_data2.min()) / (feature_data2.max() - feature_data2.min())

    matrix_norm = np.array(list(zip(feature_data_norm_1, feature_data_norm_2)))

    iter_array = []
    wcss_array = []

    print('\n-----------------')
    print('Opis:', desc1, desc2)
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, init='random', max_iter=100)
        kmeans.fit(matrix_norm)
        iter_array.append(kmeans.n_iter_)
        wcss_array.append(kmeans.inertia_)
        print(f'K={k}, iter={kmeans.n_iter_}, wcss={kmeans.inertia_}')

k_means_wcss(0, 0, data['Dlugosc kielicha'], data['Szerokosc kielicha'], 'Dlugosc kielicha', 'Szerokosc kielicha')
k_means_wcss(0, 1, data['Dlugosc kielicha'], data['Dlugosc platka'], 'Dlugosc kielicha', 'Dlugosc platka')
k_means_wcss(1, 0, data['Dlugosc kielicha'], data['Szerokosc platka'], 'Dlugosc kielicha', 'Szerokosc platka')
k_means_wcss(1, 1, data['Szerokosc kielicha'], data['Dlugosc platka'], 'Szerokosc kielicha', 'Dlugosc platka')
k_means_wcss(2, 0, data['Szerokosc kielicha'], data['Szerokosc platka'], 'Szerokosc kielicha', 'Szerokosc platka')
k_means_wcss(2, 1, data['Dlugosc platka'], data['Szerokosc platka'], 'Dlugosc platka', 'Szerokosc platka')

# Ustawienie odstępów między wykresami
plt.subplots_adjust(wspace=0.3, hspace=0.6)

# Wyświetlenie wykresów
plt.show()
