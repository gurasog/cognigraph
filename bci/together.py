from pylsl import StreamInlet, resolve_stream, resolve_streams
import numpy as np
import pickle
from sklearn.cluster import KMeans
import scipy.signal as sp
from scipy.signal import hilbert
from matplotlib import pyplot as plt
from catch_data import *
from gain_matrix_clasterization import *
from chek_perfomance import *
from filtering import *
from classifiers import *
from feature_extraction import *

from sklearn.decomposition import PCA





initial_data = prepare_data(1000)
clusters_number = 50
G,L = read_data()
predicted_clusters = G_clasterization(G, L, clusters_number)
list_of_list_of_clusters_and_their_columns=get_list_of_list_of_clusters_and_their_columns(initial_data, predicted_clusters, clusters_number)
list_of_mean_activity = mean_activity_acc_to_clusterization(initial_data, list_of_list_of_clusters_and_their_columns, clusters_number)
#checking_plots_of_clustered_data_with_threshold(initial_data, list_of_list_of_clusters_and_their_columns, 3, 100)
#checking_plots_of_clustered_data(initial_data, list_of_list_of_clusters_and_their_columns)
checking_quality_of_clusters(initial_data, list_of_list_of_clusters_and_their_columns, 4)
filter_order = 6
Fs = 100
Nf = Fs/2

#получаем лист размера 10 на колчиесвто сэмплов, тут было бы уместно проделать даунсемплинг
# вроде как это можно сделать с помощью scipy.signal.decimate пока что делать не буду
outcome_data=[]
for i in range(clusters_number):
    a_1, a_2, a_3 = three_envelope_of_signal(list_of_mean_activity[i], filter_order,Nf)
    outcome_data.append(a_1)
    outcome_data.append(a_2)
    outcome_data.append(a_3)



data_after_pca = run_pca(outcome_data) # here we see data with length of time point
                                        # in first dimension and with length of components in the second dimension

data_after_window=prepare_data_with_window(data_after_pca,5)

print(len(data_after_window))
y = np.random.randint(low=0, high=2, size=len(data_after_window))
y_predicted=run_naive_bayes(data_after_window, y)
print(len(y_predicted))





'''

fig, axs = plt.subplots(6, 5)
for i in range(6):
    for j in range(5):
        axs[i, j].plot(np.arange(len(outcome_data[i*5+j])),outcome_data[i*5+j])

plt.show()

'''
