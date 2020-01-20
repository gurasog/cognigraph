from pylsl import StreamInlet, resolve_stream, resolve_streams
import numpy as np
import pickle
from sklearn.cluster import KMeans
import scipy.signal as sp
from scipy.signal import hilbert
from matplotlib import pyplot as plt


def checking_quality_of_clusters(x, list_of_clusters_and_their_columns, clusters_number):
    x = np.array(x)
    list_of_mean_activity = []
    list_of_var_activity = []

    fig, axs = plt.subplots(clusters_number, 1)
    for i in range(clusters_number):
        voxels_of_cluster = list_of_clusters_and_their_columns[i]
        sum_of_voxels_of_cluster = sum(x[voxels_of_cluster])
        num_of_voxels_of_cluster = len(voxels_of_cluster)
        avg_activity = sum_of_voxels_of_cluster / num_of_voxels_of_cluster
        # print("Средняя активность в канале: "+str(avg_activity))

        list_of_mean_activity.append(sum_of_voxels_of_cluster / num_of_voxels_of_cluster)

        variance = 0
        # print("Количество вокрселей в кластере: "+str(len(x[voxels_of_cluster])))

        for j in range(len(voxels_of_cluster)):
            index_of_voxels_claster = voxels_of_cluster[j]  # нашли номер вокселя по которому будем делать
            data_in_one_voxel_of_claster = x[index_of_voxels_claster]  # взяли данные из этого вокселя
            dif = data_in_one_voxel_of_claster - avg_activity  # вычли средние
            variance = variance + (dif) ** 2  # склыдваем суммы возведенные в квадрат

        variance = variance / len(voxels_of_cluster)
        # list_of_mean_activity.append(sum_of_voxels_of_cluster / num_of_voxels_of_cluster)
        list_of_var_activity.append(variance)

        # print(avg_activity/variance) # по идее должно быть большим
        # list_of_var_activity=np.array(list_of_var_activity)
        # list_of_mean_activity=np.array(list_of_mean_activity)
        # print(list_of_var_activity/list_of_mean_activity)
        axs[i].plot(np.arange(len(avg_activity / variance)), avg_activity / variance)
    print("checking_quality_of_clusters was run")
    plt.show()


def checking_plots_of_clustered_data(x, list_of_clusters_and_their_columns, clusters_number):
    x = np.array(x)
    fig = plt.figure(figsize=(20, 12))

    for i in range(clusters_number):
        voxels_of_cluster = list_of_clusters_and_their_columns[i]
        amount_of_samples=len(x[0])
        x_for_plot=np.arange(amount_of_samples)
        ax = fig.add_subplot( 2, 2, i+1)
        for j in voxels_of_cluster:
            ax.plot(x_for_plot,x[j])
            plt.grid()

    print("checking_plots_of_clustered_data was run")
    plt.show()


def checking_plots_of_clustered_data_with_threshold(x, list_of_clusters_and_their_columns, clusters_number,threshold):
    x = np.array(x)
    x=x[:,:threshold]
    fig = plt.figure(figsize=(20, 12))

    for i in range(clusters_number):
        voxels_of_cluster = list_of_clusters_and_their_columns[i]
        amount_of_samples=len(x[0])
        x_for_plot=np.arange(amount_of_samples)
        ax = fig.add_subplot( 2, 2, i+1)
        for j in voxels_of_cluster:
            ax.plot(x_for_plot,x[j])
            plt.grid()

    print("checking_plots_of_clustered_data was run")
    plt.show()
