import pickle

import numpy as np
import os
from sklearn.cluster import KMeans

import scipy.io as sio
import scipy.signal as sp
from os import listdir
from os.path import isfile, join

import sklearn.linear_model as lm
import time

# do stuff


'''
data_folder = os.path.join("Desktop", "Master")
file_to_open = os.path.join(data_folder, "G_matrix.txt")
'''
#считали g_matrix

def read_data():
    with open('../../matrix.pickle', 'rb') as f:
        G = pickle.load(f)

    #считали какие воксели матрицы будем использовать
    with open('../../labels_for_clasterization.pickle', 'rb') as f:
         L= pickle.load(f)

    L = list(map(int, L))
    G = G.reshape(G.shape[0], 8196, 3) # that is not fact that we will always have 3 dimensional matrix

    return G, L

def G_change(G_new,L):
    G_selected=G_new[:,L]
    u,d,v=np.linalg.svd(G_selected)
    list_of_selected_coef=(np.cumsum(d)/sum(d)>0.95).tolist()
    number_of_selected_coef=list_of_selected_coef.index(True)
    #почему такое разложение дает какой-то адеватный результат
    G_selected_decreased= np.matmul(u[:,0:number_of_selected_coef+1].T , G_selected)
    return G_selected_decreased


def G_clasterization(G):
    G_1=G[:,:,0]
    G_2=G[:,:,1]
    G_3=G[:,:,2]

    G_1_dec=G_change(G_1,L)
    G_2_dec=G_change(G_2,L)
    G_3_dec=G_change(G_3,L)

    G_all=np.vstack((G_1_dec, G_2_dec,G_3_dec))


    clusters_number=10
    km=KMeans(n_clusters=clusters_number)
    km.fit(G_all.T)
    predicted=km.predict(G_all.T)
    return predicted



"""здесь лолден быть объявлен поток данных принимаемый из приложки"""

x=np.random.rand(1203,100) # здесь нужно вписать правильное количество вокселей

def mean_activity_acc_to_clasterization(x,predicted, clusters_number):
    '''
    x - data
    predicted - which voxel is in one group with others
    clusters_number - the number of clasters
    '''

    list_of_mean_activity=[]
    list_of_clusters_and_their_columns=[]
    for k in range(clusters_number):
        list_of_clusters_and_their_columns.append([i for i, x in enumerate(predicted) if x == k])
    #print(list_of_clusters_and_their_colums)

    for i in range(clusters_number):
        list_of_mean_activity.append(sum(x[list_of_clusters_and_their_columns[0]])/len(x[list_of_clusters_and_their_columns[0]]))

     return list_of_mean_activity
