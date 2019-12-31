import pickle

import numpy as np
import os
from sklearn.cluster import KMeans


'''
data_folder = os.path.join("Desktop", "Master")
file_to_open = os.path.join(data_folder, "G_matrix.txt")
'''

with open('../../matrix.pickle', 'rb') as f:
    G = pickle.load(f)

with open('../../labels_for_clasterization.pickle', 'rb') as f:
     L= pickle.load(f)

L=list(map(int,L))
G=G.reshape(G.shape[0], 8196, 3)
"""плохо что тут написано 59 по идее нужно передавать описание всей выборки """
G_new=G[:,:,0]+(G[:,:,1])+G[:,:,2]
G_selected=G_new[:,L]


u,d,v=np.linalg.svd(G_selected)
list_of_selected_coef=(np.cumsum(d)/sum(d)>0.95).tolist()
number_of_selected_coef=list_of_selected_coef.index(True)
G_selected_decreased= np.matmul(u[:,0:lol_index+1].T , G_selected)


clusters_number=10
km=KMeans(n_clusters=clusters_number)
km.fit(G_selected_decreased.T)
predicted=km.predict(G_selected_decreased.T)


"""здесь лолден быть объявлен поток данных принимаемый из приложки"""
x=np.random.rand(483,100)

list_of_clusters_and_their_colums=[]
list_of_mean_activity=[]
for k in range(clusters_number):
    list_of_clusters_and_their_colums.append([i for i, x in enumerate(predicted) if x == k])
    list_of_mean_activity.append(sum(x[list_of_clusters_and_their_colums[k]]) / len(x[list_of_clusters_and_their_colums[k]]))


"""а дальше должны быть фильтры"""