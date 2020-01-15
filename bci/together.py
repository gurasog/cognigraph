from pylsl import StreamInlet, resolve_stream, resolve_streams
import numpy as np
import pickle
from sklearn.cluster import KMeans
import scipy.signal as sp
from scipy.signal import hilbert
from matplotlib import pyplot as plt

def prepare_data(time):
    a = resolve_streams()
    inlet = StreamInlet(a[0])
    print(a[0].name())
    data=[]
    while len(data)<=time:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        data.append(sample)

    print("prepare_data was run")
    return data


def read_data():
    with open('../../matrix.pickle', 'rb') as f:
        G = pickle.load(f)

    # считали какие воксели матрицы будем использовать
    with open('../../labels_for_clasterization.pickle', 'rb') as f:
        L = pickle.load(f)

    L = list(map(int, L))
    G = G.reshape(G.shape[0], 8196, 3)  # that is not fact that we will always have 3 dimensional matrix

    print("read_data was run")
    return G, L

def G_change(G_new,L):
    G_selected=G_new[:,L]
    u,d,v=np.linalg.svd(G_selected)
    list_of_selected_coef=(np.cumsum(d)/sum(d)>0.95).tolist()
    number_of_selected_coef=list_of_selected_coef.index(True)
    #почему такое разложение дает какой-то адеватный результат
    G_selected_decreased= np.matmul(u[:,0:number_of_selected_coef+1].T , G_selected)
    print("G_change was run")
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
    print("G_clasterization was run")
    return predicted



def mean_activity_acc_to_clasterization(x,predicted, clusters_number):
    '''
    x - data
    predicted - which voxel is in one group with others
    clusters_number - the number of clasters
    '''
    x=np.array(x)
    list_of_mean_activity = []
    list_of_clusters_and_their_columns = []

    for k in range(clusters_number):
        list_of_clusters_and_their_columns.append([i for i, x in enumerate(predicted) if x == k])

    for i in range(clusters_number):
        voxels_of_cluster = list_of_clusters_and_their_columns[i]
        sum_of_voxels_of_cluster=sum(x[voxels_of_cluster])
        num_of_voxels_of_cluster=len(voxels_of_cluster)

        list_of_mean_activity.append(sum_of_voxels_of_cluster/num_of_voxels_of_cluster)

    print("mean_activity_acc_to_clasterization was run")
    return list_of_mean_activity

def my_filter(Nf, rate_1,rate_2, A, order):
    b_alpha, a_alpha = sp.butter(order, Wn = np.array([rate_1/(Nf), rate_2/(Nf)]), btype='bandpass')
    alpha = y = sp.lfilter(b_alpha, a_alpha, A)

    print("my_filter was run")
    return alpha

def eneloper(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    print("eneloper was run")
    return amplitude_envelope


def three_envelope_of_signal(signal, order, Nf):
    theta = my_filter(Nf,4, 7, signal, order)
    alpha = my_filter(Nf,8, 15, signal, order)
    beta = my_filter(Nf,16, 31, signal, order)

    print("three_envelope_of_signal was run")
    return eneloper(theta), eneloper(alpha), eneloper(beta)

"""START"""

initial_data=prepare_data(1000)
G,L=read_data()
predicted_clasters = G_clasterization(G)
clasters_number=10
list_of_mean_activity=mean_activity_acc_to_clasterization(initial_data, predicted_clasters, clasters_number)
filter_order=6
Fs=100
Nf=Fs/2

#получаем лист размера 10 на колчиесвто сэмплов, тут было бы уместно проделать даунсемплинг
# вроде как это можно сделать с помощью scipy.signal.decimate пока что делать не буду
outcome_data=[]
for i in range(clasters_number):
    a_1,a_2,a_3 = three_envelope_of_signal(list_of_mean_activity[i], filter_order,Nf)
    outcome_data.append(a_1)
    outcome_data.append(a_2)
    outcome_data.append(a_3)


print(len(outcome_data))
'''
fig, (ax1, ax2,ax3,ax4) = plt.subplots(4)
fig.suptitle('Vertically stacked subplots')
ax1.plot(np.arange(len(outcome_data[15])),outcome_data[2])
ax2.plot(np.arange(len(outcome_data[16])),outcome_data[2])
#plt.plot(np.arange(len(outcome_data[2])),outcome_data[2])
ax1.plot(np.arange(len(outcome_data[17])),outcome_data[2])
ax2.plot(np.arange(len(outcome_data[18])),outcome_data[2])
#plt.plot(np.arange(len(outcome_data[2])),outcome_data[2])
plt.show()
'''

fig, axs = plt.subplots(6, 5)
for i in range(6):
    for j in range(5):
        axs[i, j].plot(np.arange(len(outcome_data[i*5+j])),outcome_data[i*5+j])

plt.show()
