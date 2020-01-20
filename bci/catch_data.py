from pylsl import StreamInlet, resolve_stream, resolve_streams
import numpy as np
import pickle

def prepare_data(time):
    a = resolve_streams()
    inlet = StreamInlet(a[0])
    print(a[0].name())
    data=[]
    while len(data) <= time:
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