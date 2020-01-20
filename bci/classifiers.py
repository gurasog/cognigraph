from sklearn.naive_bayes import GaussianNB
import numpy as np

def prepare_data_with_window(data, window_shape):
    index_for_data = len(data) // window_shape
    print(index_for_data)
    cut_data = data[0:index_for_data * window_shape]
    print(len(cut_data))
    new_sample = np.empty(0)
    print(type(new_sample))
    new_data = []

    for i in range(index_for_data * window_shape):
        if i % window_shape != window_shape - 1:
            new_sample = np.append(new_sample, data[i])
            # print(i)
            # print(data[i])
        else:
            new_sample = np.append(new_sample, data[i])
            new_data.append(new_sample.flatten())
            new_sample = []

    new_data = np.array(new_data)

    print("prepare_data_with_window was run")
    return new_data



def run_naive_bayes(x, y):
    gnb = GaussianNB()
    gnb.fit(x, y)
    y_predicted = gnb.predict(x)

    print("run_pca was run")
    return y_predicted

