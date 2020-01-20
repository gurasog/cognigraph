from sklearn.decomposition import PCA
import  numpy as np

def run_pca(data):
    data=np.array(data)
    data=data.T
    pca = PCA(n_components=5, svd_solver='full') #как определеить необходимое количество компоненет
    transformed_data = pca.fit_transform(data)
    print("run_pca was run")
    return transformed_data