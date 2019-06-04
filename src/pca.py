import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


def prepare_data(train_list, test_list):
    scaler = preprocessing.StandardScaler()
    #print(train_list)
    # Fit on training set only.
    scaler.fit(train_list)

    # Apply transform to both the training set and the test set.
    train_list = scaler.transform(train_list)
    test_list = scaler.transform(test_list)

    #print(train_img)

    pca = PCA(.95) # how much variance to keep
    pca.fit(train_list)
    print(pca.n_components_)
    train_list = pca.transform(train_list)
    test_list = pca.transform(test_list)
    for i in range(5):
        print(len(train_list[i]))
    print(len(train_list))
    print(len(test_list))
    return train_list, test_list

def reduce_dim():

    genes = ['image' + str(i) for i in range (1,101)]
    # Output: ['gene1', ..., 'gene100']
    #print(genes)

    wt= ['pixel'+str (i) for i in range (1,785)]
    #ko = ['ko' + str (i) for i in range(1,6)]
    # Output ['ko1', ... , 'ko5']

    data = pd.DataFrame (columns = [*wt], index=genes)
    for gene in data.index:
        data.loc[gene,'pixel1':'pixel784'] = np.random.poisson(lam=rd.randrange(10,1000), size=784)
        #data.loc[gene,'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
    #     data.loc [genes,'wt1' :'wt5']= np.random.normal(loc=rd.randrange(10,1000), scale=rd.randint(1,100), size=5)
    #     data.loc [genes,'ko1':'ko5']= np.random.normal(loc=rd.randrange(10,1000), scale=rd.randint(1,100), size=5)
    # print (data.head())
    print(data.shape)
    print(data.head())
    scaled_data = preprocessing.scale(data.T)
    pca = PCA()
    pca.fit(scaled_data)
    pca_data=pca.transform(scaled_data)
    variation=np.round(pca.explained_variance_ratio_*100,decimals=1)
    labels=('PCA'+str(i) for i in range(1,len(variation)+1))

    plt.bar(x=range(1,len(variation)+1), height=variation)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    pca_df = pd.DataFrame(pca_data, index=[*wt], columns=labels)
    print (pca_df.head())
    print(pca_df.shape)
    plt.scatter(pca_df.PCA1, pca_df.PCA2)
    plt.title('My PCA Graph')
    plt.xlabel('PC1 - {0}%'.format(variation[0]))
    plt.ylabel('PC2 - {0}%'.format(variation[1]))

    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PCA1.loc[sample], pca_df.PCA2.loc[sample]))

    plt.show()


if __name__ == '__main__':
    reduce_dim()