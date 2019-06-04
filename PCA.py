import pandas as pd
import numpy as np
import random as rd

from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

genes = ['gene' + str(i) for i in range (1,101)]

wt= ['wt'+str (i) for i in range (1,6)]
ko = ['ko' + str (i) for i in range(1,6)]

data = pd.DataFrame (columns = [*wt, *ko], index = genes)
for gene in data.index:
    data.loc[gene,'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
    data.loc[gene,'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10,1000), size=5)
# """ data.loc [genes,'wt1' :'wt5']= np.random.normal(loc=rd.randrange(10,1000), scale=rd.randint(1,100), size=5)
#     data.loc [genes,'ko1':'ko5']= np.random.normal(loc=rd.randrange(10,1000), scale=rd.randint(1,100), size=5)
# print (data.head())
# print(data.shape) """
print (data.head())
scaled_data = preprocessing.scale(data.T)

pca= PCA()
pca.fit(scaled_data)
pca_data=pca.transform(scaled_data)
variation=np.round(pca.explained_variance_ratio_*100,decimals=1)
labels=('PCA'+str(i) for i in range(1,len(variation)+1))

plt.bar(x=range(1,len(variation)+1), height=variation)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)
# print (pca_df.head())
plt.scatter(pca_df.PCA1, pca_df.PCA2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(variation[0]))
plt.ylabel('PC2 - {0}%'.format(variation[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PCA1.loc[sample], pca_df.PCA2.loc[sample]))

plt.show()

#if __name__ == '__main__':
