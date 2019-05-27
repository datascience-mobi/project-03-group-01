import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib as plt

genes = ['gene' + str(i) for i in range (1,101)]

wt= ['wt'+str (i) for i in range (1,6)]
ko = ['ko' + str (i) for i in range(1,6)]

data = pd.DataFrame (columns = [*wt, *ko], index = genes)
data.loc [genes,'wt1' :'wt5']= np.random.normal(loc=rd.randrange(10,1000), scale=rd.randint(1,100), size=5)
data.loc [genes,'ko1':'ko5']= np.random.normal(loc=rd.randrange(10,1000), scale=rd.randint(1,100), size=5)
print (data.head())
print(data.shape)
scaled_data = preprocessing.scale(data.T)
pca= PCA()
pca.fit(scaled_data)
pca_data=pca.transform(scaled_data)