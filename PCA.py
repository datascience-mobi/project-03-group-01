import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib as plt

genes = ["gene" + str(i) in range (1,101)]

wt= ["wt"+str (i) in range (1,6)]
ko = ["ko" + str (i) in range(1,6)]

data = pd.dataframe (columns = [*wt, *ko], index = genes)
data.loc [gene,"wt1" :'wt5']= np.random.normal(loc=rd.randrange(10,1000), scale=rd.randint(1,100), size=5)
data.loc [gene,"ko1" :'ko5']= np.random.normal(loc=rd.randrange(10,1000), scale=rd.randint(1,100), size=5)
print (data.head())
