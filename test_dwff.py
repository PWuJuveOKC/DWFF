import pandas as pd
import numpy as np
from DWFF import DWFF


Alldat = pd.read_csv('Data/SampleTS_Med.csv')
Alldat1 = Alldat.loc[:, (Alldat != 0).any(axis=0)]
y = np.array(Alldat1['Label'])
Dat = Alldat1.iloc[:,:-2]

#train_idx = np.random.choice(np.arange(0,len(Dat)),500)
train_idx = np.random.rand(len(Dat)) < 0.6
X_train = Dat[train_idx]
y_train = y[train_idx]
X_test = Dat.iloc[~train_idx]
y_test = y[~train_idx]

model = DWFF()
model.feature_agg(Data=X_train)
model.F_test(Data=X_train,y=y_train)

model.fit(Data=X_train,y=y_train,size=5)
model.score(X_test,y_test)

