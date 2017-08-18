import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import scale
from scipy import stats



class DWFF():

  def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
               min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07,
               class_weight=None, presort=False):
    self.criterion=criterion
    self.splitter=splitter
    self.max_depth=max_depth
    self.min_samples_split=min_samples_split
    self.min_samples_leaf=min_samples_leaf
    self.min_weight_fraction_leaf=min_weight_fraction_leaf
    self.max_features=max_features
    self.random_state=random_state
    self.max_leaf_nodes=max_leaf_nodes
    self.min_impurity_split=min_impurity_split
    self.class_weight=class_weight
    self.presort=presort



  def feature_agg(self,Data,size=5):
    clus = FeatureAgglomeration(n_clusters=size).fit(np.array(scale(Data)))
    self.features_clusters = []
    self.features_name = list(Data)
    for i in range(size):
        self.features_clusters.append(np.array(self.features_name)[np.where(clus.labels_ == i)])

  def F_test(self,Data,y,size=5):

   self.AllFs = []
   for i in range(size):
       Domain_features = Data.loc[:, self.features_clusters[i]]
       Fs = [];
       ps = []
       for j in range(len(self.features_clusters[i])):
           features1 = np.array(Domain_features)[:, j]
           lst = [np.array(features1[np.where(y==i)]) for i in range(len(np.unique(y)))]
           F, p = stats.f_oneway(*lst)
           Fs.append(F)
           ps.append(p)
       Fs = np.nan_to_num(Fs)
       Fs[np.where(Fs >= 100)] = 100
       self.AllFs.append(Fs)


  def fit(self, Data, y, size=5,num=5):
    self.features_chosen = []
    for i in range(size):
        self.features_chosen = np.append(self.features_chosen,
                        np.random.choice(self.features_clusters[i], size=num, p=self.AllFs[i] / np.sum(self.AllFs[i])))
    self.clf = DecisionTreeClassifier(random_state=self.random_state,max_features=self.max_features,
    criterion=self.criterion, splitter=self.splitter, max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_leaf_nodes=self.max_leaf_nodes, min_impurity_split=self.min_impurity_split,class_weight=self.class_weight,
            presort=self.presort)
    self.clf.fit(Data.loc[:, self.features_chosen],y)

    return self

  def predict(self,X):

    classification = self.clf.predict(X.loc[:, self.features_chosen])

    return classification

  def score(self,X,y):

      classification = self.clf.predict(X.loc[:, self.features_chosen])

      return np.mean(classification==y)

