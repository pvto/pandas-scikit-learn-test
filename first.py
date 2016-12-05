# Data pre-processing and ml with Python, pandas, scikit-learn.
# (Explorative code).

# This uses proprietary data that is not meaningful.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing





# 1) Build some numerical columns from the data

def mask(df, f):
  return df[f(df)]

pd.DataFrame.mask = mask
df = pd.read_csv('data/all-applications-operative-pub-20161031.csv', sep=';',parse_dates=[8,9,10,11,12,13])

#df.head()[df.verdictDays.notnull()]
df['verdictDays'] = (df.verdictGivenDate - df.submittedDate) / np.timedelta64(1, 'D')

# correct negative verdictDays
vdMedian = df.loc[(df.verdictDays.notnull()) & (df.verdictDays > 0), 'verdictDays'].quantile(q=0.5)
df.ix[(df.verdictDays.notnull()) & (df.verdictDays < 0), 'verdictDays'] = vdMedian
# create class variable for classification
df['verdictClass'] = np.where(df.verdictDays.notnull(), np.where(df.verdictDays <= 7, 1, np.where(df.verdictDays <= 14, 2, 3)), np.nan)

#df.ix[df.verdictDays <= 7,'verdictClass'] = 1
#df.ix[(df.verdictDays) <= 14 & (df.verdictDays > 7),'verdictClass'] = 2
#df.ix[df.verdictDays > 14,'verdictClass'] = 3


verdicts = df[df.verdictDays.notnull()]
v = verdicts.loc[:, ['municipalityId','permitType','operations','verdictDays','verdictClass']]
v.groupby('municipalityId').count()

# build index of municipalities, with count of applications per mun.
vmun = v.groupby('municipalityId').count().loc[:,['permitType']]
vmun['applicCount'] = vmun.permitType
vmun['municipalityId'] = vmun.index
vmun = vmun.loc[:,['municipalityId','applicCount']]
# build index of operations, with count of operation in data
vopers = v.groupby('operations').count().loc[:,['permitType']]
vopers['operations'] = vopers.index
vopers['operCount'] = vopers.permitType
vopers = vopers.loc[:,['operations','operCount']]


# merge secondary info tables with data
v_ac = pd.merge(v, vmun, on='municipalityId')
v_ac = pd.merge(v_ac, vopers, on='operations')
# create operations-in-application column
v_ac['opersinappl'] = v_ac['operations'].apply(lambda x: x.count(",") + 1)

# with low-frequency municipalities, group together...
# ie. set 'munip' var, with value =999 (for <100 appls for munip), or =munipId

v_ac['munip'] = np.where(v_ac.applicCount < 100, 999, v_ac.municipalityId)

# cleanup extra columns
v_ac = v_ac.loc[:, ['munip','permitType','operations', 'opersinappl', 'verdictDays','applicCount', 'operCount', 'verdictClass']]


# 2) Create dataset of floats with three binary class columns extracted from 'verdictClass'
# VerdictClass has values 1) <7d verdict time, 2) < 14d, 3) for others

data = v_ac.loc[:, ['opersinappl', 'applicCount', 'operCount']]
data['opersinappl'] = data['opersinappl'].apply(float)
data['applicCount'] = data['applicCount'].apply(float)
data['operCount'] = data['operCount'].apply(float)
superv = v_ac.loc[:,'verdictClass']
enc = preprocessing.OneHotEncoder()
enc.fit(superv.values.reshape(-1,1))
vv = enc.transform(superv.values.reshape(-1,1))


# 3) ml 1) MPL classifier

from sklearn.neural_network import MLPClassifier
train = int(len(data) * 0.7)
TR = data[:train].get_values()
TRs = vv[:train].toarray()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1)
clf.fit(TR, TRs)
pred = clf.predict(data[train:].get_values())

'''
>>> pred[:, 2].sum()
6366
>>> pred[:, 1].sum()
1728
>>> pred[:, 0].sum()
10
>>> vv[train:, 2].sum()
5269.0
>>> vv[train:, 1].sum()
513.0
>>> vv[train:, 0].sum()
584.0
>>> (pred[:, 2] - vv[train:, 2].toarray().flatten()).sum()
1097.0
>>> (pred[:, 1] - vv[train:, 1].toarray().flatten()).sum()
1215.0
>>> (pred[:, 0] - vv[train:, 0].toarray().flatten()).sum()
-574.0
'''
#misc testing / plotting, commented out


#plt.scatter(v_ac['munip'],v_ac['verdictClass'])
#plt.show()
#plt.scatter(v_ac['applicCount'],v_ac['verdictClass'])
#plt.show()
#plt.scatter(v_ac['opersinappl'],v_ac['verdictDays'])
#plt.show()


#v_ac.loc[v_ac.munip==999, ['munip', 'verdictDays']].describe()
#v.loc[(v.municipalityId==1123) & (v.operations=='masto-tms'), :].describe()
#v.loc[(v.municipalityId==1123) & (v.operations=='masto-tms'), :].quantile(q=0.9)

