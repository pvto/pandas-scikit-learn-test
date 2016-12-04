import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def mask(df, f):
  return df[f(df)]

pd.DataFrame.mask = mask
df = pd.read_csv('/home/paavo/src/pylearn/data/all-applications-operative-pub-20161031.csv', sep=';',parse_dates=[8,9,10,11,12,13])

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

# with low-freuquency municipalities, group together...
# set munip var

v_ac['munip'] = np.where(v_ac.applicCount < 100, 999, v_ac.municipalityId)

# cleanup old columns
v_ac = v_ac.loc[:, ['munip','permitType','operations', 'opersinappl', 'verdictDays','applicCount', 'operCount', 'verdictClass']]


# testing...
v_ac.loc[v_ac['applicCount'] < 100]



#plt.scatter(v_ac['munip'],v_ac['verdictClass'])
#plt.show()
#plt.scatter(v_ac['applicCount'],v_ac['verdictClass'])
#plt.show()
#plt.scatter(v_ac['opersinappl'],v_ac['verdictDays'])
#plt.show()


#v_ac.loc[v_ac.munip==999, ['munip', 'verdictDays']].describe()
#v.loc[(v.municipalityId==1123) & (v.operations=='masto-tms'), :].describe()
#v.loc[(v.municipalityId==1123) & (v.operations=='masto-tms'), :].quantile(q=0.9)

