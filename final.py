import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
import lightgbm as lgb
from xgboost import XGBClassifier
from collections import Counter


warnings.filterwarnings("ignore")

# for value imputaion using mode,label encoding of categorical data and adding a new feature 'id_count'
def preprocess(df): 
    id_dict = dict(Counter(df['patientid'].values.tolist()).items())
    for i in range(len(df)):
        df.loc[i, 'id_count'] = id_dict[df.loc[i, 'patientid']]
    
    df = df.drop(['case_id'], axis=1)
    
    df['City_Code_Patient'].fillna(8, inplace=True)
    df['Bed Grade'].fillna(2, inplace=True)
    
    cdat = []
    ndat = []
    for i, c in enumerate(df.dtypes):
            if c == 'object':
                cdat.append(df.iloc[:, i])
            else:
                ndat.append(df.iloc[:, i])

    cdat = pd.DataFrame(cdat).transpose()
    ndat = pd.DataFrame(ndat).transpose()

    
    le = LabelEncoder()
    for i in cdat:
        cdat[i] = le.fit_transform(cdat[i])

    df = pd.concat([cdat, ndat], axis=1)
    df['Admission_Deposit']=np.log(df['Admisison_Deposit'])
    count_dict = df['Age'].value_counts().to_dict()
    df['Age'].map(count_dict)
        
    return df


df0 = pd.read_csv("train.csv")

factor = pd.factorize(df0['Stay'])
df0.Stay = factor[0]
defn = factor[1]

df0 = preprocess(df0)
df1 = df0.drop(['Stay'], axis=1)

X = df1.values
y = df0['Stay'].values

x_train, x_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model = lgb.LGBMClassifier(max_depth=50, n_estimators=500, objective='multiclass', metrics='multi_logloss', learning_rate=0.08, random_state=42, num_class=11)


model.fit(x_train, Y_train)
print("Train accuracy score", accuracy_score(Y_train, model.predict(x_train)))
print("Test accuracy score", accuracy_score(Y_test, model.predict(x_test)))

fin_df = pd.read_csv("test.csv")

dff = preprocess(fin_df)
x_final = dff.values
y_final = model.predict(x_final)

reversefactor = dict(zip(range(11), defn))
y_final = np.vectorize(reversefactor.get)(y_final)


submission = pd.DataFrame()
submission['case_id'] = fin_df['case_id']
submission['Stay'] = y_final
submission.to_csv('final0.csv', header=True, index=False)



