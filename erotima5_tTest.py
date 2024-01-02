#libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#reading the xlsx file with pandas
df = pd.read_excel('BreastTissue.xlsx',sheet_name='Data')

print(df)
df = df.drop(['Case #'],axis=1)
print(df)
print('\n-------------------------------------Printing unique class values : \n ')

print(df['Class'].unique())
print('\n-------------------------------------Checking if there are null values \n ')

print(df.isna().sum()) #no null values

#
scaler = MinMaxScaler(feature_range=(-1,1))
numericals = ['I0','PA500','HFS','DA','Area','A/DA','Max IP','DR','P']
df[numericals] = scaler.fit_transform(df[numericals])
print('\n-------------------------------------After scaling to [-1,+1] : \n ',df)
#
#assigning classes to categorical attributes
df['Class'] = df['Class'].map({'car': 1, 'fad': 2, 'mas': 3,'gla': 4, 'con': 5 , 'adi': 6})
print('\n-------------------------------------After encoding categorical attributes : \n ',df)

################

X = df[numericals]
y = df['Class']

###
test = SelectKBest(score_func=f_classif,k=4)
fit = test.fit(X,y)
print(fit.scores_)
features = fit.transform(X)
print(features[0:5,:])
print(fit.feature_names_in_)