
#libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
iris = datasets.load_iris()
#reading the xlsx file with pandas
df = pd.read_excel('BreastTissue.xlsx',sheet_name='Data')

print(df)
df = df.drop(['Case #'],axis=1)#afairo thn case giati apla aparithma
print(df)
print('\n-------------------------------------Printing unique class values : \n ')

print(df['Class'].unique())
print('\n-------------------------------------Checking if there are null values \n ')

print(df.isna().sum()) #no null values
print(df.describe())
#
scaler = MinMaxScaler(feature_range=(-1,1))
numericals = ['I0','PA500','HFS','DA','Area','A/DA','Max IP','DR','P']
df[numericals] = scaler.fit_transform(df[numericals])
print('\n-------------------------------------After scaling to [-1,+1] : \n ',df)
#
#assigning classes to categorical attributes
df['Class'] = df['Class'].map({'car': 1, 'fad': 2, 'mas': 3,'gla': 4, 'con': 5 , 'adi': 6})
print('\n-------------------------------------After encoding categorical attributes : \n ',df)

dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)
dataframe['category'] = iris.target
print(dataframe)