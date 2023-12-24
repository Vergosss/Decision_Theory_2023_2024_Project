#libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier#Algorithmos knn
from sklearn.preprocessing import MinMaxScaler#kanonikopoihsh dedomenon
from sklearn.model_selection import cross_val_score#gia to cross validation
import numpy as np#mathimatika gia tis metrikes
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
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
#######--------erotima2-----------#####
#splitting the dataset
X = df[numericals]
y = df['Class']

####---K-Nearest Neighbors------###
scores =[]
margin = range(3,16)
for k in margin:
	knn = KNeighborsClassifier(n_neighbors=k)
	score = cross_val_score(knn,X,y,cv=5,scoring='accuracy')#to score einai ena dianysma diastasis 5 diladi 5 times.
	scores.append(np.mean(score))#o mesos aytou tou dianysmatos(athrisma stixion/5) = apodosi. #vazo to skor tou ekastote k sth lista

plt.plot(margin,scores,'x')#kai tin kano visualize
plt.xlabel('values of k')
plt.ylabel('Accuracy score')
plt.show()
print('5-Fold cross validation knn optimal k accuracy:',max(scores))
######---- NAIVE BAYES-----#####

naive_bayes = GaussianNB()
print('5-Fold cross validation naive bayes optimal k accuracy: ',np.mean(cross_val_score(naive_bayes,X,y,cv=5,scoring='accuracy')))#ypotheto to accuracy einai to default metric
scores = []
######---SVM classifier----###
for c in range(1,201,5):
	svc = SVC(kernel='rbf',C=c)
	score = cross_val_score(svc,X,y,cv=5,scoring='accuracy')
	scores.append(np.mean(score))

plt.plot(range(1,201,5),scores,'x')
plt.xlabel('values of C')
plt.ylabel('accuracy score')
plt.show()
##
print('5-fold cross validation svm optimal C accuracy ',max(scores))
c=26
scores = []
for g in np.arange(0.0,11.0,0.5):
	svc = SVC(kernel='rbf',C=c,gamma=g)
	score = cross_val_score(svc,X,y,cv=5,scoring='accuracy')
	scores.append(np.mean(score))

plt.plot(np.arange(0.0,11.0,0.5),scores,'x')
plt.xlabel('values of gamma')
plt.ylabel('accuracy score')
plt.show()
##
print('5-fold cross validation svm optimal gamma accuracy ',max(scores))
