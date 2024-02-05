#libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier#Algorithmos knn
from sklearn.preprocessing import MinMaxScaler#kanonikopoihsh dedomenon
from sklearn.model_selection import cross_val_score#gia to cross validation
import numpy as np#mathimatika gia tis metrikes
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import GridSearchCV

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
print(df.describe())
####---K-Nearest Neighbors------###
margin = range(3,16)
scores =[]
gm_scorer = make_scorer(geometric_mean_score, greater_is_better=True)
for k in margin:
	knn = KNeighborsClassifier(n_neighbors=k)
	score = cross_val_score(knn,X,y,cv=5,scoring=gm_scorer)#to score einai ena dianysma diastasis 5 diladi 5 times.
	scores.append(np.mean(score))#o mesos aytou tou dianysmatos(athrisma stixion/5) = apodosi. #vazo to skor tou ekastote k sth lista

plt.plot(margin,scores,'x')#kai tin kano visualize
plt.xlabel('values of k')
plt.ylabel('Accuracy score')
plt.show()
print('5-Fold cross validation knn optimal k gmean:',100*max(scores))

classifier = KNeighborsClassifier()
parameters  = [{'n_neighbors':margin}]
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = gm_scorer, cv = 5)
grid_search = grid_search.fit(X, y)
best_gmean = grid_search.best_score_
best_parameters = grid_search.best_params_
print('Best KNN gmean:',100*best_gmean,' best parameters(k) : ',best_parameters)

######---- NAIVE BAYES-----#####
classifier = GaussianNB()
parameters  = [{}]
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = gm_scorer, cv = 5)
grid_search = grid_search.fit(X, y)
best_gmean = grid_search.best_score_
best_parameters = grid_search.best_params_
print('Best naive bayes gmean:',100*best_gmean,' best parameters() : ',best_parameters)

print('5-Fold cross validation naive bayes optimal gmean: ',100*np.mean(cross_val_score(classifier,X,y,cv=5,scoring=gm_scorer)))#ypotheto to accuracy einai to default metric

######---SVM classifier----###

#######

########


##

classifier = SVC(kernel = 'linear')
parameters  = [{'C': range(1,201,5)}]
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = gm_scorer, cv = 5)
grid_search = grid_search.fit(X, y)
best_gmean = grid_search.best_score_
best_parameters = grid_search.best_params_
print('Best linear SVM gmean:',100*best_gmean,' best parameters : ',best_parameters)


#########

#after finding the optimal C from the previous senario:
classifier = SVC(kernel = 'rbf',C=best_parameters['C'])
parameters  = [{'gamma':np.arange(0.0,11.0,0.5)}]
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = gm_scorer, cv = 5)
grid_search = grid_search.fit(X, y)
best_gmean = grid_search.best_score_
best_parameters = grid_search.best_params_
print('Best rbf SVM gmean:',100*best_gmean,' best parameters : ',best_parameters)