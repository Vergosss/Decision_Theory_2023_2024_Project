#libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.stats import f_oneway
from sklearn.model_selection import cross_val_score#gia to cross validation
import numpy as np#mathimatika gia tis metrikes
from sklearn.svm import SVC
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

################

X = df[numericals]
y = df['Class']

#############
p_values = []
#ANOVA test because we have multiple classes
for feature in X.columns:
    #t_stat,p_value = stats.ttest_ind(X[feature][y == 1],X[feature][y == 2],X[feature][y == 3],X[feature][y == 4],X[feature][y == 5],X[feature][y == 6])
    f_stat,p_value = f_oneway(X[feature][y == 1],X[feature][y == 2],X[feature][y == 3],X[feature][y == 4],X[feature][y == 5],X[feature][y == 6])
    p_values.append(p_value)


alpha = 0.05#katofli an kai de xreiazetai epeidh apla diataso ta p values
sorted_features = [x for y, x in sorted(zip(p_values, X.columns),reverse=True)]
sorted_features2 = [y for y, x in sorted(zip(p_values, X.columns),reverse=True)]
print(sorted_features)
print(sorted_features2)

input('...')
gm_scorer = make_scorer(geometric_mean_score, greater_is_better=True)
X_new = df[['HFS','Area','A/DA','DR']]
y_new = df['Class']
classifier = SVC(kernel = 'rbf')
#parameters  = [{'C': range(1,201,5),'gamma':np.arange(0.0,11.0,0.5)}]
parameters = [{'C':[66],'gamma':[2.0]}]
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = gm_scorer, cv = 5)
grid_search = grid_search.fit(X_new, y_new)
best_gmean = grid_search.best_score_
best_parameters = grid_search.best_params_
print('Best gmean:',100*best_gmean,' best parameters : ',best_parameters)
