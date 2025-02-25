
#%%IMPORT MODULES
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer,IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
#%%Cramer's V function https://www.statology.org/interpret-cramers-v/
def cramers_corrected_stat(confusion_matrix):
    '''
    Calculate Cramer's V statistic for categorical data
    '''
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
#%%
#INITIALIZES CONSTANTS FOR THE PATH
CSV_PATH = os.path.join(os.getcwd(),'dataset','diabetes.csv')
MODEL_PATH =os.path.join(os.getcwd(),'model','model.pkl')
#DATA LOADING
df = pd.read_csv(CSV_PATH)
#%%%
matrix = pd.crosstab(df['Pregnancies'],df['Outcome']).to_numpy()
print(ss.contingency.association(matrix,method='cramer')) #V carmer corelation
print(cramers_corrected_stat(matrix)) #corrected v carmer
#pregnancy have low correlation
#%%EDA
print(df.info())
print(df.describe())#possible outliers
print(df.head(10))#abnormal values of blood pressure, skin thickness, BMI -> value = 0
#Categorical and continuos
cat = ['Pregnancies','Outcome']
cont = df.drop(cat, axis = 1).columns
#%% DATA VISSUALIZATION
#plot continuous data
for i in cont:
    plt.figure()
    sns.displot(df[i],kde = True)#notice abnormality in glucose level
plt.figure()
df.boxplot(rot = 45)
#use plt.show or run in jupyter
# %% Count number of zeroes
print((df['Glucose']==0).sum()) # 5 entries
print((df['BloodPressure']==0).sum()) # 35 entries
print((df['SkinThickness']==0).sum()) # 227 entries
print((df['BMI']==0).sum()) # 11 entries
#%% IMPUTE THE MISSING VALUES
k_imp = KNNImputer(missing_values = 0)
i_imp = IterativeImputer(missing_values = 0, random_state=45)
for column in cont:
    df[column] = k_imp.fit_transform(df[[column]])
print(df.head(10))
#%% CHECK FOR DUPLICATES
print(df.duplicated().sum())
print(df.corr())
# %% CHECK THE CORRELATION BETWEEN CONTINUOUS DATA AND TARGET
for i in cont:
    print(i)
    ir = LogisticRegression()
    ir.fit(df[[i]],df['Outcome'])
    print(ir.score(df[[i]],df['Outcome']))
#All the features in cont has high correlation with our target column
# %% SELECT FEATURES AND LABEL
X = df.loc[:,cont]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=55)
#%% BUILDING PIPELINE
#logistics regression
pipeline_mms_lr = Pipeline([
    ('Min_max_scaler', MinMaxScaler()),
    ('Logisitc_regression', LogisticRegression())
])
pipeline_ss_lr = Pipeline([
    ('Standard_scaler', StandardScaler()),
    ('Logisitc_regression', LogisticRegression())
])
#knn
pipeline_mms_knn = Pipeline([
    ('Min_max_scaler', MinMaxScaler()),
    ('KNeighborsClassifier', KNeighborsClassifier())
])
pipeline_ss_knn = Pipeline([
    ('Standard_scaler', StandardScaler()),
    ('KNeighborsClassifier', KNeighborsClassifier())
])
#Decision tree
pipeline_mms_dt = Pipeline([
    ('Min_max_scaler', MinMaxScaler()),
    ('DecisionTreeClassifier', DecisionTreeClassifier())
])
pipeline_ss_dt = Pipeline([
    ('Standard_scaler', StandardScaler()),
    ('DecisionTreeClassifier', DecisionTreeClassifier())
])
#gradient boost
pipeline_mms_gb = Pipeline([
    ('Min_max_scaler', MinMaxScaler()),
    ('GradientBoostingClassifier', GradientBoostingClassifier())
])
pipeline_ss_gb = Pipeline([
    ('Standard_scaler', StandardScaler()),
    ('GradientBoostingClassifier', GradientBoostingClassifier())
])
#random forest
pipeline_mms_rf = Pipeline([
    ('Min_max_scaler', MinMaxScaler()),
    ('RandomForestClassifier', RandomForestClassifier())
])
pipeline_ss_rf = Pipeline([
    ('Standard_scaler', StandardScaler()),
    ('RandomForestClassifier', RandomForestClassifier())
])
#VC
pipeline_mms_svc = Pipeline([
    ('Min_max_scaler', MinMaxScaler()),
    ('SVC', SVC())
])
pipeline_ss_svc = Pipeline([
    ('Standard_scaler', StandardScaler()),
    ('SVC', SVC())
])
#%% 
#train every pipelines
pipeline_list = [pipeline_ss_lr,pipeline_mms_lr,pipeline_ss_knn,pipeline_mms_knn,
                pipeline_ss_dt,pipeline_mms_dt,pipeline_ss_gb,pipeline_mms_gb,
                pipeline_ss_rf,pipeline_mms_rf,pipeline_ss_svc,pipeline_mms_svc]
for pipe in pipeline_list:
    pipe.fit(X_train,y_train)
#print the score
pipe_score = []
for i, pipe in enumerate(pipeline_list):
    pipe_score.append(pipe.score(X_test,y_test))
print(pipeline_list[np.argmax(pipe_score)])
print(pipe_score[np.argmax(pipe_score)])
# %% Classification report
best_pipe = pipeline_list[np.argmax(pipe_score)]
y_pred = best_pipe.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# %% HYPER PARAMETER TUNING AND GRID SEARCH
pipeline_mms_gb = Pipeline([
    ('Min_max_scaler', MinMaxScaler()),
    ('GradientBoostingClassifier', GradientBoostingClassifier())
])
#parameter tuning
grid_param = [{'GradientBoostingClassifier__loss':['log_loss','exponential']}]
#grid search CV
grid_search = GridSearchCV(pipeline_mms_gb,grid_param, cv=5, verbose=1,n_jobs=1)
grid = grid_search.fit(X_train,y_train)
grid_search.score(X_test,y_test)
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)
# %%
pipeline_ss_rf = Pipeline([
    ('Standard_scaler', StandardScaler()),
    ('RandomForestClassifier', RandomForestClassifier(random_state=42))
])

#parameter tuning
grid_param = [{
    'RandomForestClassifier__criterion':['gini', 'entropy', 'log_loss'],
    'RandomForestClassifier__min_samples_split':[2,3,4],
    'RandomForestClassifier__max_features':['sqrt', 'log2', None],
    'RandomForestClassifier__bootstrap':[True, False]
    }]
#Model performs about the same even after hyperparameter tuning
#grid search CV
grid_search = GridSearchCV(pipeline_ss_rf,grid_param, cv=5, verbose=1,n_jobs=1)
grid = grid_search.fit(X_train,y_train)
grid_search.score(X_test,y_test)
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)
# %% MODEL SAVING
#note that pickle module is required to save as binary
with open(MODEL_PATH,'wb') as file:
    pickle.dump(grid.best_estimator_,file) #save the best model after hyperparameter tuning