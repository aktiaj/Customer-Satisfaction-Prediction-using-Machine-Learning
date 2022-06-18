import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


import seaborn as sns
from scipy import stats
from scipy.stats import boxcox
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix,classification_report
import warnings
warnings.filterwarnings("ignore")
sns.set()

import os
for dirname, _, filenames in os.walk('C:/Users/CK069TX/Desktop/Machine learning/ACT task/ACT India D&A GCC_Data Scientist_Case/train.csv'):
    for filename in filenames:
        print(os.path.join(dirname,'train.csv'))

for dirname, _, filenames in os.walk('C:/Users/CK069TX/Desktop/Machine learning/ACT task/ACT India D&A GCC_Data Scientist_Case/test.csv'):
    for filename in filenames:
        print(os.path.join(dirname,'test.csv'))

data_train=pd.read_csv('C:/Users/CK069TX/Desktop/Machine learning/ACT task/ACT India D&A GCC_Data Scientist_Case/train.csv')
data_test=pd.read_csv('C:/Users/CK069TX/Desktop/Machine learning/ACT task/ACT India D&A GCC_Data Scientist_Case/test.csv')

data_train.head()
data_test.head()

data_train.columns
data_test.columns

print(f"Train data has {data_train.shape[0]} rows and  {data_train.shape[1]} columns.")
print("Distribution of target value:\n")
data_train.satisfaction.value_counts()



print(f"Test data has {data_test.shape[0]} rows and {data_test.shape[1]} columns.")
print("Distribution of target value:\n")
data_test.satisfaction.value_counts()

# Joining two datasets into one whole dataset -1 
data=data_train.append(data_test)
data.head()

#Checking null values -2 
data.isna().sum()
#checking for duplicated values -3
data.duplicated().sum()

# Unique element features -4
data.nunique()

# 5 Describe
data.describe().T

data.satisfaction.unique()


data.loc[data["Customer Type"]=="disloyal Customer","Customer Type"]="disloyal Customer"
data.loc[data["Type of Travel"]=="Business travel","Type of Travel"]="Business Travel"

# VAlue count bar plot of satisfaction column -6
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='satisfaction', data=data, palette=["#f08080","#87cefa"])

#pie chart of satisfield vs nuetral or dissatisfied customers - 7                                                   
plt.subplot(1, 2, 2)
plt.pie(data['satisfaction'].value_counts(), labels=["neutral or dissatisfied","satisfied"], explode=[0, 0.05], autopct='%1.2f%%', shadow=True,colors=["lightcoral","lightskyblue"])
plt.title('satisfaction', fontsize=15)
plt.show()

# 8
categories=['Gender', 'Customer Type','Type of Travel', 'Class','Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness']
# 9
for i in categories:
    plt.figure(figsize=(20,6))
    plt.subplot(1,2,1)
    sns.countplot(x=data[i],palette="Pastel1")

    
plt.subplot(1,2,2)
sns.countplot(x=data[i],hue=data.satisfaction, palette=["#f08080","#87cefa"])
plt.show()

                                                 

#Removing [ 'Gender' , 'Customer_Type' , 'Type_of_Travel' , 'Class' ] features from categories list - 10
  
for i in ['Gender','Customer Type','Type of Travel','Class']:
  categories.remove(i)
# calculating the mean of categories and sorting them accordingly to descending order -11
data[categories].mean().sort_values(ascending=False)


# 12 error
total = float(len(data))
ax = data[categories].mean().sort_values(ascending=False).plot(kind="barh",ylabel="Features",colormap="Pastel1",xticks=[0, 5, 0.5],figsize=(14,6))
plt.title('Average satisfaction ratings of services', fontsize=16)
for p in ax.patches:
    count = '{:.1f}'.format(p.get_width())
    x, y = p.get_x() + p.get_width()+0.15, p.get_y()
    ax.annotate(count, (x, y), ha='right')
plt.show()




plt.figure(figsize=(10,10))
sns.catplot(y='Departure/Arrival time convenient',col='Type of Travel',x ='Customer Type',
            hue='satisfaction',row='Class', data=data, kind= 'bar',palette='Pastel1')
plt.show()


#13
def percentage(x):
  return round(100*x.count()/data.shape[0],2)
table1=data.pivot_table(index=["Gender"],columns=["satisfaction"],aggfunc={"satisfaction":["count",percentage]},fill_value=0)
table1


#ax = data[categories].std().sort_values(ascending=False).plot(kind="barh",ylabel="Features",colormap="Pastel1",figsize=(20,10))
#plt.title('Standard deviation of service ratings', fontsize=16)
#for p in ax.patches:
#    count = '{:.1f}'.format(p.get_width())
 #   x, y = p.get_x() + p.get_width()+0.05, p.get_y()
  #  ax.annotate(count, (x, y), ha='right')
#plt.show()




def percentage(x):
  return round(100*x.count()/data.shape[0],2)
table1=data.pivot_table(index=["Gender"],columns=["satisfaction"],aggfunc={"satisfaction":["count",percentage]},fill_value=0)
table1


gender="female"
for i,j,k,l in table1.values:
  print("Satisfaction rate for {} is: {:.3f}".format(gender,j/(i+j)))
  gender="male"

#  satisfaction rates of women and men -14
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pie(data.loc[data.Gender=="Female",'satisfaction'].value_counts(), labels=["neutral or dissatisfied","satisfied"], explode=[0, 0.05], autopct='%1.2f%%', shadow=True,colors=["lightcoral","lightskyblue"])
plt.title('Satisfaction (Female)', fontsize=15)

plt.subplot(1, 2, 2)
plt.pie(data.loc[data.Gender=="Male",'satisfaction'].value_counts(), labels=["neutral or dissatisfied","satisfied"], explode=[0, 0.05], autopct='%1.2f%%', shadow=True,colors=["lightcoral","lightskyblue"])
plt.title('Satisfaction (Male)', fontsize=15)



#15
data.pivot_table(index=["Customer Type","Class"],columns=["satisfaction"],aggfunc={"satisfaction":["count",percentage]})

ax = data.pivot_table(index=["Customer Type","Class"],columns=["satisfaction"],aggfunc={"satisfaction":"count"}).plot(kind="barh",figsize=(24,6))
plt.title('Satisfaction based on Customer Type and Class', fontsize=16)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_width()/total)
    x,y  = p.get_x() + p.get_width()+1000, p.get_y()
    ax.annotate(percentage, (x, y),ha='right')
plt.show()



# graph showing that Gender is not a discriminative factor in scores -16
ax = pd.crosstab([data["Gender"], data["Customer Type"]],data["Inflight wifi service"],
            rownames=['Gender ', " Customer Type"],
            colnames=["Inflight wifi service"],
            dropna=False).plot(kind="bar",figsize=(30,6),rot=0)
plt.title('Inflight wifi service ratings based on Gender and Customer Type', fontsize=16)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x, y = p.get_x() + p.get_width(), p.get_height()
    ax.annotate(percentage, (x, y),ha='right')
plt.show()

# Business class passengers seem to give more points to the food and drink service. -17
ax = pd.crosstab(data["Class"],data["Food and drink"],
            rownames=['Class '],
            colnames=['Food and drink'],
            dropna=False).plot(kind="bar",figsize=(30,6),rot=0)
plt.title('Food and drink service points based on Class', fontsize=16)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='right')
plt.show()




# Numeric features visualization -18



numerics=['Departure Delay in Minutes', 'Arrival Delay in Minutes','Flight Distance',"Age"]
sns.PairGrid(data[[*numerics,"satisfaction"]],hue="satisfaction")
plt.show()



#checking outliers -19
sns.boxplot(x="Age",y="satisfaction",data=data)
plt.show()

plt.figure(figsize=(20, 6))
for i,j in enumerate(numerics):
  plt.subplot(1,len(numerics),i+1)
  sns.boxplot(data[j]) #- there are some outliers 
  
# Arrival in delay and departure in delay  - 20
fig, ax = plt.subplots(1,len(numerics),figsize=(20,5))
fig.suptitle("Distribution of numeric features",y=1)
for i,j in enumerate(numerics):
  sns.distplot(x=data[j],ax=ax[i])
  ax[i].set_xlabel(j)
fig.tight_layout(pad=1.5)




sns.regplot(x=data['Arrival Delay in Minutes'],y=data['Departure Delay in Minutes'], fit_reg= False)
plt.show()


data[['Arrival Delay in Minutes','Departure Delay in Minutes']].corr()

#Dropping columns which are not useful -21
data.drop(["Unnamed: 0","id"],axis=1,inplace=True)
data_backup=data.copy()
data.head()
data.drop(["Arrival Delay in Minutes","Departure Delay in Minutes"],axis=1,inplace=True)

#Correlation graph -22
plt.figure(figsize=(22,10))
sns.heatmap(data.corr(), vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True,annot=True,fmt='.2f',)
plt.show()

#Correlation with online boarding services -23
data_temp=data.copy()
data_temp["satisfaction"]=data_temp["satisfaction"].map({"satisfied":1,"neutral or dissatisfied":0})
data_temp.corr()['Online boarding'].sort_values().drop(['Online boarding','satisfaction']).plot(kind='barh',title="Correlation with Online Boarding service")
plt.show()

# range distributed to online boarding decreases and its score increases- 24
sns.boxplot(x=data['Inflight wifi service'], y = data_temp['Online boarding'])
plt.show()







# PRE-PROCESSING

# all categories -25
data[["Gender","Customer Type","Type of Travel","Class","satisfaction"]].head()


#mapping ordinal features -26
data["Class"] = data["Class"].map({'Business':2, 'Eco Plus':1, 'Eco':0})
data["satisfaction"]=data["satisfaction"].map({"satisfied":1,"neutral or dissatisfied":0})

#for nominal features,- 27
data_new=pd.get_dummies(data,drop_first=True)
#i use drop_first parameter so my model does not get any confusion by counting some features second time
data_new.reset_index(inplace=True)
data_new.drop("index",axis=1,inplace=True)
data_new.head()


data_new[["Gender_Male","Customer Type_Loyal Customer","Type of Travel_Personal Travel","Class","satisfaction"]].head()




# Outliers detection -28

df_local=data_new.copy()
#temp = df_local.drop("satisfaction", axis=1)
#local_outlier = LocalOutlierFactor(n_neighbors=2).fit_predict(temp)
#outlier_local=list(np.where(local_outlier == -1)[0])
#del temp
#print(f"Outlier Count: {len(outlier_local)} \nSample Count: {len(df_local)} \nFraction: {round(len(outlier_local)/len(df_local),3)}")
#df_local=df_local.drop(outlier_local).reset_index(drop=True)



# Feature transformation for "Flight Distance and Age" Columns -29

df_log=df_local.copy()
df_log["Flight Distance"]=np.log(df_log["Flight Distance"])
df_log["Age"]=np.log(df_log["Age"])



# Flight Distance Feature -30
plt.figure(figsize=(20, 12))

plt.subplot(2, 4, 1)
plt.boxplot(df_local['Flight Distance'])
plt.title('Flight Distance')

plt.subplot(2, 4, 2)
plt.boxplot(df_log["Flight Distance"])
plt.title('Flight Distance (Log Transformation)')



plt.subplot(2, 4, 3)
plt.hist(df_local['Flight Distance'])
plt.title('Flight Distance')

plt.subplot(2, 4, 4)
plt.hist(df_log["Flight Distance"])
plt.title('Flight Distance (Log Transformation)')



# For Age Feature -31
plt.figure(figsize=(20, 12))

plt.subplot(2, 4, 1)
plt.boxplot(df_local['Age'])
plt.title('Age')

plt.subplot(2, 4, 2)
plt.boxplot(df_log["Age"])
plt.title('Age (Log Transformation)')

plt.subplot(2, 4, 3)
plt.hist(df_local['Age'])
plt.title('Age')

plt.subplot(2, 4, 4)
plt.hist(df_log["Age"])
plt.title('Age (Log Transformation)')


# Checking Normality of transformed features -32



#for j in ["Flight Distance","Age"]:
 
# transforms=[df_local[j], df_log[j]]
 # processes=["original","log"]
  #for i,k in zip(transforms,processes):
   # print(f"Normality for {j} Feature ({k}):",stats.shapiro(i))



# Splitting the data -33

X_train, X_test, y_train, y_test= train_test_split(df_local.drop("satisfaction",axis=1),df_local["satisfaction"],test_size=0.3,random_state=42)
    
print("Train size:", X_train.shape)    
print("Test size:", X_test.shape)    



# Feature scaling - 34

scaler=MinMaxScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)



8#Cr///e/ating a function that creates a dataframe for testing model performance -35
def model_perf(model,X_train,X_test,y_train,y_test,pred,model_name):
  """Takes the data, returns a dataframe that calculates the performance of the model"""
  cv_results=cross_val_score(model,X_train,y_train,cv=5)
  perf_df=pd.DataFrame({"Mean_CV":np.mean(cv_results),"Std_CV":np.std(cv_results),'Train_Score':model.score(X_train,y_train),"Test_Score":model.score(X_test,y_test),"Precision_Score":precision_score(y_test,pred),"Recall_Score":recall_score(y_test,pred),"F1_Score":f1_score(y_test,pred)},index=[model_name])
  return perf_df

# I have used two ML models which are XGboost and Random forest
 
# Random forest -36

rf=RandomForestClassifier(random_state=42,n_estimators=300).fit(X_train_scaled,y_train)

pred_rf=rf.predict(X_test_scaled)

perf_rf=model_perf(rf,X_train_scaled,X_test_scaled,y_train,y_test,pred_rf,"Random Forest")

perf_rf




# Xgboost - 37

xgb=XGBClassifier(random_state=42, max_depth=9, min_child_weight=3, n_estimators=100) #Tuned parameters (with GridCV)
xgb.fit(X_train_scaled,y_train)

pred_xgb = xgb.predict(X_test_scaled)

perf_xgb=model_perf(xgb,X_train_scaled,X_test_scaled,y_train,y_test,pred_xgb,"XGBoost")
perf_xgb

# Comparing the two models - 38
pd.concat([perf_rf, perf_xgb])






#confusion matrix - 39
plt.figure(figsize=(12, 8))

cf_matrix=confusion_matrix(y_test,pred_rf)

group_names = ["True Negative","False Positive","False Negative","True Positive"]

group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

ax=sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues',xticklabels=["neutral or dissatisfied","satisfied"], yticklabels=["neutral or dissatisfied","satisfied"])

ax.set_xlabel('Predicted Label',fontsize = 15)

ax.set_ylabel('Actual Label',fontsize = 15)
plt.show()




# Important features -40

feature_names = X_train.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 10))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color="red", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()







































































































