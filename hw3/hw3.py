import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import random
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay

url_data = (r'https://raw.githubusercontent.com/carlson9/KocPythonFall2021/main/inclass/10ML/cses4_cut.csv')

data = pd.read_csv(url_data,sep=',', engine='python')


#check whether unknown data is high in the columns

included_columns = ["age"]

list1=["D2003","D2010","D2015","D2021","D2022","D2023","D2026", "D2028"] #97, 98, 99

list2=["D2002","D2004","D2005","D2006","D2007","D2008","D2009","D2012", "D2013", "D2014", "D2017", "D2018", "D2019", "D2020", "D2024", "D2025", "D2031"] #7, 8, 9

list3=["D2011", "D2016", "D2027", "D2029", "D2030"] #997

for i in list1:
    a = len(data[data[i] >= 97]) / len(data)
    if a < 0.50:
        included_columns.append(i)


for i in list2:
    a = len(data[data[i] >= 7]) / len(data)
    if a < 0.50:
        included_columns.append(i)


for i in list3:
    a = len(data[data[i] >= 997]) / len(data)
    if a < 0.50:
        included_columns.append(i)

data_new = data[included_columns]


# D2027, D2028 çok fazla kategori var ele.
data_new = data_new.drop(['D2027','D2028'], axis=1)

for i in data_new.columns:
    if i in list1:
        data_new.loc[data_new[i]>=97,i] = 99
    elif i in list2:
        data_new.loc[data_new[i]>=7,i] = 9
    elif i in list3:
        data_new.loc[data_new[i]>=997,i] = 999
        
data_new.loc[data_new["D2003"]==96,"D2003"] = 99

categorical_columns = ["D2003","D2004","D2005","D2006","D2010","D2013","D2014","D2029","D2031"]

ordered_columns = ["D2020","D2024","D2025"]

X = pd.get_dummies(data_new, columns = categorical_columns)

X_1 = X.drop(['D2003_99','D2004_9','D2005_9','D2006_9','D2010_9','D2010_99','D2013_9','D2014_9','D2029_999','D2031_9'], axis=1)

for i in range(len(X_1)):
    if ((X_1["D2023"][i] == 99) & (X_1["D2022"][i] < 99)):
        # X_1["D2023"][i] = random.randint(0, X_1["D2022"][i])
        X_1.loc[i, 'D2023'] = random.randint(0, X_1["D2022"][i])

for i in range(len(X_1)):
    if ((X_1["D2022"][i] == 99) & (X_1["D2021"][i] < 99)):
        #X_1["D2022"][i] = random.randint(0, X_1["D2021"][i])
        X_1.loc[i, 'D2022'] = random.randint(0, X_1["D2021"][i])

for i in range(len(X_1)):
    if ((X_1["D2021"][i] == 99) & (X_1["D2022"][i] < 99)):
        #X_1["D2021"][i] = random.randint(X_1["D2022"][i], X_1["D2022"][i] + 4)
        X_1.loc[i, 'D2021'] = random.randint(X_1["D2022"][i], X_1["D2022"][i] + 4)

for i in range(len(X_1)):
    if ((X_1["D2023"][i] == 99) & (X_1["D2022"][i] < 99)):
        #X_1["D2023"][i] = random.randint(0, X_1["D2022"][i])
        X_1.loc[i, 'D2023'] = random.randint(0, X_1["D2022"][i])

y = data["voted"]
y.replace({True: 1, False: 0}, inplace=True)

print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_1, y, random_state=1)

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred)
print("Accuracy of Decision Tree:",accuracy_dt)
auc_dt = roc_auc_score(y_test, y_pred)
print("AUC of Decision Tree:",auc_dt)


rf = RandomForestClassifier(n_estimators = 10, random_state = 0)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
print("Accuracy of Random Forest:",accuracy_rf)
auc_rf = roc_auc_score(y_test, y_pred)
print("AUC of Random Forest:",auc_rf)


xgboost = XGBClassifier(use_label_encoder = False, eval_metric = "auc")
xgboost.fit(X_train,y_train)
y_pred = xgboost.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred)
print("Accuracy of XGBoost:",accuracy_xgb)
auc_xgb = roc_auc_score(y_test, y_pred)
print("AUC of XGBoost:",auc_xgb)

ConfusionMatrixDisplay.from_estimator(xgboost, X_test, y_test)

over = SMOTE(sampling_strategy=0.5)
X_2, y_1 = over.fit_resample(X_1, y)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_2, y_1, random_state=1)

xgboost_2 = XGBClassifier(use_label_encoder = False, eval_metric = "auc")
xgboost_2.fit(X_train_1,y_train_1)
y_pred = xgboost_2.predict(X_test_1)
accuracy_xgb = accuracy_score(y_test_1, y_pred)
print("Accuracy of XGBoost with Oversampling:",accuracy_xgb)
auc_xgb = roc_auc_score(y_test_1, y_pred)
print("AUC of XGBoost with Oversampling:",auc_xgb)


ConfusionMatrixDisplay.from_estimator(xgboost_2, X_test_1, y_test_1)



