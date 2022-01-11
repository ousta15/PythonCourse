import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,roc_curve
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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

y = data["voted"]
y.replace({True: 1, False: 0}, inplace=True)


print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_1, y, random_state=1)

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred)
print(accuracy_dt)
auc_dt = roc_auc_score(y_test, y_pred)
print(auc_dt)


rf = RandomForestClassifier(n_estimators = 10, random_state = 0)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
print(accuracy_rf)
auc = roc_auc_score(y_test, y_pred)
print(auc)


xgboost = XGBClassifier()
xgboost.fit(X_train,y_train)
y_pred = xgboost.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred)
print(accuracy_xgb)
auc_xgb = roc_auc_score(y_test, y_pred)
print(auc_xgb)


