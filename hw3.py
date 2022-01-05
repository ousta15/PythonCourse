import pandas as pd
from sklearn.preprocessing import OneHotEncoder

url_data = (r'https://raw.githubusercontent.com/carlson9/KocPythonFall2021/main/inclass/10ML/cses4_cut.csv')

data = pd.read_csv(url_data,sep=',', engine='python')

# Encoding of some features

#pd.get_dummies(X.Embarked)
#embarked_dummies = pd.get_dummies(X.Embarked)


#X=pd.concat([df1, embarked_dummies], axis=

#print(data.nunique())

#check whether unknown data is high in the columns

excluded_columns = []

list1=["D2003","D2010","D2015","D2021","D2022","D2023","D2026", "D2028"] #97, 98, 99

list2=["D2004","D2005","D2006","D2007","D2008","D2009","D2012", "D2013", "D2014", "D2017", "D2018", "D2019", "D2020", "D2024", "D2025", "D2031"] #7, 8, 9

list3=["D2011", "D2016", "D2027", "D2029", "D2030"] #997

for i in list1:
    a = len(data[data[i] >= 97]) / len(data)
    if a >= 0.50:
        excluded_columns.append(i)


for i in list2:
    a = len(data[data[i] >= 7]) / len(data)
    if a >= 0.50:
        excluded_columns.append(i)


for i in list3:
    a = len(data[data[i] >= 997]) / len(data)
    if a >= 0.50:
        excluded_columns.append(i)

print(excluded_columns)






