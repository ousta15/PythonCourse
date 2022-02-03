import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df=pd.read_csv(r'https://raw.githubusercontent.com/ousta15/PythonCourse/main/final_project/zingat_usecase_trainset.csv')
df.head()

df[['city', 'distinct','neighbourhood']] = df['path'].str.split('/', 2, expand=True)
df.head()

# Room count

print(df["room_count"].value_counts())

df[['num_room', 'num_livingroom']] = df['room_count'].str.split('+', 1, expand=True)

df_room = df[(df["room_count"]!="1149+0") & (df["room_count"]!="-")]

df_room["num_room"] = df_room["num_room"].astype("int32")
df_room["num_livingroom"] = df_room["num_livingroom"].astype("int32")
df_room["total_room"] = df_room["num_room"] + df_room["num_livingroom"]

df.loc[df["property_type"] == "Yalı Dairesi" , "property_type"] = "Köşk / Konak / Yalı"

ave_num_room = round(df_room.groupby('property_type')['num_room'].median())

ave_num_livingroom = round(df_room.groupby('property_type')['num_livingroom'].median())

df["ave_num_room"] = ave_num_room[df["property_type"]].reset_index(drop=True)
df["ave_num_livingroom"] = ave_num_livingroom[df["property_type"]].reset_index(drop=True)

df.loc[(df["num_room"]=="1149") | (df["room_count"]=="-"),"num_room"] = df["ave_num_room"]
df.loc[(df["num_room"]=="1149") | (df["room_count"]=="-"),"num_livingroom"] = df["ave_num_livingroom"]

df["num_room"] = df["num_room"].astype("int32")
df["num_livingroom"] = df["num_livingroom"].astype("int32")
df["num_room"].describe()

df["total_room"] = df["num_room"] + df["num_livingroom"]

# Gross and Net Area

print(df["grossm²"].describe())

print(df["netm²"].describe())

print(len(df[(df["grossm²"]<=5) & (df["netm²"]<=5)]))

df = df[(df["grossm²"]>5) | (df["netm²"]>5)]

print(len(df[df["grossm²"]<df["netm²"]]))

df.loc[df["grossm²"]<df["netm²"],"grossm²"] = df["netm²"]

print(df.sort_values(by = "netm²").head())

df.loc[df["netm²"]<=5,"netm²"] = df["grossm²"]


print(df.groupby(["property_type"])["netm²"].describe())

property_type = df["property_type"].unique().tolist()

df['net_percentile'] = df.groupby('property_type')['netm²'].rank(pct=True)

df['gross_percentile'] = df.groupby('property_type')['grossm²'].rank(pct=True)

df.groupby(["property_type"])["grossm²"].describe()

net_capping_max = df[(df["net_percentile"]<0.999)].groupby("property_type")[["netm²"]].max().iloc[:,0]


df.groupby(["property_type"])["netm²"].describe()

gross_capping_max = df[(df["gross_percentile"]<0.999)].groupby("property_type")[["grossm²"]].max().iloc[:,0]


for i in property_type:
    df.loc[(df["property_type"]==i) & (df["net_percentile"]>=0.999),"netm²"] = net_capping_max[i]
    df.loc[(df["property_type"]==i) & (df["gross_percentile"]>=0.999),"grossm²"] = gross_capping_max[i]


df["area_diff"] = df["grossm²"] - df["netm²"]
df["ave_area"] = (df["grossm²"] + df["netm²"]) / 2


# building age

print(df["building_age"].value_counts())


df.loc[(df["building_age"] == "1") | (df["building_age"] == "2") | (df["building_age"] == "3") |
         (df["building_age"] == "4") | (df["building_age"] == "5"), "building_age"] = "0-5 arası"

df.loc[(df["building_age"] == "31-35 arası") | (df["building_age"] == "36-40 arası") | (df["building_age"] == "40 ve üzeri")
         ,"building_age"] = "30+"

df.loc[(df["building_age"] == "-") ,"building_age"] = "Unknown_building_age"


# floor no

print(df["floor_no"].value_counts())

df.loc[(df["floor_no"] == "1") | (df["floor_no"] == "2") | (df["floor_no"] == "3") | (df["floor_no"] == "4") |
         (df["floor_no"] == "5") | (df["floor_no"] == "6") | (df["floor_no"] == "7") | (df["floor_no"] == "8") |
         (df["floor_no"] == "9") | (df["floor_no"] == "10") | (df["floor_no"] == "11") | (df["floor_no"] == "12") |
         (df["floor_no"] == "13") | (df["floor_no"] == "14") | (df["floor_no"] == "15") | (df["floor_no"] == "16") |
         (df["floor_no"] == "17") | (df["floor_no"] == "18") | (df["floor_no"] == "19"), "floor_no"] = "Ara Kat"

df.loc[(df["floor_no"] == "Kot 1") | (df["floor_no"] == "Kot 2") | (df["floor_no"] == "Kot 3") |
         (df["floor_no"] == "Kot 4"),"floor_no"] = "Bodrum Kat"

df.loc[(df["floor_no"] == "Bahçe katı") | (df["floor_no"] == "Yüksek Giriş") | (df["floor_no"] == "Zemin Kat") |
         (df["floor_no"] == "Kot 4"),"floor_no"] = "Giriş Katı"

df.loc[(df["floor_no"] == "Komple"),"floor_no"] = "Müstakil"

df.loc[(df["floor_no"] == "Teras Kat") | (df["floor_no"] == "Çatı Katı"),"floor_no"] = "En Üst Kat"

df.loc[(df["floor_no"] == "-"),"floor_no"] = "Unknown_floor_no"

# detached_house (new column)

df["detached_house"] = 0
df.loc[df["property_type"] == "Müstakil Ev", "detached_house"] = 1
df.loc[df["property_type"] == "Villa", "detached_house"] = 1
df.loc[df["property_type"] == "Çiftlik Evi", "detached_house"] = 1
df.loc[df["property_type"] == "Köşk / Konak / Yalı", "detached_house"] = 1
df.loc[df["property_type"] == "Çiftlik Evi", "detached_house"] = 1
df.loc[df["floor_no"] == "Müstakil", "detached_house"] = 1
df.loc[df["floor_no"] == "Komple", "detached_house"] = 1


# total floor count

print(df["totalfloorcount"].value_counts())

df.loc[(df["totalfloorcount"] == "1") | (df["totalfloorcount"] == "2") | (df["totalfloorcount"] == "3") |
         (df["totalfloorcount"] == "4") | (df["totalfloorcount"] == "5") | (df["totalfloorcount"] == "6") |
         (df["totalfloorcount"] == "7") | (df["totalfloorcount"] == "8") |
         (df["totalfloorcount"] == "9") | (df["totalfloorcount"] == "10"), "totalfloorcount"] = "0-10"

df.loc[(df["totalfloorcount"] == "-") | (df["totalfloorcount"] == "Çatı Katı"),"totalfloorcount"] = "Unknown_totalfloor"


# Heating Type

print(df["heating_type"].value_counts())

df = df.drop(["heating_type"], axis = 1)

# Bath Count

df["bath_count"].value_counts()

df.loc[(df["bath_count"] == "6 ve üzeri"),"bath_count"] = 6

df.loc[(df["bath_count"] == "-"),"bath_count"] = 1

df["bath_count"] = df["bath_count"].astype("int64")


# Landspace

print(df["landscape"].value_counts())

df = df.drop(["landscape"], axis = 1)

# Car Park

print(df["car_park"].value_counts())

df = df.drop(["car_park"], axis = 1)

# Target Value

df_inflation = pd.read_excel(r'https://raw.githubusercontent.com/ousta15/PythonCourse/main/final_project/inflation.xlsx')

df_inflation['year'] = pd.DatetimeIndex(df_inflation['Date']).year
df_inflation['month'] = pd.DatetimeIndex(df_inflation['Date']).month

df['date'] = pd.to_datetime(df['date'])
df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month

df = df.merge(df_inflation[["year","month","Coef"]],on=["year","month"],how="left")


df['price'] = df['price'].str.replace(r' TRY$', '')
df['price'] = df['price'].astype("int64")
df["price_inf"] = df["price"]*df["Coef"]

df_ist = df[df["city"]=="İstanbul"]
df_izm = df[df["city"]=="İzmir"]

df_ist[["price_inf"]].describe().astype("int32")

df_izm[["price_inf"]].describe().astype("int32")

df_ist['property_percentile'] = df_ist.groupby('property_type')['price_inf'].rank(pct=True)
df_ist = df_ist[(df_ist["property_percentile"]<=0.999) & (df_ist["property_percentile"]>=0.001)]

df_izm['property_percentile'] = df_izm.groupby('property_type')['price_inf'].rank(pct=True)
df_izm = df_izm[(df_izm["property_percentile"]<=0.999) & (df_izm["property_percentile"]>=0.001)]

df_ist[["price_inf"]].describe().astype("int32")

df_izm[["price_inf"]].describe().astype("int32")

df_ist = df_ist.drop(["date","path","price","room_count","Ilan_ID","ave_num_room","ave_num_livingroom","Coef","property_percentile","neighbourhood"], axis = 1)
df_izm = df_izm.drop(["date","path","price","room_count","Ilan_ID","ave_num_room","ave_num_livingroom","Coef","property_percentile","neighbourhood"], axis = 1)

df_ist[["property_type","floor_no","totalfloorcount"]] = df_ist[["property_type","floor_no","totalfloorcount"]].astype(str)
df_izm[["property_type","floor_no","totalfloorcount"]] = df_izm[["property_type","floor_no","totalfloorcount"]].astype(str)

df_ist[["Intercom","earthquake_reg","elevator","children_playground","dressing_room","parents_bathroom"]] = df_ist[["Intercom","earthquake_reg","elevator","children_playground","dressing_room","parents_bathroom"]].replace({"VAR": 1, "YOK": 0})
df_izm[["Intercom","earthquake_reg","elevator","children_playground","dressing_room","parents_bathroom"]] = df_izm[["Intercom","earthquake_reg","elevator","children_playground","dressing_room","parents_bathroom"]].replace({"VAR": 1, "YOK": 0})

categorical_columns = ["property_type","building_age","floor_no","totalfloorcount"]
df_ist = pd.get_dummies(df_ist, columns = categorical_columns)
df_izm = pd.get_dummies(df_izm, columns = categorical_columns)

# İstanbul

df_ist = df_ist.drop("city", axis=1)
df_ist = pd.get_dummies(df_ist, columns = ["distinct"])

y_ist = df_ist['price_inf'].values
X_ist = df_ist.drop(["price_inf"],axis=1)

X_train_ist, X_test_ist, y_train_ist, y_test_ist = train_test_split(X_ist, y_ist, test_size=0.2, random_state=0)


dt = DecisionTreeRegressor(random_state = 0)
dt.fit(X_train_ist,y_train_ist)
y_pred = dt.predict(X_test_ist)
print(r2_score(y_test_ist,y_pred))
print(mean_absolute_percentage_error(y_test_ist, y_pred))
print(mean_absolute_error(y_test_ist, y_pred))

rf = RandomForestRegressor(random_state = 0)
rf.fit(X_train_ist,y_train_ist)
y_pred = rf.predict(X_test_ist)
print(r2_score(y_test_ist,y_pred))
print(mean_absolute_percentage_error(y_test_ist, y_pred))
print(mean_absolute_error(y_test_ist, y_pred))

xgb = XGBRegressor()
xgb.fit(X_train_ist,y_train_ist)
y_pred = xgb.predict(X_test_ist)
print(r2_score(y_test_ist,y_pred))
print(mean_absolute_percentage_error(y_test_ist, y_pred))
print(mean_absolute_error(y_test_ist, y_pred))



param_grid = {
    'bootstrap': [True, False],
    'max_depth': [60, 80, 100, 120],
    'max_features': ["auto", "sqrt", "log2"],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [50, 100, 200]
}


grid_search = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = param_grid, cv = 3, n_jobs=-1)

grid_search.fit(X_ist, y_ist)
print('Best Parameters: ', grid_search.best_params_)

rf = RandomForestRegressor(n_estimators = 50, min_samples_split = 2, min_samples_leaf = 1, max_features = 'sqrt', max_depth = 100,
                           bootstrap = False)
rf.fit(X_train_ist,y_train_ist)
y_pred = rf.predict(X_test_ist)
print(r2_score(y_test_ist,y_pred))
print(mean_absolute_percentage_error(y_test_ist, y_pred))
print(mean_absolute_error(y_test_ist, y_pred))

important_features = pd.Series(data=rf.feature_importances_, index=X_train_ist.columns)
important_features.sort_values(ascending=False,inplace=True)
print(important_features[:10])

# İzmir

df_izm = df_izm.drop("city", axis=1)
df_izm = pd.get_dummies(df_izm, columns = ["distinct"])

y_izm = df_izm['price_inf'].values
X_izm = df_izm.drop(["price_inf"],axis=1)
X_izm.head()

X_train_izm, X_test_izm, y_train_izm, y_test_izm = train_test_split(X_izm, y_izm, test_size=0.2, random_state=0)


dt = DecisionTreeRegressor(random_state = 0)
dt.fit(X_train_izm,y_train_izm)
y_pred = dt.predict(X_test_izm)
print(r2_score(y_test_izm,y_pred))
print(mean_absolute_percentage_error(y_test_izm, y_pred))
print(mean_absolute_error(y_test_izm, y_pred))

rf = RandomForestRegressor(random_state = 0)
rf.fit(X_train_izm,y_train_izm)
y_pred = rf.predict(X_test_izm)
print(r2_score(y_test_izm,y_pred))
print(mean_absolute_percentage_error(y_test_izm, y_pred))
print(mean_absolute_error(y_test_izm, y_pred))


xgb = XGBRegressor()
xgb.fit(X_train_izm,y_train_izm)
y_pred = xgb.predict(X_test_izm)
print(r2_score(y_test_izm,y_pred))
print(mean_absolute_percentage_error(y_test_izm, y_pred))
print(mean_absolute_error(y_test_izm, y_pred))

from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'bootstrap': [True, False],
    'max_depth': [60, 80, 100, 120],
    'max_features': ["auto", "sqrt", "log2"],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [50, 100, 200]
}


grid_search = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = param_grid, cv = 3, n_jobs=-1)

grid_search.fit(X_izm, y_izm)
print('Best Parameters: ', grid_search.best_params_)

rf = RandomForestRegressor(n_estimators = 100, min_samples_split = 2, min_samples_leaf = 1, max_features = 'log2', max_depth = 80,
                           bootstrap = True)
rf.fit(X_train_izm,y_train_izm)
y_pred = rf.predict(X_test_izm)
print(r2_score(y_test_izm,y_pred))
print(mean_absolute_percentage_error(y_test_izm, y_pred))
print(mean_absolute_error(y_test_izm, y_pred))

important_features = pd.Series(data=rf.feature_importances_, index=X_train_izm.columns)
important_features.sort_values(ascending=False,inplace=True)

print(important_features[:10])



