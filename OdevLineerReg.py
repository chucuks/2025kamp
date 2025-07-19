import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df=pd.read_csv('OdevLineerReg.csv')
#print(df.columns)
#print(df.describe())
df['Real GDP (Percent Change)'] = df['Real GDP (Percent Change)'].ffill()
#tarihler ilgili değil diğerleri fazla eksik veriden
df.drop(["Year","Month","Day","Federal Funds Upper Target","Federal Funds Lower Target","Federal Funds Target Rate"], axis=1, inplace=True) 
#df.info()
#print(df.corr())
from sklearn.model_selection import train_test_split
X = df.drop('Inflation Rate', axis=1)
y = df['Inflation Rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from lazypredict.Supervised import LazyRegressor
"""reg= LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models) #öğrenilenlerden en iyisi Elasticnetcv olarak gözüküyor adjusted R^2 ile 0.59 genel en iyi extratreesreg 0.95"""
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
"""elastic = ElasticNetCV(cv=5, random_state=42)  
elastic.fit(X_train, y_train)  
y_pred_elastic_cv = elastic.predict(X_test)  
elastic_cv_r2 = r2_score(y_test, y_pred_elastic_cv)  
elastic_cv_adjusted_r2 = 1 - (1 - elastic_cv_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1) 
print("ElasticNetCV Regression R^2 Score:", elastic_cv_r2)
print("ElasticNetCV Adjusted R^2 Score:", elastic_cv_adjusted_r2)  """
df= pd.read_csv("dataset/OdevLineerReg2.csv")
#address gereksiz diye freq,purpose bir unique var diye Location hem çok unique hem sadece bir tane olan yerler var
df.drop(["Rent_per_sqft","Rent_category","Posted_date","Latitude","Longitude","Address","Frequency","Purpose","Location"], axis=1, inplace=True)
type_mean=df.groupby("Type")["Rent"].mean()
df["Type"] = df["Type"].map(type_mean)
df['Furnishing'] = df['Furnishing'].map({'Furnished': 1, 'Unfurnished': 0})
city_mean = df.groupby("City")["Rent"].mean()
df["City"] = df["City"].map(city_mean)
#print(df.head())
#print(df.columns)

X= df.drop('Rent', axis=1)
y = df['Rent']
#print(X.corr())            #birbirleri arasında çok korelasyonlu olan özellikler yok
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#neden lazy çalışmıyor bilmiyorum
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import PolynomialFeatures

"""poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
lineer= LinearRegression()
lineer.fit(X_train_poly, y_train)
y_pred = lineer.predict(X_test_poly)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test_poly.shape[1] - 1)
print("Linear Regression R^2 Score:", r2)
print("Linear Regression Adjusted R^2 Score:", adjusted_r2)

elastic = ElasticNet(random_state=42)
elastic.fit(X_train_poly, y_train)
y_pred_elastic = elastic.predict(X_test_poly)
elastic_r2 = r2_score(y_test, y_pred_elastic)
elastic_adjusted_r2 = 1 - (1 - elastic_r2) * (len(y_test) - 1) / (len(y_test) - X_test_poly.shape[1] - 1)
print("ElasticNet Regression R^2 Score:", elastic_r2)
print("ElasticNet Adjusted R^2 Score:", elastic_adjusted_r2)"""