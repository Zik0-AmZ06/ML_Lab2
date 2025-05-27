# Lab21 : Prédiction Crédit Logement
# Réalisé par : Zak AMZ EMSI 2023/2024
# Réference :



import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Step 1 : DataSet
dt=pd.read_csv("./datasets/train.csv")
print(dt.head())
print(dt.info())
print(dt.isna().sum())
    #Data transformation
"""
Remplacement des valeures manquantes
add Codeadd Markdown
Nous allons remplacer les variables manquantes categoriques par leurs modes
Nous allons remplacer les variables manquantes numériques par la médiane
"""

def trans(data):
    for c in data.columns:
        if data[c].dtype=='int64' or data[c].dtype=='float64':
            data[c].fillna(data[c].median(),inplace=True)
        else:
              data[c].fillna(data[c].mode()[0],inplace=True)

trans(dt)
print(dt.isna().sum())

var_num=dt[["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]]
print(var_num.describe())
var_cat=dt[["Loan_Status","Gender","Married","Dependents","Education","Self_Employed","Property_Area","Credit_History"]]
var_cat=pd.get_dummies(var_cat,drop_first=True)
print(var_cat.head())
transformed_dataset=pd.concat([var_cat,var_num],axis=1)
transformed_dataset.to_csv("./datasets/transformed.csv")

# split dataset on target y and features x
y=transformed_dataset["Loan_Status_Y"]
x=transformed_dataset.drop("Loan_Status_Y",axis=1)

# train 80% test 20% split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)


# Data Visualization
print(dt["Loan_Status"].value_counts())
print(dt["Loan_Status"].value_counts(normalize=True)*100)
fig = px.histogram(dt, x="Loan_Status",title='Crédit accordé ou pas', color="Loan_Status",template= 'plotly_dark')
fig.show(font = dict(size=17,family="Franklin Gothic"))

fig = px.pie(dt, names="Dependents",title='Dependents',color="Dependents",template= 'plotly_dark')
fig.show(font = dict(size=17,family="Franklin Gothic"))


# Step 2 : Model

model = LogisticRegression()

# Step 3 : Train
model.fit(x_train,y_train)

# Step 4 : Test
print("model accuracy",model.score(x_test,y_test)*100,"%")
