import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve,mean_squared_error, r2_score,classification_report
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors
from sklearn.svm import SVC
import xgboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


import numpy as np

pokemon=pd.read_csv('pokemon.csv')
combat=pd.read_csv('combats.csv')
pokemonlist=list(pokemon["Name"].unique())

def idgetir(name):
    df=pokemon.set_index("Name")
    id=df.loc[name]["#"]
    return id
col1,col2=st.columns(2)
with col1:
    poke1=st.selectbox("First Pokemon",pokemonlist)
    poke1=poke1.replace("Mega ","")
    poke1=poke1.replace(" ","")
    poke1=poke1.replace(" X","")
    poke1=poke1.replace("♂","")
    poke1=poke1.replace("♀","")
    link1="images/images/"+poke1.lower()+".png"
    st.image(link1)
with col2:
    poke2=st.selectbox("Second Pokemon",pokemonlist)
    poke2 = poke2.replace("Mega ", "")
    poke2 = poke2.replace(" ", "")
    poke2 = poke2.replace(" X", "")
    poke2 = poke2.replace("♂", "")
    poke2 = poke2.replace("♀", "")
    link2 = "images/images/" + poke2.lower() + ".png"
    st.image(link2)

cdf=combat
cdf["Winner"]=cdf["First_pokemon"]==cdf["Winner"]
cdf["Winner"]=np.where(cdf["Winner"],0,1)

y=cdf[["Winner"]]
x=cdf.drop("Winner",axis=1)

trainsec=st.sidebar.slider("Train Size",0,100,80)
trainsec=trainsec/100

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=trainsec)

modelsec=st.sidebar.selectbox("Model Seç",["Decision Tree","Random Forest","LogisticRegression","KNeighborsClassifier","SVC",
                                           "MLPClassifier","GradientBoostingClassifier","LGBMClassifier","CatBoostClassifier"])

if modelsec=="Decision Tree":
    tree=DecisionTreeClassifier()
    model=tree.fit(x_train,y_train)
    cart_tuned = DecisionTreeClassifier(max_depth=5, min_samples_split=30).fit(x_train, y_train)
    y_pred = cart_tuned.predict(x_test)

elif modelsec=="Random Forest":
    agacsec=st.sidebar.number_input("Ağaç Sayısı",value=100)
    forest=RandomForestClassifier(n_estimators=agacsec)
    model=forest.fit(x_train,y_train)

elif modelsec=="LogisticRegression":
    randomState=st.sidebar.number_input("Random State",value=40)
    loj_model  = LogisticRegression(solver="liblinear",random_state=randomState)
    model=loj_model .fit(x_train,y_train)
    y_pred = loj_model.predict(x_test)

elif modelsec == "KNeighborsClassifier":
    knn_model = KNeighborsClassifier()
    model = knn_model.fit(x_train, y_train)
    knn_tuned = KNeighborsClassifier(n_neighbors=11).fit(x_train, y_train)
    y_pred = knn_tuned.predict(x_test)
elif modelsec == "MLPClassifier":
    mlp_model = MLPClassifier()
    model = mlp_model .fit(x_train, y_train)
    mlp_tuned = MLPClassifier(solver="lbfgs", activation="logistic",
                              alpha=5, hidden_layer_sizes=(100, 100)).fit(x_train, y_train)
    y_pred = mlp_tuned.predict(x_test)

elif modelsec == "SVC":
    svm_model= SVC(kernel = "linear")
    model = svm_model.fit(x_train, y_train)
    svm_tuned  = SVC(kernel="linear",C=2).fit(x_train,y_train)
    y_pred = svm_tuned.predict(x_test)

elif modelsec == "LGBMClassifier":
    lgb_model  = LGBMClassifier()
    model = lgb_model.fit(x_train, y_train)
    lgb_tuned = LGBMClassifier(learning_rate=0.1,
                               max_depth=1,
                               n_estimators=40).fit(x_train, y_train)
    y_pred = lgb_tuned.predict(x_test)

elif modelsec == "CatBoostClassifier":
    catb_model   = CatBoostClassifier()
    model = catb_model.fit(x_train, y_train)
    catb_tuned = CatBoostClassifier(depth=8,
                                    iterations=200,
                                    learning_rate=0.01).fit(x_train, y_train)
    y_pred = catb_tuned.predict(x_test)


col1,col2,col3=st.columns(3)
with col1:
    pass
with col2:
    saldir = st.button("Savaş Başlasın")
with col3:
    pass

if saldir:

    sonuc=model.predict([[idgetir(poke1),idgetir(poke2)]])
    st.write(int(sonuc))
    if sonuc==0:
        st.header("Kazanan")
        st.image(link1)
        st.write("Model Skoru",model.score(x_test,y_test))
        st.write("Accuracy Skor",accuracy_score(y_test, y_pred))
    else:
        st.header("Kazanan")
        st.image(link2)
        st.write("Model Skoru",model.score(x_test,y_test))
        st.write("Accuracy Skor",accuracy_score(y_test, y_pred))

