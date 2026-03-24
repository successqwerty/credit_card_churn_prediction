import pandas as pd
import pickle
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv("data/BankChurners.csv")   

df=df.drop(['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],axis=1)

df['Attrition_Flag']=df['Attrition_Flag'].map({
    "Existing Customer":0,
    "Attrited Customer":1
})

df=pd.get_dummies(df,drop_first=True)

X=df.drop('Attrition_Flag',axis=1)
y=df["Attrition_Flag"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

pickle.dump(scaler,open("Scaler.pkl","wb"))

model=Sequential()
model.add(Dense(16,activation='relu',input_dim=X_train.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train,y_train,epochs=50,batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Accuracy: {accuracy:.4f}")

print("✅ Model trained and saved successfully!")

from sklearn.metrics import confusion_matrix,classification_report
y_pred=model.predict(X_test)
y_pred=(y_pred>0.3)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

model.save("model.h5")
with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)
