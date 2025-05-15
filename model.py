import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Crop_recommendation.csv')
df

df.head()

df.shape

df.dtypes

df.isnull().sum()

df.dropna(inplace=True)

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train.head()
y_train.head()

model = RandomForestClassifier()
model.fit(x_train,y_train)

predictions=model.predict(x_test)

accuracy = model.score(x_test,y_test)
print("Accuracy:",accuracy)

new_features = [(36,58,25,28.66024,59.31891,8.399136,36.9263)]
predicted_crop = model.predict(new_features)
print("Predicted Crop:",predicted_crop[0])

import pickle
with open('crop_recommendation_model.pkl','wb') as f:
    pickle.dump(model,f)

import pickle
  #Replace protocol=pickle.HIGHEST_PROTOCOL with the desired protocol version
with open('crop_recommendation_model.pkl','wb') as f:
   pickle.dump(model,f, protocol=pickle.HIGHEST_PROTOCOL)


