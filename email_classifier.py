import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('mail_data.csv')
data=df.where(pd.notnull(df),'')

data.loc[data["Category"]=="spam","Category",]=0
data.loc[data["Category"]=="ham","Category",]=1

X = data["Message"]
Y = data["Category"]

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=3)

print(X_test.size)
print(X_train.size)

feature_extractor= TfidfVectorizer(min_df=1, stop_words="english",lowercase=True)

X_train_features= feature_extractor.fit_transform(X_train)
X_test_features= feature_extractor.transform(X_test)

Y_train = Y_train.astype(int)
Y_test= Y_test.astype(int)

print(X_train_features)

model=LogisticRegression()

model.fit(X_train_features,Y_train)

prediction_on_training_data=model.predict(X_train_features)
accuracy=accuracy_score(Y_train,prediction_on_training_data)

print(f"Accuracy on train data {accuracy}")

prediction_on_test_data= model.predict(X_test_features)
accuracy2=accuracy_score(Y_test,prediction_on_test_data)

print(f"Accuracy on test data {accuracy2}")

input=["Hi,I heard your exams are coming.Good luck"]
input_features= feature_extractor.transform(input)
prediction=model.predict(input_features)

if  prediction==1:
    print(f"Your input email is :good")
else:
    print("It is spam email")