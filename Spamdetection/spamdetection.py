import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('spam.csv', encoding='latin-1')
df=df[['v1','v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
x=df['message']
y=df['label']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# convernt into numeric  form
vectorize=CountVectorizer()
x_train_vec=vectorize.fit_transform(x_train)
x_test_vec=vectorize.transform(x_test)
# using naive bayes model
nb_model=MultinomialNB()
nb_model.fit(x_train_vec,y_train)
nb_prediction=nb_model.predict(x_test_vec)
label_map = {0: "Not Spam ✅", 1: "Spam ❌"}
nb_accuracy=accuracy_score(y_test,nb_prediction)
# using logistic regression model
lg_model=LogisticRegression()
lg_model.fit(x_train_vec,y_train)
lg_prediction=lg_model.predict(x_test_vec)
lg_accuracy=accuracy_score(y_test,lg_prediction)
print(f"Accuracy of Logisticregression is : {lg_accuracy}")
print(f"Accuracy of Naive_bayes is : {nb_accuracy}")

# checking for custom message:
custom_message=["Congratulations! You've won a free ticket to Bahamas!",
          "Hey, are we still on for lunch tomorrow?"]
custom_message_vec=vectorize.transform(custom_message)
for i in range(len(custom_message)):
    print(f"\nActual message : {custom_message[i]} ")
    print(f"Prediction of Lg is : {label_map[lg_model.predict(custom_message_vec)[i]]}")
    print(f"Prediction of Nb is : {label_map[nb_model.predict(custom_message_vec)[i]]}")






