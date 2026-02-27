import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv(r"C:\Data Science\Project Assignments and Project Topic list\House Price Prediction\Housing.csv")

df['mainroad'] = df['mainroad'].map({'yes':1, 'no':0})
df['guestroom'] = df['guestroom'].map({'yes':1, 'no':0})
df['basement'] = df['basement'].map({'yes':1, 'no':0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes':1, 'no':0})
df['airconditioning'] = df['airconditioning'].map({'yes':1, 'no':0})
df['prefarea'] = df['prefarea'].map({'yes':1, 'no':0})

df['furnishingstatus'] = df['furnishingstatus'].map({
    'unfurnished':0,
    'semi-furnished':1,
    'furnished':2
})


X = df.drop('price', axis=1)
y = df['price']

model = LinearRegression()

model.fit(X, y)


pickle.dump(model, open("house_model.pkl", "wb"))

print("Model trained and saved successfully")