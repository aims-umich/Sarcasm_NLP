import pandas as pd

df=pd.read_csv("HSA_Faclon_Pred.csv")

df=df[['Tweets','Sentiment','Sarcasm_Col']]

df.drop_duplicates(subset=['Tweets'], keep='first', inplace=True)

df['Sentiment'] = df['Sentiment'].replace({-1: 0, 0: 1, 1: 2})

print(df['Sentiment'].unique())

X=df[['Tweets']]

y=df['Sentiment']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train['label']=y_train

X_test['label']=y_test

X_train.to_csv("TA_train_n.csv", index=False)

X_test.to_csv("TA_test_n.csv", index=False)
