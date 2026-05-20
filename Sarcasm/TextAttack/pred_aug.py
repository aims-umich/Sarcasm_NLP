import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from sklearn.metrics import classification_report
df=pd.read_csv("Sarcasm_Aug_Data.csv")

mapping={"Negative": 0, "Neutral": 1, "Positive": 2}

df["Sentiment"]=df["Sentiment"].replace(mapping)
               
df.drop_duplicates(subset=['Tweets'], keep='first', inplace=True)
               
x=df['Tweets'].tolist()

df.rename(columns={'Sentiment': 'label'}, inplace=True)

df['label'] = df['label'].replace({-1: 0, 0: 1, 1: 2})

y_true = df['label'].tolist()
               
user_input = input("Enter a number (1 Falcon, 2 Llama, or 3 Mistral): ")

# Mapping input to checkpoint values
if user_input == "1":
    checkpoint = "falcon"
    fname="Falcon_Pred_Aug.csv"
elif user_input == "2":
    checkpoint = "llama"
    fname="llama_Pred_Aug.csv"
elif user_input == "3":
    checkpoint = "mistral"
    fname="Mistral_Pred_Aug.csv"
else:
    checkpoint = "Invalid Input"
               
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
               
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, 
                                                               num_labels=3,
                                                               id2label=id2label, 
                                                               label2id=label2id)

sentiment_task = pipeline("sentiment-analysis", 
                          model=model, 
                          tokenizer=tokenizer,device="cuda")
y_pred = []
for i in range (len(x)):
    res = sentiment_task(x[i])
    label = res[0]['label']
    if (label=='neutral'):
        y_pred.append(1)
    elif (label=='positive'):
        y_pred.append(2)
    elif (label=='negative'):
        y_pred.append(0)
    print(i,'/',len(x))

df['Pred']=y_pred

df.to_csv(fname)

target_names = ['negative', 'neutral', 'positive']

print("Matching sentiment: ",(df['label'] == df['Pred']).sum())

print(classification_report(df['label'], df['Pred'], target_names=target_names))
