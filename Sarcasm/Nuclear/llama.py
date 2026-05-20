from transformers import AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd

df=pd.read_csv("Human_Sentiment_Agree.csv")
checkpoint = 'llama2'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}
#print("Hi")    

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, 
                                                       num_labels=3,
                                                       id2label=id2label, 
                                                       label2id=label2id,
                                                       device_map='auto')

#print("Hi1")

sentiment_task = pipeline("sentiment-analysis", 
                          model=model, 
                          tokenizer=tokenizer)

#print("Hi2")

l_pred=[]
for i in df['Sarcasm_Aug']:
    try:
        p=sentiment_task(i)
        l_pred.append(p)
        print(p)
    except Exception as e:
        l_pred.append("Error")
        print("Error")
        
    
df['llama_Pred']=l_pred

df.to_csv("HSA_Llama_Pred_Nuc.csv")
