import pandas as pd

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

import ast

aug_df=pd.read_csv("HSA_Mistral_Pred_Gen.csv")         #Change
print(aug_df.columns)
aug_df = aug_df.rename(columns={'Mistral_Pred': 'Pred'})    #Change

#print(aug_df.head())

def extract_label(json_str):
    try:
        # Convert string to a Python object (list of dictionaries)
        json_obj = ast.literal_eval(json_str)
        if isinstance(json_obj, list) and len(json_obj) > 0 and 'label' in json_obj[0]:
            return json_obj[0]['label']
    except (ValueError, SyntaxError):
        print(json_obj)
        return None  # Return None if parsing fails
    return None  # Return None if format is incorrect


#falcon_df['Falcon_Pred'] =falcon_df['Falcon_Pred'].apply(extract_label)
aug_df['Pred'] = aug_df['Pred'].apply(extract_label)

mapping_1={'positive':1,'negative':-1, 'neutral':0}

mapping_2={'Positive':1,'Negative':-1, 'Neutral':0}

#falcon_df['Falcon_Pred'] = falcon_df['Falcon_Pred'].replace(mapping_1)

aug_df['Pred'] = aug_df['Pred'].replace(mapping_1)

aug_df['Sentiment'] = aug_df['Sentiment'].replace(mapping_2)

aug_df=aug_df[['Tweets', 'Sentiment', 'Sarcasm_Col','Topic', 'Pred']]

c=0
for index,row in aug_df.iterrows():
    if row['Sentiment']==row['Pred']:
        c+=1

print(c/aug_df.shape[0])
aug_df=pd.read_csv("HSA_Mistral_Pred_Gen.csv")         #Change
print(aug_df.columns)
aug_df = aug_df.rename(columns={'Mistral_Pred': 'Pred'})    #Change

aug_df = aug_df[aug_df['Sarcasm_Col'] == 'Sarcastic']

aug_df['Pred'] = aug_df['Pred'].apply(extract_label)

mapping_1={'positive':1,'negative':-1, 'neutral':0}

mapping_2={'Positive':1,'Negative':-1, 'Neutral':0}

#falcon_df['Falcon_Pred'] = falcon_df['Falcon_Pred'].replace(mapping_1)

aug_df['Pred'] = aug_df['Pred'].replace(mapping_1)

aug_df['Sentiment'] = aug_df['Sentiment'].replace(mapping_2)

aug_df=aug_df[['Tweets', 'Sentiment', 'Sarcasm_Col','Topic', 'Pred']]

c=0
for index,row in aug_df.iterrows():
    if row['Sentiment']==row['Pred']:
        c+=1

print(c/aug_df.shape[0])
