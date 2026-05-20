import pandas as pd

df1=pd.read_csv("llama_Final_Pred_HSA.csv")
c=0
for index,row in df1.iterrows():
    if row['llama_Pred']==row['label']:
        c+=1
print("Accuracy of Data without Sarcasm Augmentation:")
print(c/df1.shape[0])

df2=pd.read_csv("llama_Pred_sa.csv")

df1_filtered = df1[~((df1['label'] != df1['llama_Pred']) & (df1['Sarcasm_Col'] == 'Sarcastic'))]

# Print the final dataframe
#print(df1_filtered)

print(df1_filtered.shape)

# Append both dataframes vertically
df_final = pd.concat([df1_filtered, df2], ignore_index=True)

c=0
for index,row in df_final.iterrows():
    if row['llama_Pred']==row['label']:
        c+=1

print("Accuracy of Data with Sarcasm Augmentation:")
print(c/df1.shape[0])

df1=pd.read_csv("llama_Final_Pred_HSA.csv")

df1=df1[df1['Sarcasm_Col']=='Sarcastic']
c=0
for index,row in df1.iterrows():
    if row['llama_Pred']==row['label']:
        c+=1

print("Accuracy of Data only on Sarcastic Tweets:")
print(c/df1.shape[0])
