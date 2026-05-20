import pandas as pd

df1=pd.read_csv("Falcon_Final_Pred_HSA.csv")                              #Change
c=0
for index,row in df1.iterrows():
    if row['falcon_Pred']==row['label']:                          #Change
        c+=1
print("Accuracy of Data without Sarcasm Augmentation:")
print(c/df1.shape[0])

df2=pd.read_csv("falcon_Pred_sa.csv")                             #Change

df1_filtered = df1[~((df1['label'] != df1['falcon_Pred']) & (df1['Sarcasm_Col'] == 'Sarcastic'))]          #Change

# Print the final dataframe
#print(df1_filtered)

print(df1_filtered.shape)

# Append both dataframes vertically
df_final = pd.concat([df1_filtered, df2], ignore_index=True)

c=0
for index,row in df_final.iterrows():
    if row['falcon_Pred']==row['label']:                                 #Change
        c+=1

print("Accuracy of Data with Sarcasm Augmentation:")
print(c/df1.shape[0])

df1=pd.read_csv("Falcon_Final_Pred_HSA.csv")                                     #Change

df1=df1[df1['Sarcasm_Col']=='Sarcastic']
c=0
for index,row in df1.iterrows():
    if row['falcon_Pred']==row['label']:                                 #Change
        c+=1

print("Accuracy of Data only on Sarcastic Tweets:")
print(c/df1.shape[0])
