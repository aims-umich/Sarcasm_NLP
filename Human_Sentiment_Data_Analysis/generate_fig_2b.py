import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df1=pd.read_csv("Human_Sentiment_Agree.csv")
df1 = df1[df1['Sarcasm_Col'] == 'Sarcastic']
mapping_1={1:'Positive',-1:'Negative', 0:'Neutral'}
df1['Sentiment'] = df1['Sentiment'].replace(mapping_1)
value_counts = df1['Sentiment'].value_counts()

p = Path('Paper_Figure')
p.mkdir(parents=True, exist_ok=True)

# Plot using Matplotlib
plt.figure(figsize=(6, 4))
value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
#plt.title("Category Counts")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
#plt.grid(axis='y', alpha=0.7)

# Show plot
#plt.show()
plt.savefig("./Paper_Figure/Fig_2_b.png")
