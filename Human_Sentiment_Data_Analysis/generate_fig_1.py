import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 

df1=pd.read_csv("Human_Sentiment_Agree.csv")

value_counts = df1['Sarcasm_Col'].value_counts()

value_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])

p = Path('Paper_Figure')
p.mkdir(parents=True, exist_ok=True)
plt.savefig("./Paper_Figure/Fig_1.png")
