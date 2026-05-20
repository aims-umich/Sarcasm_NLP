import pandas as pd
from pathlib import Path 

df=pd.read_csv("tweets_nuclear_train(1).csv")

print(df.columns)

d_orig={3:0,4:0,5:0,6:0,7:0}

for index,row in df.iterrows():

    nw=row['tweets'].split(' ')

    if len(nw)!=1:
    
        m=row['max']

        d_orig[m]+=1

print(d_orig)

df1=pd.read_csv("vote_Paraphase_Combine.csv")

d_new={3:0,4:0,5:0,6:0,7:0}

for index,row in df1.iterrows():

    nw=row['tweet'].split(' ')

    if len(nw)!=1:
    
        m=max(row['count_positive'],row['count_neutral'],row['count_negative'])

        d_new[m]+=1

#print(d)

d_final={3:0,4:0,5:0,6:0,7:0}

d_final[3]=d_new[3]

d_final[4]=d_new[4]

d_final[5]=d_orig[5]+d_new[5]

d_final[6]=d_orig[6]+d_new[6]

d_final[7]=d_orig[7]+d_new[7]

import matplotlib.pyplot as plt
import numpy as np

# Data
categories = [3, 4, 5, 6, 7]
original_counts = [d_orig[3], d_orig[4], d_orig[5], d_orig[6],d_orig[7]]
new_counts = [d_final[3], d_final[4], d_final[5], d_final[6], d_final[7]]


print(original_counts)
print(new_counts)
# Bar width
bar_width = 0.4
x = np.arange(len(categories))

p = Path('Paper_Figure')
p.mkdir(parents=True, exist_ok=True)

# Plot bars
plt.figure(figsize=(10, 6), dpi=300)
plt.bar(x - bar_width / 2, original_counts, width=bar_width, label="Original Data", color="blue", alpha=0.7)
plt.bar(x + bar_width / 2, new_counts, width=bar_width, label="Paraphrased Data", color="orange", alpha=0.7)

# Labels and title
plt.xlabel("Categories", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Comparison of Original and Paraphrased Data Distribution", fontsize=14)
plt.xticks(x, categories)
plt.legend()

# Show grid
#plt.grid(axis='y', linestyle="--", alpha=0.7)

# Save high-resolution image
plt.savefig("fig_4.png", dpi=300, bbox_inches="tight")

# Show plot
#plt.show()
