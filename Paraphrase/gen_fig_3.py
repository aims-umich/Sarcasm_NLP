import os
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb #!
import torch
import torch.nn as nn
import transformers
from datasets import Dataset #!
from peft import LoraConfig, PeftConfig #!
from trl import SFTTrainer   #!
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from huggingface_hub import login
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def predict(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 1, 
                        temperature = 0.2,
                       )
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("=")[-1]
        answer=answer.lower()
        print(answer)
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        elif "neutral" in answer:
            y_pred.append("neutral")
        else:
            y_pred.append("none")
    return y_pred


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"

import warnings
warnings.filterwarnings("ignore")

print("Script Started.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X = pd.read_csv('Final_test.csv')

X.rename(columns={"Tweet": "text"}, inplace=True)

def generate_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = {data_point["sentiment"]}
            """.strip()

def generate_test_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = """.strip()

X = pd.DataFrame(X.apply(generate_test_prompt, axis=1), 
                       columns=["text"])

#X_train.rename(columns={"Tweet": "text"}, inplace=True)

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)


#merged_model_path = "./merged_model"
repo_name = "namanb/llama_nuc"
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    repo_name,
    device_map="auto",
    torch_dtype=compute_dtype,
    quantization_config=bnb_config, 
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(repo_name, 
                                          trust_remote_code=True,
                                         )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



# QLORA
peft_config = LoraConfig(
        lora_alpha=16, 
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

# Testing
y_pred = predict(X, model, tokenizer)

#print(y_pred)

X['Sent_Pred']=y_pred

cm = confusion_matrix(X['Sentiment'], X['Sent_Pred'])

# Compute percentages
cm_percentage = cm.astype('float') / cm.sum() * 100

p = Path('Paper_Figure')
p.mkdir(parents=True, exist_ok=True)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

#


# Adding percentage annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j + 0.5, i + 0.55, f"\n({cm_percentage[i, j]:.2f}%)",
                 ha='center', va='center', color='black', fontsize=10)

plt.savefig("./Paper_Figure/fig_3.png", dpi=300, bbox_inches="tight")
