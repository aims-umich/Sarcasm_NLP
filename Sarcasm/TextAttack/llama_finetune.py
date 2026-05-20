import torch
from transformers import AutoTokenizer
import numpy as np
from peft import AutoPeftModelForCausalLM
from transformers import pipeline
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import warnings
from transformers import (AutoModelForSequenceClassification, 
                          BitsAndBytesConfig,
                          AutoModelForCausalLM)
from transformers import DataCollatorWithPadding
import evaluate
import pandas as pd
from huggingface_hub import login
import datasets
from datasets import Dataset
from datasets import load_dataset
login(token='')
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import bitsandbytes as bnb #!
from peft import (LoraConfig, 
                  PeftConfig, 
                  get_peft_model, 
                  AutoPeftModelForSequenceClassification,
                  prepare_model_for_kbit_training) 
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
import os

os.environ["WANDB_DISABLED"] = "true"



cp = ["tiiuae/falcon-7b", 
      "meta-llama/Llama-2-7b-hf",
      "mistralai/Mistral-7B-Instruct-v0.2"]

checkpoint = cp[1]
output_dir = "llama" 

# Data Preperation
df = pd.read_csv('Human_Data_Aug_train.csv')
test = pd.read_csv('Human_Data_Aug_test.csv')
df.drop_duplicates(subset=['aug_text'], keep='first', inplace=True)
test.drop_duplicates(subset=['aug_text'], keep='first', inplace=True)
df= df.rename(columns={"sentiment": "label"})
test= test.rename(columns={"sentiment": "label"})
#df['label'] = df.label.replace({'positive':2,'negative':0,'neutral':1})
#test['label'] = test.label.replace({'positive':2,'negative':0,'neutral':1})
df_tr, df_te = train_test_split(df, test_size=0.2)
train_dataset = Dataset.from_dict(df_tr)
test_dataset = Dataset.from_dict(df_te)
my_dataset_dict = datasets.DatasetDict({"train":train_dataset,
                                        "test":test_dataset})

# Tokenizer definition
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

# Function to tokenize the data
def preprocess_function(examples):
    return tokenizer(examples["aug_text"], truncation=True, max_length=1024)
# Use the function
tokenized_text = my_dataset_dict.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Use accuarcy for evaluation metrics 
def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy':accuracy_score(predictions,labels)}

# Define labels
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

# Define model
if (checkpoint=="bert-base-uncased"):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, 
                                                               num_labels=3,
                                                               id2label=id2label, 
                                                               label2id=label2id)
    model.to("cuda")
else:
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                               num_labels=3,
                                                               id2label=id2label, 
                                                               label2id=label2id,
                                                               device_map='auto')

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = model.config.eos_token_id
model.config.use_cache = False

# to determine target_modeuls, use print(model) to identify "Linear" layers 
# use the name between parentheses

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, 
            dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss
        #return (loss(logits, labels), outputs) if return_output else loss(logits, labels)
  
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=16,
    gradient_checkpointing=True,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps = 25,
    bf16=True,
    evaluation_strategy="epoch",
    save_strategy="no",
report_to=None)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_text["train"],
    eval_dataset=tokenized_text["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics)

trainer.train()

trainer.save_model()


sentiment_task = pipeline("sentiment-analysis", 
                          model=model, 
                          tokenizer=tokenizer)

y_true = test['label']

y_pred = []
for i in range (len(test)):
    if (len(test['aug_text'][i]) > 1024):
        pass
    else:
        res = sentiment_task(test['aug_text'][i])
        label = res[0]['label']
        if (label=='neutral'):
            y_pred.append(1)
        elif (label=='positive'):
                y_pred.append(2)
        elif (label=='negative'):
            y_pred.append(0)
    print(i,'/',len(test))

target_names = ['negative', 'neutral', 'positive']
print(classification_report(y_true, y_pred, target_names=target_names))


print("Human Augmented Data")
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from sklearn.metrics import classification_report
df=pd.read_csv("TextAttack_Final_Data.csv")
               
df.drop_duplicates(subset=['Tweets'], keep='first', inplace=True)
               
x=df['Tweets'].tolist()

df['label'] = df['label'].replace({-1: 0, 0: 1, 1: 2})

y_true = df['label'].tolist()
               
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

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

df['falcon_Pred']=y_pred

target_names = ['negative', 'neutral', 'positive']
print(classification_report(df['label'], df['llama_Pred'], target_names=target_names))

df.to_csv("falcon_Pred_n.csv")

