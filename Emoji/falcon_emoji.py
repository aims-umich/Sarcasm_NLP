import torch
from transformers import AutoTokenizer
import numpy as np
from peft import AutoPeftModelForCausalLM
from transformers import pipeline
import os
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"

cp = ["tiiuae/falcon-7b", 
      "meta-llama/Llama-2-7b-hf",
      "mistralai/Mistral-7B-Instruct-v0.2"]

checkpoint = cp[0]
output_dir = "falcon_emoji_orig" 

# Data Preperation
df = pd.read_csv('Emoji_train.csv') #Import
test = pd.read_csv('Emoji_test.csv') #Import
df=df[['Orig_Tweet','FinalLabel']]
test=test[['Orig_Tweet','FinalLabel']]

#print(df.columns)

df= df.rename(columns={"FinalLabel": "label"})
test= test.rename(columns={"FinalLabel": "label"})
df['label'] = df.label.replace({'Positive':2,'Negative':0,'Neutral':1})
test['label'] = test.label.replace({'Positive':2,'Negative':0,'Neutral':1})


df_tr, df_te = train_test_split(df, test_size=0.2)
df_tr['Orig_Tweet'] = df_tr['Orig_Tweet'].astype(str)
df_te['Orig_Tweet'] = df_te['Orig_Tweet'].astype(str)


train_dataset = Dataset.from_dict(df_tr)
test_dataset = Dataset.from_dict(df_te)

my_dataset_dict = datasets.DatasetDict({"train":train_dataset,
                                        "test":test_dataset})

# Tokenizer definition
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

# Function to tokenize the data
def preprocess_function(examples):
    #print(examples["Orig_Tweet"])
    return tokenizer(examples["Orig_Tweet"], truncation=True, max_length=1024)
# Use the function
#print(my_dataset_dict)
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
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
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
'''
tm_llama = ['v_proj', 
            'down_proj', 
            'up_proj', 
            'q_proj', 
            'gate_proj', 
            'k_proj', 
            'o_proj',
            'score']

tm_falcon = ["query_key_value",
             "dense",
             "dense_h_to_4h",
             "score"]

tm_mistral = ['q_proj', 
              'k_proj', 
              'v_proj', 
              'o_proj',
              'gate_proj',
              'up_proj',
              'down_proj',
              'score']

peft_config = LoraConfig(
        lora_alpha=24, 
        lora_dropout=0.1,
        r=256,
        bias="none",
        target_modules=tm_falcon, 
        task_type="SEQ_CLS")

#model = prepare_model_for_kbit_training(model)
#model = get_peft_model(model, peft_config)
#model.print_trainable_parameters()

#model.to('cuda')
'''
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, 
            dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps = 25,
    bf16=True,
    evaluation_strategy="epoch",
    save_strategy="no")

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

# Evaluate the model on test data
def generate_predictions(model, df_test, batch_size):
    sentences = df_test.Orig_Tweet.tolist()
    all_outputs = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=512)

        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs['logits'])

    final_outputs = torch.cat(all_outputs, dim=0)
    df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()

# Compile the function
generate_predictions(model, test, 32)
y_pred =test['predictions']
y_test = test['label']

# Classification Report
target_names = ['negative', 'neutral', 'positive']
print(classification_report(y_test, y_pred, target_names=target_names))
