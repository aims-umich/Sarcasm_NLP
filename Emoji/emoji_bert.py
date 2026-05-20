from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import sklearn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification, TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import InputExample, InputFeatures

df=pd.read_csv("average_Tweet_Emoji.csv")

#print(df.columns)

df = df[df['Orig_Tweet'].notna()]

df = df[df['Dec_Tweet'].notna()]

def cat2num(value):
    if value=='Positive':
        return 1
    elif value=='Negative':
        return 2
    elif value=='Neutral':
        return 0
    else:
        print("Error")


df['FinalLabel']  =  df['FinalLabel'].apply(cat2num)

train,test = train_test_split( df, test_size=0.25, random_state=42)

def convert_data_to_examples(train, test, review, sentiment):
    train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[review],
                                                          label = x[sentiment]), axis = 1)

    validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[review],
                                                          label = x[sentiment]), axis = 1,)

    return train_InputExamples, validation_InputExamples

train_InputExamples, validation_InputExamples = convert_data_to_examples(train,  test, 'Dec_Tweet',  'FinalLabel')


def bert_ver(tokenizer,model,epoch_no):
    def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
        features = [] # -> will hold InputFeatures to be converted later

        for e in tqdm(examples):
            input_dict = tokenizer.encode_plus(
                e.text_a,
                add_special_tokens=True,    # Add 'CLS' and 'SEP'
                max_length=max_length,    # truncates if len(s) > max_length
                return_token_type_ids=True,
                return_attention_mask=True,
                pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
                truncation=True
            )

            input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],input_dict["token_type_ids"], input_dict['attention_mask'])
            features.append(InputFeatures( input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label) )

        def gen():
            for f in features:
                yield (
                    {
                        "input_ids": f.input_ids,
                        "attention_mask": f.attention_mask,
                        "token_type_ids": f.token_type_ids,
                    },
                    f.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )


    DATA_COLUMN = 'Dec_Tweet'
    LABEL_COLUMN = 'FinalLabel'

    train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
    train_data = train_data.shuffle(100).batch(32).repeat(2)

    validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
    validation_data = validation_data.batch(32)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

    print(model.fit(train_data, epochs=epoch_no, validation_data=validation_data, steps_per_epoch=80))

print("albert-base-v2:")

model = TFAutoModelForSequenceClassification.from_pretrained("albert-base-v2", num_labels=3) #Change
#model.layers[0].trainable = False
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2") #Change

bert_ver(tokenizer,model,2)

print("ConvBERT-")

model = TFAutoModelForSequenceClassification.from_pretrained("YituTech/conv-bert-base", num_labels=3) #Change
#model.layers[0].trainable = False
tokenizer = AutoTokenizer.from_pretrained("YituTech/conv-bert-base") #Change

bert_ver(tokenizer,model,5)