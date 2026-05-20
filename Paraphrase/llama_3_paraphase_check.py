"""
Script to compare original and paraphased text using Llama 3.
"""

import transformers
import torch
import pandas as pd

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

df=pd.read_csv("Merged_Tweets_100k.csv") #File path

for index,row in df.iterrows():

    orig_text=row['Original_Tweet']

    para_text=row['Paraphased_Tweet']

    merge_text="Text_1: "+orig_text+" "+"Text_2: "+para_text

    print(merge_text)

    messages = [
        {"role": "system", "content": "Reply with yes or no whether Text_1 and Text_2 convey the same meaning."},
        {"role": "user", "content": merge_text},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=20,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"][len(prompt):])
