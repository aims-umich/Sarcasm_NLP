## On the Impact of Language Nuances on Sentiment Analysis with Large Language Models: Paraphrasing, Sarcasm, and Emojis

This is github repo for research on exploring how textual nuances, including emojis and sarcasm, affect sentiment analysis, with a particular focus on improving data quality through text paraphrasing techniques. <br>

# Environment Installation

Create a new conda virtual environment with following command- <br>
```bash
conda create -n "myenv" python=3.11
```
Replace "myenv" with any name. Install all the required libraries using requirements.txt . <br>
```bash
pip install -r requirements.txt
```
For fine-tuning, having a GPU is a MUST. Check whether pytorch+cuda is installed using: 
```bash
import torch
print(torch.cuda.is_available())
```
If this prints ```False```, you can download torch+cuda from [Pytorch](https://pytorch.org/get-started/locally/) website.

# Folders 

```./Emoji```: This folder contains code to finetune BERT and LLM models on emoticon dataset.

```./Human_Sentiment_Data_Analysis```: This folder contains code to analyze human labelled sentiment analysis dataset.

```./Paper_Figure```: This code contains all the figures in the paper.

```./Paraphrase```: This folder contains code to finetune LLM models on paraphrased dataset.

```./Sarcasm```: This folder contains code to finetune large language models on Sarcasm. It contains following subfolders: <br>
  ```./Sarcasm/General```: Analyze performance of LLM models fine-tuned on General Tweets [dataset](https://www.kaggle.com/datasets/daniel09817/twitter-sentiment-analysis). <br>
  ```./Sarcasm/Nuclear```: Analyze performance of LLM models fine-tuned on Nuclear Tweets dataset. **Due to X/twitter restrictions, we are unable to publish this dataset publicly, but we can send it upon request**. <br>
  ```./Sarcasm/TextAttack```: Analyze performance of LLM models fine-tuned on data augumented using [TextAttack](https://textattack.readthedocs.io/en/latest/index.html) package.

# How to Get the Results
## Figures
To generate **Figures 1 and 2**, go to ```./Human_Sentiment_Data_Analysis/``` folder and run the following:
```bash
python generate_fig_1.py
python generate_fig_2a.py
python generate_fig_2b.py
```
To generate **Figures 3 and 4**, you need to download data files using these links: [link1](https://www.dropbox.com/scl/fi/zg55qnn3wh7oeiyd12yqi/vote_Paraphase_Combine.csv?rlkey=ir6qrz3k6gwtv67zpmqe6lrzt&st=vhbn1v7m&dl=0) and [link2](https://www.dropbox.com/scl/fi/8excrlwsflow6x1zjqg4b/tweets_nuclear_train-1.csv?rlkey=nsy2c1gevdlvbyu5pmgv6a31k&st=jto4baav&dl=0) for **Figure 4**. Once the download is complete, move the ```.csv``` files to ```./Paraphrase/``` folder and navigate there and run:
```bash
python gen_fig_3.py
python gen_fig_4.py
```
for faster execution of this code GPU is required. 

To generate **Figure 5**, go to folder ```./Sarcasm/Nuclear``` and execute: 
```bash
python generate_fig_5a.py
python generate_fig_5b.py
```
For all the above scripts, results will be stored as ```./Paper_Figure/fig_no.png```.
## Tables
To generate result of **Table 2** and **Table 3**, go to ```./Paraphrase``` folder and run scripts: 
```bash
python Falcon_Paraphrase.py > output_falcon.txt
python llama_paraphrase.py > output_llama.txt
python mistral_paraphrase.py > output_mistral.txt
```
To generate results of **Table 5**, go to folder ```Sarcasm/Nuclear```. Executing 
```bash
python Falcon_HSA.py
python llama_HSA.py
python Mistral_HSA.py
```
will generate results of **first** and **third** row of **Table 5**. Running 
```bash
python Falcon_Sarc_Aug.py
python Mistral_Sarc_Aug.py
python llama_Sarc_Aug.py
```
will generate results of **second** row of **Table 5**.

To generate results of **Table 6**, go to folder ```./Sarcasm/General```. Executing:
```bash
python Falcon_HSA.py
python llama_HSA.py
python Mistral_HSA.py 
```
will generate results of **first** and **third** row of table. Running:
```bash
python Falcon_Sarc_Aug.py
python Mistral_Sarc_Aug.py
python llama_Sarc_Aug.py 
```
will generate results of **second row** of table.

To generate results of **Table 7**, go to folder ```Sarcasm/TextAttack``` and run the following commands:
```bash
python Falcon_Table.py
python llama_table.py
python mistral_table.py 
```
This will generate all the results for Falcon, Llama and Mistral models respectively.
