r"""
Script to send api request to UM-GPT.
"""

from openai import AzureOpenAI
import os
#from dotenv import load_dotenv
import json
from datetime import datetime
import pandas as pd
import time
#Sets the current working directory to be the same as the file.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load environment file for secrets.
"""
try:
    if load_dotenv('.env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()
"""
#Create Azure client



# Send a completion call to generate an answer
#print('Sending a test completion job')
df=pd.read_csv("Nuclear_Test_lc.csv",lineterminator='\n')   #Enter csv file name here
#df=df[44183:]
label=[]
st=datetime.now()
print(st)
c=1
for i in df['tweets']:

    
    s="text: "+i
    

    try:
        response = client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=[
                        {"role": "system", "content": "Paraphrase the following text while keeping the response length approximately same as text."},
                        {"role": "user", "content": s}
                    ],
                    temperature=0,
                    stop=None,
                    max_tokens=190)

            #Print response.
            #print(response)
            #print(response.choices[0].message.content)
        label.append(response.choices[0].message.content)
            
        with open("results_test_100k.txt", "a") as outfile:
            json.dump(response.choices[0].message.content, outfile)
            outfile.write("\n")

        with open("respnse_log_test_100k.txt", "a") as outfile2:
            outfile2.write(str(c))
            outfile2.write("\n")
            outfile2.write(str(response))
            outfile2.write("\n")

    except Exception as error:

        label.append("Error")

        with open("results_test_100k.txt", "a") as outfile:
            outfile.write("Error")
            outfile.write("\n")

        with open("respnse_log_test_100k.txt", "a") as outfile2:
            outfile2.write(str(c))
            outfile2.write("\n")
            outfile2.write(str(error))
            outfile2.write("\n")


    c+=1    
    time.sleep(1.7)

et=datetime.now()

print("Time Difference: ",et-st)

file = open('label_test_100k.txt','a')
for i in label:
	file.write(i+"\n")

#df['GPT_3_Label']=label

#To convert the string in file to text
"""
file = open("results.txt", "r")
Lines = file.readlines()
for line in Lines:
    
    #print(line)
    #print(type(line))
    res = json.loads(line)
    #print(res['name'])
    print(type(res))
"""
