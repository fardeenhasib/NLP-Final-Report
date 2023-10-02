import numpy as np
import pandas as pd
import os
import json
# import stanza
from tqdm import tqdm

train_file = "/Users/adritaanika/Documents/nlp_proj/mednli_baseline/data/our_data/train.json"
trials = "/Users/adritaanika/Documents/nlp_proj/mednli_baseline/data/our_data/Clinical trial json"
# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')



# def cons_parser(sen):
#     doc = nlp(sen)
#     res = ""
#     for sentence in doc.sentences:
#         res = res + " " + sentence.constituency
#     return res.lstrip()


def create_data(example):
    if example["Type"]=="Single":
        trial = example["Primary_id"] + ".json"
        tf = open(os.path.join(trials,trial))
        data_trial = json.load(tf)
        sentence1 = " ".join(data_trial[example["Section_id"]])
        sentence1 = example["Primary_id"] + " " + sentence1
        sentence2 = example["Statement"]
        gold_label = example["Label"].lower()
        
        # labels = labels + gold_label + "\n" 
        #parsing
        # sentence1_parse = nlp(sentence1)
        # sentence2_parse = nlp(sentence2)
        # print(sentence1_parse)
        # print(sentence2_parse)
        
    else:
        trial = example["Primary_id"] + ".json"
        tf = open(os.path.join(trials,trial))
        data_trial = json.load(tf)
        
        trial2 = example["Secondary_id"] + ".json"
        tf2 = open(os.path.join(trials,trial2))
        data_trial2 = json.load(tf2)
        
        sentence1 = " ".join(data_trial[example["Section_id"]]) + " " + " ".join(data_trial2[example["Section_id"]])
        sentence1 = example["Primary_id"] +  " " + sentence1
        sentence2 = example["Statement"]
        gold_label = example["Label"].lower()
        
        #text file
    print(sentence1)
    print("meg \n", sentence2, "\n\n\n")
    file1.write(sentence1.replace('  ', ' ') + "\t" + sentence2.replace('\t', '') + "\n")
    
    return {"sentence1": sentence1, "sentence2": sentence2, "gold_label": gold_label}


f = open(train_file)
data = json.load(f)
write_data = {}
labels = ""
file1 = open("adrita_input_text.txt", "w")

for k,v in tqdm(data.items()):
    write_data[k] = create_data(v)
    labels = labels + create_data(v)['gold_label'] + "\n" 
    # break
    
## text file writing
file1.close()
file2 = open("adrita_labels.txt","w")
file2.writelines(labels)
file2.close() 

# json file writing 
with open("adrita_new_data.json", "w") as outfile:
    json.dump(write_data, outfile)
