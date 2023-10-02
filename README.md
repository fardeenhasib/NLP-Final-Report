# Multi-evidence Natural Language Inference for Clinical Trial Data

This project presents a natural language inference (NLI) task for clinical trial data, where the goal is to determine whether a hypothesis can be inferred from one or more clinical trial reports (CTRs). We utilized a dataset of breast cancer CTRs, hypotheses, explanations, and labels annotated by domain experts.  We also explored various models and methods to tackle this challenging task, such as BERT-based models, knowledge infusion, and contrastive learning.

## Dataset

The dataset consists of 1350 CTRs and 1650 hypotheses, along with explanations and labels (entailment or contradiction). The CTRs are extracted from https://clinicaltrials.gov/ct2/home and cover four sections: eligibility criteria, intervention, results, and adverse events. The hypotheses and explanations are annotated by clinical domain experts, clinical trial organizers, and research oncologists from the Cancer Research UK Manchester Institute. The dataset is split into 80-10-10 (train-validation-test) partitions.

## Models and Methods

We have experimented with several models and methods to perform NLI on the clinical trial data, such as:

- BERT-based models: We fine-tuned several pre-trained language models, such as BERT, RoBERTa, PubMedBERT, BioELECTRA, and DeBERTa, on the dataset and compares their performance.
- Knowledge infusion: We infused disease-related knowledge into the BERT-based models by pre-training them on Wikipedia articles about diseases associated with breast cancer. The result shows that knowledge infusion can improve the performance of some models, but not all.
- PairSupCon: We proposed a pairwise supervised contrastive learning method that encodes high-level categorical concepts into the sentence representations and improves the low-level semantic entailment and contradiction reasoning. The result shows that this method outperforms the BERT-based models and achieves the best performance on the dataset.

## Results

We reported the accuracy and F1-score of the models and methods on the test set of the dataset. We also provided qualitative analysis and error analysis of the results. It shows that the proposed PairSupCon method achieves the highest accuracy and F1-score of 64.24% and 62.04%, respectively, which is much better than the baseline performance of 32%. The results also show that the models and methods face challenges such as numerical and quantitative reasoning, word distribution shift, and long premise and hypothesis length.

## Commands to run the project

### To install the packages
```
pip install -r requirements.txt
```
### To finetune the model with breast cancer infused knowledge,
```
python disease_knowledge_infusion_training.py
```
### To run the model trained without breast cancer infused knowledge,
```
python run_model_without_disease_infused.py --type {model_type}
```

### To run the model trained with breast cancer infused knowledge,
```
python run_model_w_breast_knowledge.py --type {model_type}
```


## Extensive deatils on the project can be found in the NLP_report.pdf file
