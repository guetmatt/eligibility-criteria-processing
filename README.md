# Annotation and Reusability of Eligibility Criteria: Applying Natural Language Processing to Free-Text Descriptions in Clinical Trials
This repository contains the implementation of a modular NLP pipeline for processing clinical trial eligibility criteria. As part of a master's thesis, two models for Named Entity Recognition and two models for Relation Extraction were trained, with the goal to transform clinical trial eligibility criteria from free-text into a structured, machine-operable format. All models are based on BERT-derivatives and trained for task adaption on the Chia dataset by Kury et al. (2020) (https://doi.org/10.1038/s41597-020-00620-0). 



# Project Structure
```
eligibility-criteria-processing/
├── data		# Chia dataset for model training
├── data_ctg		# ClinicalTrials.gov dataset
├── models		# trained models
├── results		# Results from applying trained models to ClinicalTrials.gov dataset
├── src/
│   ├── data_statistics.py
│   ├── ner_parseChia.py 
│   ├── ner_training.py
│   ├── parse_ctg.py
│   ├── pipeline_inference.py
│   ├── re_parseChia.py
│   └── re_training.py
├── statistics
├── training_notebooks
└── requirements.txt
```




# ADJUST
If downloading the trained models from this GitHub repository causes errors, try downloading the models from the backup upload on huggingface.
For an introduction on how to download models from huggingface, see https://huggingface.co/docs/hub/models-downloading .
The models on huggingface can be found here:
- https://huggingface.co/gutbier/NER_chia_tok
- https://huggingface.co/gutbier/NER_chia_seq
- https://huggingface.co/gutbier/RE_clinicalBERT
- https://huggingface.co/gutbier/RE_sapBERT


# Module Documentation & Usage
The python scripts in the ```src/``` directory build a modular pipeline. Below is the functional breakdown and usage guide for each component.

## ClinicalTrials.gov data preparation
### ```parse_ctg.py```
- Description
	- Processes raw csv exports from ClinicalTrials.gov
	- Extracts eligibility criteria text blocks and segments them into sentence-level entries

- Arguments
	- ```data_dir```
		- Path to the directory with raw ClinicalTrials.gov csv files
		- Default = ```../data_ctg/Studies_with_id_andEligibilityCriteria.csv```
	- ```output_dir```
		- Path to directory to save the processed dataset
		- Default = ```../data_ctg/parsedCTG_sentlevel.csv```

- Example uses
```console
	python src/parse_ctg.py
```
```console
	python src/parse_ctg.py \
		--data_dir data_ctg/Studies_with_id_andEligibilityCriteria.csv
		--output_dir data_ctg/parsedCTG_sentlevel.csv
```



## Named Entity Recognition - Training
### ```ner_parseChia.py```
- Description
	- Prepares the chia dataset for NER training
	- Parses the ```.ann```- and ```.txt```-files from the directory ```data/chia_without_scope``` into tokenized, BIO-labeled sequences for a sentence-level training dataset
	- Splits dataset into train/eval/test splits
	- Handles discontinuous entity spans, realigns entity indices to line-level, and uses iterative stratification to ensure balenced entity distribution across splits

- Arguments
	- ```data_dir```
		- Path to directory with raw dataset
		- Default = ```../data/chia_without_scope```
	- ```output_dir```
		- Path to directory to save the parsed dataset and label mappings to
		- Default = ```../data/chia_without_scope_parsedNER_v1```
	- ```model_checkpoint```
		- HuggingFace model checkpoint for tokenizer
		- Default = ```emilyalsentzer/Bio_ClinicalBERT```
	- ```max_len```
		- Maximum sequence length for tokenization
		- Default = ```128```

- Example uses
```console
	python src/ner_parseChia.py
```
```console
	python src/ner_parseChia.py \
		--data_dir data/chia_without_scope \
		--output_dir data/chia_without_scope_parsedNER_v1 \
		--model_checkpoint emilyalsentzer/Bio_ClinicalBERT \
		--max_len 128
```


### ```ner_training.py```
- Description
	- Handles training and hyperparameter optimization for the NER model and tokenizer
	- Supports span-based as well as token-based evaluation
 
- Arguments
	- ```data_dir```
		- Path to directory with parsed dataset for NER (from ```ner_parseChia.py```)
		- Default = ```../data/chia_without_scope_parsedNER_v1```
	- ```output_dir```
		- Path to directory to save the trained model and tokenizer to
		- Default = ```../models/NER_chia_v1```
	- ```model_checkpoint```
		- HuggingFace model checkpoint for model and tokenizer that will be trained
		- Default = ```emilyalsentzer/Bio_ClinicalBERT```
	- ```eval_method```
		- Model evaluation strategy during training: ```seq``` (span-based) or ```tok``` (token-based)
		- Default = ```seq```
	- ```do_hpo```
		- Flag to enable hyperparameter optimization (hpo): ```True``` (hpo) or ```False``` (no hpo)
		- Default = ```True```
	- ```hpo_trials```
		- Number of trials for hyperparameter optimization, starts at 0
		- Default = ```10```
- Example uses
```console
	python src/ner_training.py
```
```console
	python src/ner_training.py \
		--data_dir data/chia_without_scope_parsedNER_v1 \
		--output_dir models/NER_chia_v1 \
		--model_checkpoint emilyalsentzer/Bio_ClinicalBERT \
		--eval_method seq \
		--do_hpo True \
		--hpo_trials 10 \
```

## Relation Extraction - Training
### ```re_parseChia.py```
- Description
	- Prepares the chia dataset for Relation extraction by generating candidate entity pairs
	- Splits dataset into train/eval/test splits
	- Generates directed permutations of entities within a sentence
	- Handles extreme calss imbalance via negative downsampling
 
- Arguments
	- ```data_dir```
		- Path to directory with raw dataset
		- Default = ```../data/chia_without_scope```
	- ```output_dir```
		- Path to directory to save the parsed dataset and label mappings to
		- Default = ```../data/chia_without_scope_parsedRE_v1```
	- ```global_downsample_rate```
		- Ratio of negative samples ("NO_RELATION") tp keep in the entire dataset
		- [0.0-1.0], 0.2 = keep 20%, 1.0 = keep 100%, None = keep 100%
		- Default = ```None```
	- ```train_downsample_rate```
		- Ratio of negative samples ("NO_RELATION") to keep in the training split
		- [0.0-1.0], 0.2 = keep 20%, 1.0 = keep 100%, None = keep 100%
		- Default = ```0.2```
	- ```seed```
		- Random seed for reproducibility and stratification
		- Default = ```42```
- Example uses
```console
	python src/re_parseChia.py
```
```console
	python src/re_parseChia.py \
	--data_dir data/chia_without_scope \
	--output_dir data/chia_without_scope_parsedRE_v1 \
	--global_downsample_rate 0.5 \
	--train_downsample_rate 0.2 \
	--seed 42
```


### ```re_training.py```
- Description
	- Trains a sequence classifier for relation identification
	- Injects entity markers into text and resizes model embeddings for new special tokens
	- Uses weighted cross-entropy loss to account for class imbalance
 
- Arguments
	- ```data_dir```
		- Path to the parsed chia dataset directory from ```re_parseChia.py```
		- Default = ```../data/chia_without_scope_parsedRE_v1```
	- ```output_dir```
		- Path to directory to save the trained model and tokenizer to
		- Default = ```../models/RE_chia_v1```
	- ```model_checkpoint```
		- HuggingFace model checkpoint for model and tokenizer that will be trained
		- Default = ```emilyalsentzer/Bio_ClinicalBERT```
	- ```max_len```
		- Maximum sequence length after token injection
		- Default = ```256```
	- ```do_hpo```
		- Flag to enable hyperparameter optimization (hpo): ```True``` (hpo) or ```False``` (no hpo)
		- Default = ```True```
	- ```hpo_trials```
		- Number of trials for hyperparameter optimization, starts at 0
		- Default = ```10```
	 
- Example uses
```console
	python src/re_training.py
```
```console
	python src/re_training.py \
		--data_dir data/chia_without_scope_parsedRE_v1 \
		--output_dir models/RE_chia_v1 \
		--model_checkpoint emilyalsentzer/Bio_ClinicalBERT \
		--max_len 256 \
		--do_hpo True \
		--hpo_trials 10
```


## Inference
### ```pipeline_inference.py```
- Description
	- End-to-end pipeline to perform NER and RE on the ClinicalTrials.gov data
	- Loads preprocessed ClinicalTrials.gov data
	- Loads trained NER and RE models and tokenizers
	- Performs NER and RE on ClinicalTrials.gov data
	- Saves predictions to disk

- Arguments
	- ```data_dir```
		- Path to the directory containing the preprocessed ctg dataset
		- Default = ```../data_ctg/parsedCTG_sentlevel```
	- ```output_dir```
		- Path to the directory to save the predictions to
		- Automatically adds 'ner_predictions' and 're_predictions' as endings to filenames
		- Default = ```../results```
	- ```ner_model_path```
		- Path to the directory containing the NER model
		- No default
	- ```re_model_path```
		- Path to the directory containing the RE model
		- No default
	- ```mode```
		- Falg to apply NER + RE inference, NER only, or RE only
		- Options: [ner+re, ner, re]
		- Default = ```ner+re```
	- ```sample_size```
		- Number of sentences from ctg-dataset to perform inference on
		- None = inference on all sentences
		- Default = ```None```

 - Example uses
```console
	python src/pipeline_inference.py \
		--ner_model_path models/NER_chia_seq_v1 \
		--re_model_path models/RE_clinicalBERT_globalAndTrainDownsampling
```
```console
	python src/pipeline_inference.py \
		--data_dir data_ctg/parsedCTG_sentlevel.csv \
		--output_dir results \
		--ner_model_path models/NER_chia_seq_v1 \
		--re_model_path models/RE_clinicalBERT_globalAndTrainDownsampling
		--mode ner+re \
		--sample_size None

```


## Data analysis

### ```data_statistics.py```
- Description
	- Provides descriptive statistics for parsed datasets from ```ner_parseChia.py``` and ```re_parseChia.py```
	- Metrics: Sentence counts, entity span density, average token counts per entity type, relation class balance ratios
 
- Arguments
	- ```ner_dir```
		- Path to the directory with the parsed NER dataset
		- None = do not get statistics for NER dataset 
		- Default = ```../data/chia_without_scope_parsedNER_v1```
	- ```re_dir```
		- Path to the directory with the parsed RE dataset
		- None = do not get statistics for RE dataset 
		- Default = ```../data/chia_without_scope_parsedRE_v1```
 
- Example uses
```console
	python src/data_statistics.py
```
```console
	python src/data_statistics.py \
		--ner_dir data/chia_without_scope_parsedNER_v1 \
		--re_dir data/chia_without_scope_parsedNER_v1
```



## Requirements
- Python 3.8+
- PyTorch
- Transformers & Datasets (HuggingFace)
- Optuna (for HPO)
- Evaluate & Seqeval
- Scikit-learn, Pandas, Numpy
