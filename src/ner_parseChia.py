"""
Chia Dataset Preprocessing Module for Named Entity Recognition (NER)

This script parses the Chia dataset (chia_without_scope) for NER tasks.
--> file processing, line-level label alignment, tokenization, iterative stratification, train/test/eval splitting

Usage (argument values to be adjusted by user):
    python ner_parseChia.py --data_dir ./data/chia_raw --output_dir ./data/processed_ner
"""



# --- imports --- #
import os
import glob
import re
import json
import logging
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit



# --- logging setup --- #
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)



# --- global constants --- #

# set of accepted entity labels
# based on annotation model of chia dataset
ACCEPTED_ENTITY_TYPES = {
    "Condition", "Device", "Drug", "Measurement",
    "Observation", "Person", "Procedure", "Visit",
    "Temporal", "Value", "Negation", "Qualifier",
    "Multiplier", "Reference_point", "Mood",
    "Non-query-able", "Post-eligibility", "Informed_consent",
    "Pregnancy_considerations", "Parsing_error", "Non-representable",
    "Competing_trial", "Context_Error", "Subjective_judgement", 
    "Not_a_criteria", "Undefined_semantics", "Intoxication_considerations"
    }



# --- preprocessing functions --- #

def parse_brat_file(ann_path: str):
    """
    Parses a brat annotation file (.ann) from the chia_without_scope dataset
    to extract named entities with character indices.
    Discontinuous index spans (e.g. 10 15; 20 25) are flattened into
    distinct entity mentions sharing the same id, text and type.
    
    Args:
        ann_path (str): File path to the .ann file from chia
    
    Returns:
        data_parsed (list[dict[str, any]]): Sorted list of entity dictionaries with keys:
            - id (str): Entity id (e.g. "T1")
            - type (str): Entitity label (e.g. "Condition")
            - start (int): Character index of entity start
            - end (int): Character index of entity end
            - text (str): Text content of entity    
    """   
    entities = list()
    
    try:
        with open(ann_path, "r", encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                
                # ignore non-entity annotations
                if not line.startswith("T"):
                    continue
                
                # get entity parts
                parts = line.split("\t")
                # ignore malformed annotations
                if len(parts) < 3:
                    continue
                entity_id = parts[0]
                entity_type_and_indices = parts[1]
                entity_text = parts[2]
                
                # parse entity type and indices
                # example discontinuous entity:
                # "Condition 1311 1318; 1334 1341"
                meta_parts = entity_type_and_indices.split(" ")
                index_string = " ".join(meta_parts[1:])
                entity_type = meta_parts[0]
                # ignore annotations with unwanted entity types
                if entity_type not in ACCEPTED_ENTITY_TYPES:
                    continue
                
                # capture start/end index-pairs
                start_end_pairs = re.findall(r"(\d+)\s(\d+)", index_string)
                
                # create entities for each start/end index-pair
                for start, end in start_end_pairs:
                    entities.append({
                        "id": entity_id,
                        "type": entity_type,
                        "start": int(start),
                        "end": int(end),
                        "text": entity_text
                    })
    
    except FileNotFoundError:
        logger.warning(f"File not found: {ann_path}")
    
    # sort entities by start position
    data_parsed = sorted(entities, key=lambda x: x["start"])
    
    return data_parsed



def split_text_and_realign_entities(text: str, entities: list):
    """
    Splits the file text into lines and realigns entity indices
    relative to line start.
    Converts document-level entity indices to sentence-level indices
    due to sentence-level processing in NER task.
    
    Args:
        text (str): Text content of a chia .txt file
        entities (list[dict]): List of entity dictionaries with document-level indices 
    
    Returns:
        data_linelevel (list[dict]): List of line-level sample dictionaries with keys:
            - "text" (str): Text content of a line
            - "entities" (list[dict]): List of entity dictionaries appearing in the current line,
                with character indices realigned to line start
    """
    linelevel_data = list()
    current_global_idx = 0
    
    # lines splitted by newline
    lines = text.split("\n")
    
    for line in lines:
        # get line indices
        line_len = len(line)
        line_end_idx = current_global_idx + line_len
        local_entities = list()
        
        # find entities within current line
        for ent in entities:
            if ent["start"] >= current_global_idx and ent["end"] <= line_end_idx:
                # calculate line-level indices
                local_start = ent["start"] - current_global_idx
                local_end = ent["end"] - current_global_idx
                
                # crreate list of entity-dicts in current line
                local_entities.append({
                    "id": ent["id"],
                    "type": ent["type"],
                    "start": local_start,
                    "end": local_end,
                    "text": ent["text"]
                })
        
        # collect non-empty line data
        if line.strip():
            linelevel_data.append({
                "text": line,
                "entities": local_entities
            })
        
        # update global index for next line
        # +1 for newline-character
        current_global_idx += line_len + 1
    
    return linelevel_data



def process_file_line_by_line(
    txt_path: str,
    ann_path: str,
    tokenizer: AutoTokenizer,
    label2id: dict[str, int],
    max_len: int
    ):
    """
    Processes a pair of .txt and .ann files from the chia_without_scope dataset
    into tokenized and aligned samples with BIO labels.
    Reads text, parses entities, splits text into lines, realigns entity indices,
    tokenizes lines, labels and aligns subword tokens. 
    
    Args:
        txt_path (str): Path to the .txt file
        ann_path (str): Path to the .ann file
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace 
        label2id (dict[str, int]): Mapping from labels to ids, updated in-place
        max_len (int): Maximum sequence length for tokenization

    Returns:
        processed_data (list[dict]): list of processed linelevel dictionaries with keys:
            - "input_ids" (list[int]): token ids of the tokenized line
            - "attention_mask" (list[int]): attention mask of the tokenized line
            - "labels" (list[int]): BIO label_ids aligned to token ids
            - "filename" (str): original filename of the .txt file
            - "sentence_text" (str): text content of the line 

    """
    with open(txt_path, "r", encoding="UTF-8") as f:
        text = f.read()
    
    # parse and align entities
    global_entities = parse_brat_file(ann_path)
    linelevel_data = split_text_and_realign_entities(text, global_entities)
    
    processed_data = list()
    
    for item in linelevel_data:
        line_text = item["text"]
        line_entities = item["entities"]
        
        # tokenization
        tokenized_line = tokenizer(
            line_text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_offsets_mapping=True
        )
        index_mapping = tokenized_line["offset_mapping"]
        
        # initialize labels to id of "O" (outside)
        # for each token in line
        labels = [label2id["O"]] * len(tokenized_line["input_ids"])
        
        # align entity indices with token-level BIO labels
        for ent in line_entities:
            entity_type = ent["type"]
            b_label = f"B-{entity_type}"
            i_label = f"I-{entity_type}"
            
            # dynamically update label2id mapping
            # -> add new labels if not present
            if b_label not in label2id:
                label2id[b_label] = len(label2id)
            if i_label not in label2id:
                label2id[i_label] = len(label2id)
            
            # get label ids
            b_id = label2id[b_label]
            i_id = label2id[i_label]
            
            # label subtokens with corresponding BIO labels
            found_start = False
            for idx, (start, end) in enumerate(index_mapping):
                # skip special tokens ([CLS], [PAD], etc.)
                if start == 0 and end == 0:
                    continue
                
                # token within entity span
                if start >= ent["start"] and end <= ent["end"]:
                    if not found_start:
                        labels[idx] = b_id
                        found_start = True
                    else:
                        labels[idx] = i_id
                
                # edge case: entity-ids include token boundaries
                # treat as beggining of entity
                elif start < ent["start"] and end > ent["start"]:
                    labels[idx] = b_id
                    found_start = True

        # collect processed line samples
        processed_data.append({
            "input_ids": tokenized_line["input_ids"],
            "attention_mask": tokenized_line["attention_mask"],
            "labels": labels,
            "filename": os.path.basename(txt_path),
            "sentence_text": line_text
        })
    
    return processed_data



def get_entity_presence_matrix(dataset: Dataset, label_column: str = "labels"):
    """
    Constructs a binary matrix indicating the presence of entity types per sample.
    Used for iterative stratification to ensure balanced labels across splits.
    
    Args:
        dataset: HuggingFace dataset
        label_column: Name of the column containing label_ids
    
    Returns:
        matrix (np.array): Binary matrix of shape (n_samples, n_entity_types)
            where 1 indicates the entity type is present in the sentence.
    """
    # identify unique labels in dataset
    # excluding 0="O" (non-entity label in BIO-scheme)
    all_labels = list()
    for sent_labels in dataset["labels"]:
        all_labels.extend(sent_labels)
    unique_labels = set(all_labels)
    unique_labels.remove(0)
    
    # map label_ids to matrix columns
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # build matrix
    # rows = n sentences
    # columns = n entity types (labels)
    matrix = np.zeros((len(dataset), len(unique_labels)))
    for row_idx, labels in enumerate(dataset[label_column]):
        sent_labels = set(label for label in labels if label != 0)
        for label in sent_labels:
            if label in label_to_idx:
                matrix[row_idx, label_to_idx[label]] = 1

    return matrix



# --- main functions for dataset processing and stratification --- #

def load_chia_dataset(data_dir: str, model_checkpoint: str, max_len: int):
    """
    Loads, parses and formats the chia_without_scope dataset for NER tasks.
    
    Args:
        data_dir (str): path to dictionary containing .txt and .ann files
        model_checkpoint (str): HuggingFace model checkpoint for tokenizer
        max_len (int): Maximum token length for tokenizer
    
    Returns:
        dataset (Dataset): HuggingFace dataset containing processed samples
        label2id (dict[str, int]): mapping from labels to label_ids
        id2label (dict[int, str]): mapping from label_ids to labels
    """
    
    # initialize tokenizer and data structures
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    label2id = {"O": 0}
    data_lines = list()
    
    # get paths of all .txt files
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    logger.info(f"Found {len(txt_files)} files in {data_dir}.\nStarting processing...")
    
    for txt_path in txt_files:
        # corresponding .ann file path
        ann_path = txt_path.replace(".txt", ".ann")
        
        # ingore txt file without ann file
        if not os.path.exists(ann_path):
            continue
        
        # retrieve criteria type from filename
        # default to "unknown"
        file_name = os.path.basename(txt_path)
        if "_exc" in file_name:
            criteria_type = "exclusion"
        elif "_inc" in file_name:
            criteria_type = "inclusion"
        else:
            criteria_type = "unknown"
        
        # process file-pair line by line
        # returns a list of line-level processed data
        # label2id updated in-place
        processed_lines = process_file_line_by_line(txt_path, ann_path, tokenizer, label2id, max_len)
        
        # add criteria_type to each line sample
        # and collect processed lines in a list
        for line in processed_lines:
            line["criteria_type"] = criteria_type
            data_lines.append(line)
        
    # convert lsit of dicts to HuggingFace dataset
    # via Pandas DataFrame for convenience
    df = pd.DataFrame(data_lines)
    dataset = Dataset.from_pandas(df)
    
    # convert criteria_type to ClassLabel
    # for potential stratification purposes
    features = dataset.features.copy()
    features["criteria_type"] = ClassLabel(names=["exclusion", "inclusion", "unknown"])
    dataset = dataset.cast_column("criteria_type", features["criteria_type"])
    
    # create id2label mapping from label2id mapping
    id2label = {val: key for key, val in label2id.items()} 
    
    logger.info(f"Total samples processed: {len(dataset)}")
    
    return dataset, label2id, id2label



def save_label_map(label2id: dict, id2label: dict, output_dir: str, filename: str="label_map.json"):
    """
    Saves label mappings to a JSON file.

    Args:
        label2id (dict): mapping of labels to label_ids
        id2label (dict): mapping of label_ids to labels
        output_dir (str): directory to save json-file to
        filename (str): name of the output file
            
    Returns:
        None
    """
    # create output directory if not existing
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    data = {
        "label2id": label2id,
        "id2label": id2label
    }

    try:
        with open(path, "w", encoding="UTF-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Label map saved to {path}")
    except Exception as e:
        logger.error(f"Error saving label map: {e}")
        
    return None



def split_and_save_dataset_iterative(dataset: Dataset, output_dir: str, seed: int = 42, label_column: str = "labels"):
    """
    Splits the dataset into train, validation, and test sets and saves them to disk.
    Applies iterative stratification on labels for label-balance in all splits. 
    Splitting has to be done in two steps due to functionality of library.
    Splits: 10% Test, 10% Validation, 80% Train

    Args:
        dataset (Dataset): complete and processed HuggingFace dataset
        output_dir (str): directory path to save the splitted dataset to
        seed (int, optional): random seed for reproducibility
        label_column (str, optional): name of column containing label_ids
    
    Returns:
        final_dataset (Dataset): HuggingFace dataset, splitted and stratified
    """
    # multi-hot-matrix for iterative stratification
    matrix = get_entity_presence_matrix(dataset, label_column=label_column)
    # sklearn api expects (X, y) format for stratificaiton
    # -> dummy X-array
    X = np.zeros(len(dataset))
    
    # split 1 - test set (10%)
    msss_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    # msss.split returns indices
    # -> only need first split
    train_val_idx, test_idx = next(msss_test.split(X, matrix))
    test_dataset = dataset.select(test_idx)
    remaining_dataset = dataset.select(train_val_idx)
    
    # slice matrix to match remaining dataset
    # for second split
    matrix_remaining = matrix[train_val_idx]
    x_remaining = X[train_val_idx]
    
    # split 2 - validation set (10% of remaining -> 1/9 of total)
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1/9, random_state=seed)
    train_idx, val_idx = next(msss_val.split(x_remaining, matrix_remaining))
    
    train_dataset = remaining_dataset.select(train_idx)
    val_dataset = remaining_dataset.select(val_idx)
    
    # combine splits into full dataset
    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    # save to disk
    logger.info(f"Saving dataset to {output_dir}...")
    logger.info(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    final_dataset.save_to_disk(output_dir)
    
    return final_dataset



def main(args):
    """
    Main parsing pipeline execution.
    Arguments provided via command line.
    
    Args:
        data_dir (str): Path to directory containing raw .txt and .ann files, default="../data/chia_without_scope"
        output_dir (str): Directory to save the parsed chia dataset, default="../data/chia_without_scope_parsedNER_v1"
        model_checkpoint (str): HuggingFace model checkpoint for tokenizer, default="emilyalsentzer/Bio_ClinicalBERT
        max_len (int): Maximum token length for padding/truncation, default=128
        seed (int): Random seed for stratification, default=42
    """
    logger.info(f"Data Directory: {args.data_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # (1) load dataset
    if not os.path.exists(args.data_dir):
        logger.critical(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    dataset, label2id, id2label = load_chia_dataset(
        args.data_dir,
        args.model_checkpoint,
        args.max_len
    )
    if len(dataset) == 0:
        logger.warning("No data processed. Check input directory.")
        sys.exit(1)
    
    # (2) stratify, split, and save parsed dataset
    split_and_save_dataset_iterative(dataset, args.output_dir, seed=args.seed)
    
    # (3) save label mappings
    save_label_map(label2id, id2label, args.output_dir)
    logger.info(f"Parsed dataset and label mapping saved to {args.output_dir}")    

    logger.info(f"Pipeline completed successfully.")



# --- main execution --- #
# boilerplate    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the chia dataset for NER.")   
    
    # get system paths
    # and then define default paths
    # __file__ = src/ner_parseChia.py
    # .parent  = src/
    # .parent.parent = project_root/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "chia_without_scope"
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "chia_without_scope_parsedNER_v1"
    
    # paths
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                       help="Path to directory containing raw .txt and .ann files.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save the parsed chia dataset.")
   
   # configuration
    parser.add_argument("--model_checkpoint", type=str, default="emilyalsentzer/Bio_ClinicalBERT",
                        help="HuggingFace model checkpoint for tokenizer.")
    parser.add_argument("--max_len", type=int, default=128,
                        help="Maximum token length for padding/truncation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stratification.")
    
    
    # check if interactive environment (e.g. jupyter notebook)
    # or command line
    if "ipykernel" in sys.modules:
        # notebook mode
        args = parser.parse_args([])
    else:
        # command line mode
        args = parser.parse_args()

    main(args)