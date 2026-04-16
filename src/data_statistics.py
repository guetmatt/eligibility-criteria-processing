"""
Dataset Statistics Module

Descriptive statistics for datasets processed from the chia dataset.
- label distributions, class frequency
- sentence / sequence length
- entity density
- relation-to-entity ratios
- class (im)balance statistics

Usage (argument values to be adjusted by user):
    python data_statistics.py --ner_dir ../data/ner_dataset --re_dir ../data/re_dataset
"""


# --- imports --- #
import os
import sys
import logging
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from datasets import load_from_disk


# --- logging setup --- #
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)



# --- functions --- #
def load_label_map(input_dir: str, filename: str = "label_map.json"):
    """
    Loads the label mapping JSON file from the dataset directory.

    Args:
        input_dir (str): Path to the directory containing the label map.
        filename (str): Name of the JSON file.

    Returns:
        id2label, label2id (tuple): A tuple containing label mappings (label2id, id2label).
        """
    # path to label mapping file
    path = os.path.join(input_dir, filename)
    
    if not os.path.exists(path):
        logger.error(f"Label mapping file not found at {path}")
        raise FileNotFoundError(f"Label mapping file not found at {path}")
    
    # open label mapping file
    with open(path, "r", encoding="UTF-8") as f:
        data = json.load(f)
    
    label2id = data["label2id"]
    # JSON saves keys as strings
    # --> convert to int for id2label
    id2label = {int(k): v for k, v in data["id2label"].items()}
    
    return id2label, label2id



def get_ner_spans(labels: list, id2label: dict):
    """
    Identifies entity spans from BIO-formatted entity labels.
    
    Args:
        labels (list): List of label IDs for a single sentence
        id2label (dict): Label mapping from label_ids to label names
        
    Returns:
        entity_spans (list): List of span dictionaries with keys "type", "tok_count"
    """
    entity_spans = list()
    current_span_type = None
    current_span_len = 0

    for label_id in labels:
        # no entity: "O" (0), special tokens (-100)
        if label_id == -100 or label_id == 0:
            if current_span_type:
                entity_spans.append({"type": current_span_type, "tok_count": current_span_len})
                current_span_type = None
                current_span_len = 0
            continue
        
        # labeled entities
        label_name = id2label.get(label_id, "O")
        if "B-" in label_name or "I-" in label_name:
            prefix, entity_type = label_name.split("-", 1)
        else:
            prefix, entity_type = ("O", "O")

        # new entity span
        if prefix == "B":
            # sprevious span was in progress
            # --> close it
            if current_span_type:
                entity_spans.append({"type": current_span_type, "tok_count": current_span_len})
            
            # start new span
            current_span_type = entity_type
            current_span_len = 1
        
        elif prefix == "I" and current_span_type == entity_type:
            # continue current span
            current_span_len += 1
        
        else:
            #  I-label without B-label
            if current_span_type:
                entity_spans.append({"type": current_span_type, "tok_count": current_span_len})
            current_span_type = None
            current_span_len = 0

    # final span check for last entity span
    if current_span_type:
        entity_spans.append({"type": current_span_type, "tok_count": current_span_len})
        
    return entity_spans



def analyze_ner_split(split_name: str, dataset: pd.DataFrame, id2label: dict):
    """
    Calculates and logs statistics for an NER dataset split.

    Args:
        split_name (str): Name of dataset split (train/validation/test)
        dataset (pd.DataFrame): Dataset split
        id2label (dict): Label mapping from label_ids to label names
    """
    logger.info(f"Analyzing NER split: {split_name.upper()}")
    
    # count entity spans
    all_spans = list()
    for labels in dataset["labels"]:
        all_spans.extend(get_ner_spans(labels, id2label))
    
    # counts and lenghts per entity type
    type_stats = defaultdict(list)
    for span in all_spans:
        type_stats[span["type"]].append(span["tok_count"])
    
    # print stats
    print(f"\n--- NER {split_name.upper()} STATS ---")
    print(f"Total sentences:    {len(dataset)}")
    print(f"Total entity spans: {len(all_spans)}")
    print(f"{"Entity type":<25} {"Spans":<8} {"Avg token count":<18}")
    print("-" * 55)
    
    # sort entity types by frequency
    sorted_types = sorted(type_stats.items(), key=lambda x: len(x[1]), reverse=True)
    for ent_type, tok_count in sorted_types:
        avg_count = np.mean(tok_count)
        print(f"{ent_type:<25} {len(tok_count):<8} {avg_count:<18.2f}")
    
    # label counts
    num_samples = len(dataset)
    # get flat list of all label_ids
    # ignore "O" (0), special tokens (-100)
    all_label_ids = list()
    for label_list in dataset["labels"]:
        for label_id in label_list:
            if label_id > 0:
                all_label_ids.append(label_id)
        
    return None



def analyze_re_split(split_name: str, dataset: pd.DataFrame, id2label: dict):
    """
    Calculates and logs statistics for an RE dataset split.

    Args:
        split_name (str): Name of dataset split (train/validation/test)
        dataset (pd.DataFrame): Dataset split
        id2label (dict): Label mapping from label_ids to label names
    """
    logger.info(f"Analyzing RE Split: {split_name.upper()}")
    
    # entity pair counts
    total_pairs = len(dataset)
    # label counts
    # ignore "NO_RELATION" (0), special tokens (-100)
    pos_samples = dataset[dataset["label"] > 0]
    neg_count = len(dataset[dataset["label"] == 0])
    pos_count = len(pos_samples)

    # print statistics
    print(f"\n--- RE {split_name.upper()} STATS ---")
    print(f"Total candidate entity pairs: {total_pairs}")
    print(f"Positive samples:    {pos_count}")
    print(f"Negative samples:    {neg_count}")
    
    if pos_count > 0:
        print(f"Class balance ratio:   1 positive : {neg_count/pos_count:.2f} negative")

    # relation type distribution
    counts = Counter(pos_samples["label"])
    print(f"\n{"Relation Type":<25} {"Count":<8} {"%":<6}")
    for label_id, count in counts.most_common():
        name = id2label.get(label_id, f"ID_{label_id}")
        percent = (count / pos_count) * 100
        print(f"{name:<25} {count:<8} {percent:>5.1f}%")



def main(args):
    """
    Main execution for generating statistics.
    Arguments provided via command line.
    
    Args:
        ner_dir (str): Path to the parsed NER dataset, default="../data/chia_without_scope_parsedNER_v1"
        re_dir (str): Path to the parsed RE dataset, default="../data/chia_without_scope_parsedRE_v1"
    """
    
    # NER dataset sats
    if args.ner_dir:
        if not os.path.exists(args.ner_dir):
            logger.error(f"NER directory not found: {args.ner_dir}")
        else:
            dataset_ner = load_from_disk(args.ner_dir)
            id2label_ner, label2id_ner = load_label_map(args.ner_dir)
            for split in dataset_ner.keys():
                analyze_ner_split(split, dataset_ner[split].to_pandas(), id2label_ner)

    # RE dataset stats
    if args.re_dir:
        if not os.path.exists(args.re_dir):
            logger.error(f"RE directory not found: {args.re_dir}")
        else:
            ds_re = load_from_disk(args.re_dir)
            id2label_re, label2id_re = load_label_map(args.re_dir)
            for split in ds_re.keys():
                analyze_re_split(split, ds_re[split].to_pandas(), id2label_re)
                


# boilerplate
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate descriptive statistics for NER and RE datasets.")
    
    # get system paths
    # and then define default paths
    # __file__ = src/ner_parseChia.py
    # .parent  = src/
    # .parent.parent = project_root/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_NER = PROJECT_ROOT / "data" / "chia_without_scope_parsedNER_v1"
    DEFAULT_RE = PROJECT_ROOT / "data" / "chia_wtihout_scope_parsedRE_v1"


    # paths
    parser.add_argument("--ner_dir", type=str, default=str(DEFAULT_NER),
                        help="Path to the parsed NER dataset directory.")
    parser.add_argument("--re_dir", type=str, default=str(DEFAULT_RE),
                        help="Path to the parsed RE dataset directory.")

    # check if interactive environment (e.g. jupyter notebook)
    # or command line
    if "ipykernel" in sys.modules:
        # notebook mode
        args = parser.parse_args([])
    else:
        # command line mode
        args = parser.parse_args()
    
    main(args)