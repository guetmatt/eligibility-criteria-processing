"""
Chia Dataset Preprocessing Module for Relation Extraction (RE)

This script parses the Chia dataset (chia_without_scope) for RE tasks.
--> entity extraction, relation extraction, binary relation generation (including negative samples)
--> downsampling of negative examples ("NO_RELATION") to handle class imbalance
--> stratification, splitting and saving the parsed dataset

Usage (argument values to be adjusted by user):
    python re_parseChia.py --data_dir ../data/chia_without_scope --output_dir ../data/chia_without_scope_parsedRE_v1
"""

# --- imports --- #
import os
import glob
import re
import json
import logging
import argparse
import sys
import random
import itertools
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel



# --- logging setup --- #
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
    )
logger = logging.getLogger(__name__)


# --- constants --- #
ACCEPTED_RELATION_TYPES = {
    "AND",
    "OR",
    "Subsumes",
    "Has_negation",
    "Has_multiplier",
    "Has_qualifier",
    "Has_value",
    "Has_temporal",
    "Has_index",
    "Has_mood",
    "Has_context",
    "NO_RELATION"
}



# --- helper funcitons --- #

def parse_ann_file(ann_path: str):
    """
    Parses an .ann file from chia_without_scope
    for entities (T), binary relations (R), and n-ary or-relations (*).

    Args:
        ann_path (str): Path to the .ann annotation file

    Returns:
        tuple[dict, list]:
        - entities: Dictionary of entities {entity_id: {type, start, end, text}}
        - relations: List of relation-tuples (relation_type, arg1_id, arg2_id)
    """
    entities = {}
    relations = []
    
    try:
        # open and read file
        with open(ann_path, "r", encoding="UTF-8") as f:
            for line in f:
                line = line.strip()
                # ignore empty lines
                if not line:
                    continue
                
                # parse entities
                if line.startswith("T"):
                    parts = line.split("\t")
                    # ignore malformed entity annotations
                    if len(parts) < 3:
                        continue
                    
                    entity_id = parts[0]
                    entity_type = parts[1].split(' ')[0]
                    
                    # discontinuous spans, e.g. "1311 1318;1334 1341"
                    # -> take min start, max end
                    indices = [int(x) for x in re.findall(r"\d+", parts[1])]
                    # ignore malformed annotations
                    if not indices:
                        continue
                    
                    # dictionary entry for current entity
                    entities[entity_id] = {
                        "id": entity_id,
                        "type": entity_type,
                        "start": min(indices),
                        "end": max(indices),
                        "text": parts[2]
                    }
                    
                # binary relations
                elif line.startswith("R"):
                    parts = line.split("\t")
                    # ignore malformed annotations
                    if len(parts) < 2:
                        continue
                    
                    args_part = parts[1].split(" ")
                    relation_type = args_part[0]
                    
                    # relation arguments
                    arg1, arg2 = None, None
                    for arg in args_part:
                        if arg.startswith("Arg1:"):
                            arg1 = arg.split(":")[1]
                        elif arg.startswith("Arg2:"):
                            arg2 = arg.split(":")[1]
                    
                    # sound relation found
                    # -> list entry for relation
                    if arg1 and arg2:
                        relations.append((relation_type, arg1, arg2))
                
                # n-ary or-relations
                # -> decompose into binary relations
                elif line.startswith("*"):
                    parts = line.split("\t")
                    # ignore malformed lines
                    if len(parts) < 2:
                        continue
                    
                    args_part = parts[1].split(" ")
                    relation_type = args_part[0]
                    # get relation arguments
                    args = args_part[1:]
                    # generate all pairwise permutations
                    # for the n-ary or-set
                    if len(args) >= 2:
                        for arg1, arg2 in itertools.permutations(args, 2):
                            relations.append((relation_type, arg1, arg2))

    except Exception as e:
        logger.warning(f"Error parsing {ann_path}: {e}")

    return entities, relations



def process_files(data_dir: str, global_downsample_rate: float = None):
    """
    Parses all .txt and .ann files the data_dir to generate relation samples.
    Extracts sentences and generates candidate pairs for all entities within the same sentence.
    
    Args:
        data_dir (str): Directory containing source files from chia dataset
        global_downsample_rate (float): If set, randomly drops negative "NO_RELATION" 
            samples across the entire dataset during parsing. 1.0=keep all. default=None

    Returns:
        Tuple[pd.DataFrame, set]: 
        - relations_df: DataFrame containing all generated relation samples
        - all_relation_types: Set of all unique relation types found (excluding "NO_RELATION" initially)
    """
    # generate list of paths to txt-files
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    samples = list()
    all_relation_types = set()
    
    # statistics counters
    count_pos = 0
    count_neg_kept = 0
    count_neg_dropped = 0
    count_filtered_out = 0
    relation_types_skipped = set()
    
    logger.info(f"Parsing {len(txt_files)} files for relation extraction...")
    if global_downsample_rate and global_downsample_rate < 1.0:
        logger.info(f"Applying global negative downsampling rate to complete dataset: {global_downsample_rate}")

    # process each pair of ann- and txt-files
    for txt_path in txt_files:
        ann_path = txt_path.replace(".txt", ".ann")
        if not os.path.exists(ann_path):
            continue
        
        # extract criteria type from filename
        # default to "unknown"
        file_name = os.path.basename(txt_path)
        if "_exc" in file_name:
            criteria_type = "exclusion"
        elif "_inc" in file_name:
            criteria_type = "inclusion"
        else:
            criteria_type = "unknown"
        
        # open and read txt-file
        with open(txt_path, "r", encoding="UTF-8") as f:
            text = f.read()
        
        # parse annotations
        entities, relations = parse_ann_file(ann_path)
        
        # update known relation types
        for rel in relations:
            all_relation_types.add(rel[0])
        
        # create lookup: (arg1_id, arg2_id) -> relation_type
        # --> assumes unique relation per directional pair
        relation_lookup = dict()
        for r_type, arg1, arg2 in relations:
            if r_type in ACCEPTED_RELATION_TYPES:
                relation_lookup[(arg1, arg2)] = r_type
                all_relation_types.add(r_type)
            else:
                count_filtered_out += 1
                relation_types_skipped.add(r_type)
        
        # split text into lines (sentences)
        # --> ignores cross-sentence relations
        lines = text.split('\n')
        
        global_offset = 0
        for line in lines:
            line_len = len(line)
            # line-level indices
            line_end = global_offset + line_len
            
            # entities in current line
            line_entities = []
            for ent_id, ent in entities.items():
                if ent["start"] >= global_offset and ent["end"] <= line_end:
                    line_entities.append(ent)
            
            # relation generation with entity pairs
            # less than two entities in line --> no relation
            if len(line_entities) >= 2:
                # generate all permutations (arg1, arg2)
                # because binary relations are directional
                for e1, e2 in itertools.permutations(line_entities, 2):
                    # ground truth label
                    # or "NO_RELATION"
                    label = relation_lookup.get((e1["id"], e2["id"]), "NO_RELATION")
                    
                    # ignore unaccepted relation labels
                    if label not in ACCEPTED_RELATION_TYPES:
                        continue
                    
                    # global downsampling of negative samples
                    # randomly drop 0.x amount of negative samples in complete dataset
                    # to handle class imbalance / negative sample explosion 
                    if label == "NO_RELATION" and global_downsample_rate is not None and global_downsample_rate < 1.0:
                        if random.random() > global_downsample_rate:
                            count_neg_dropped += 1
                            continue
                        count_neg_kept += 1
                    elif label == "NO_RELATION":
                        count_neg_kept += 1
                    else:
                        count_pos += 1
                    
                    # calculate local indices relative to line start
                    # for injecting special tokens [E1]...[/E1] later
                    samples.append({
                        "text": line,
                        "e1_start": e1["start"] - global_offset,
                        "e1_end": e1["end"] - global_offset,
                        "e1_type": e1["type"],
                        "e2_start": e2["start"] - global_offset,
                        "e2_end": e2["end"] - global_offset,
                        "e2_type": e2["type"],
                        "label": label,
                        "criteria_type": criteria_type,
                        "filename": file_name
                    })
            
            # update global offset
            # +1 for newline character
            global_offset += line_len + 1 
    
    # log statistics
    total_processed = count_pos + count_neg_kept + count_neg_dropped
    logger.info("--- Parsing Statistics ---")
    logger.info(f"Positive relations: {count_pos}")
    logger.info(f"Negatives kept:     {count_neg_kept}")
    logger.info(f"Negatives dropped:  {count_neg_dropped}")
    logger.info(f"Unknown relations: {count_filtered_out} ({relation_types_skipped})")
    logger.info(f"Total pairs found:  {total_processed}")
    if count_pos > 0:
        logger.info(f"Final Neg/Pos Ratio: {count_neg_kept/count_pos:.2f} : 1")

    relations_df = pd.DataFrame(samples)

    return relations_df, all_relation_types



def print_label_distribution(dataset: DatasetDict, label_names: list):
    """
    Prints label counts and label distribution statistics for dataset splits.
    
    Args:
        dataset (DatasetDict): Processed dataset
        label_names (list): List of all relation types    
    
    Returns:
        None
    """
    print("\n" + "="*50)
    print("DATASET LABEL DISTRIBUTION")
    print("="*50)
    
    # print statistics for train/test/eval splits
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        total = len(split_data)
        print(f"\nSplit: {split_name.upper()} (n={total})")
        print(f"{'ID':<4} {'Label Name':<25} {'Count':<8} {'%':<6}")
        print("-" * 45)

        # count labels
        label_ids = split_data["label"]
        counts = Counter(label_ids)
        
        # print label counts sorted by id
        for label_id in sorted(counts.keys()):
            count = counts[label_id]
            percent = (count / total) * 100
            if label_id < len(label_names):
                name = label_names[label_id]
            else:
                name = "unknown"
            print(f"{label_id:<4} {name:<25} {count:<8} {percent:>5.1f}%")



def split_and_save(
    dataset: Dataset, 
    relation_types: list, 
    output_dir: str, 
    train_downsample_rate: float = None,
    seed: int = 42
):
    """
    Splits the dataset intro train/test/eval splits
    and optionally downsamples only the training set.
    Splitting has to be done in two steps due to library functionality.
    Splits: 10% Test, 10% Eval, 80% Train
    
    Args:
        dataset (Dataset): HuggingFace dataset created by parsing chia dataset
        relation_types (list): List of all relation types
        output_dir (str): Directory to save the dataset
        train_downsample_rate: If set (0.0-1.0), keeps only this fraction of negative samples 
                               in the train split (not test/eval)
        seed (int): Random seed for stratification
    
    Returns:
        final_dataset (Dataset): Splitted (and downsampled) dataset
        id2label (dict): Mapping dicitonary from label_ids to label names
        label2id (dict): Mapping dictionary from label names to label_ids
    """
    logger.info("Splitting dataset into Train (80%), Validation (10%), Test (10%)...")
    
    # first split
    split_1 = dataset.train_test_split(test_size=0.1, stratify_by_column="label", seed=seed)
    test_dataset = split_1["test"]
    remaining = split_1["train"]
    
    # second split
    split_2 = remaining.train_test_split(test_size=0.1111, stratify_by_column="label", seed=seed)
    train_dataset = split_2["train"]
    eval_dataset = split_2["test"]
    
    # downsampling of training split
    if train_downsample_rate is not None and train_downsample_rate < 1.0:
        logger.info(f"Downsampling training split negative samples (rate={train_downsample_rate})...")
        logger.info(f"  Train size before: {len(train_dataset)}")
        
        def filter_negatives(example):
            # keep positive relations
            if example["label"] != 0:
                return True
            # return random % of negative relations
            # wrt to downsampling rate
            return random.random() < train_downsample_rate
        
        # apply downsampling filter
        train_dataset = train_dataset.filter(filter_negatives)
        logger.info(f"  Train size after downsampling:  {len(train_dataset)}")

    final_dataset = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset,
        "test": test_dataset
    })
    
    # save dataset to disk
    logger.info(f"Saving dataset to {output_dir}")
    final_dataset.save_to_disk(output_dir)
    
    # create and save label mappings
    label2id = {l: i for i, l in enumerate(relation_types)}
    id2label = {i: l for l, i in label2id.items()}
    map_path = os.path.join(output_dir, "label_map.json")
    with open(map_path, "w", encoding="UTF-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)
    logger.info(f"Label map saved to {map_path}")

    return final_dataset, id2label, label2id



# --- main execution funcion --- #
def main(args):
    """
    Main pipeline execution.
    Parses the chia_without_scope dataset for relation extraction,
    creates and saves a splitted dataset.
    Arguments provided via command line.
    
    Args:
        data_dir (str): Path to directory containing raw .txt and .ann files, default="../data/chia_without_scope"
        output_dir (str): Directory to save the processed chia dataset, default="../data/chia_without_scope_parsedRE_v1"
        train_downsample_rate (float): [0.0-1.0, None] Ratio of negative samples (NO_RELATION) to keep in the training split. 0.2=keep 20% of negatives, 1.0 or None=keep all, default=0.2
        global_downsample_rate (float): [0.0-1.0, None] Ratio of negative samples (NO_RELATION) to keep across the entire dataset during parsing (affects train, val, and test). 0.2=keep 20% of negatives, 1.0 or None=keep all, default=None"
        seed (int): Random seed for stratification and reproducibility, default=42
    """
    # (1) setup
    random.seed(args.seed)
    
    # (2) parse chia dataset files
    if not os.path.exists(args.data_dir):
        logger.critical(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    df, relation_types = process_files(args.data_dir, global_downsample_rate=args.global_downsample_rate)

    if df.empty:
        logger.error("No data found or parsed.")
        sys.exit(1)

    # (3) clean relation labels
    # remove labels with too few examples (<=1, cannot be stratified)
    counts = df["label"].value_counts()
    valid_labels = set(counts[counts > 1].index)
    valid_labels = valid_labels.intersection(ACCEPTED_RELATION_TYPES)
    dropped_labels = set(counts[counts <= 1].index)
    if dropped_labels:
        logger.warning(f"Dropping labels with only 1 example: {dropped_labels}")
    df = df[df["label"].isin(valid_labels)]
    
    # clean label list (NO_RELATION always id=0)
    current_labels = set(df["label"])
    if "NO_RELATION" in current_labels:
        current_labels.remove("NO_RELATION")
    
    relation_types = ["NO_RELATION"] + sorted(list(current_labels))
    logger.info(f"Final relation types ({len(relation_types)}): {relation_types}")

    # (4) create dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("label", ClassLabel(names=relation_types))

    # (5) split and save dataset
    # downsamples train split negative samples
    final_dataset, id2label, label2id = split_and_save(
        dataset, 
        relation_types, 
        args.output_dir, 
        train_downsample_rate=args.train_downsample_rate, 
        seed=args.seed
    )
    
    # (6) report dataset statistics
    print_label_distribution(final_dataset, relation_types)
    logger.info("Pipeline completed successfully.")
    

# boilerplate
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and preprocess chia dataset for relation extraction.")
    
    # get system paths
    # and then define default paths
    # __file__ = src/re_parseChia.py
    # .parent  = src/
    # .parent.parent = project_root/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_DATA = PROJECT_ROOT / "data" / "chia_without_scope"
    DEFAULT_OUT = PROJECT_ROOT / "data" / "chia_without_scope_parsedRE_v1"

    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA),
                        help="Path to directory containing raw .txt and .ann files.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT),
                        help="Directory to save the processed chia dataset.")
    
    # configuration
    parser.add_argument("--train_downsample_rate", type=float, default=0.2,
                        help="Ratio of negative samples (NO_RELATION) to keep in the training split. "
                             "Default 0.2 means keep 20%% of negatives. 1.0 or None = keep all.")
    parser.add_argument("--global_downsample_rate", type=float, default=None,
                        help="Ratio of negative samples (NO_RELATION) to keep across the entire dataset during parsing (affects train, val, and test)."
                            "Default None (keep all). 0.2=keep 20%%, 1.0=keep all.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    
    # check if interactive environment (e.g. jupyter notebook)
    # or command line
    if "ipykernel" in sys.modules:
        # notebook mode
        args = parser.parse_args([])
    else:
        # command line mode
        args = parser.parse_args()
    
    main(args)