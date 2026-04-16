"""
End-to-End Inference Pipeline (NER + RE)

Full inference workflow for clinical trial eligibility criteria.
Sequentially applies an NER-model to identify clinical entities
and an RE-model to determine semantic relations between these entities.

The pipeline architecture:
    (1) NER: identifying clinical entities (Condition, Drug, Measurement, etc.).
    (2) Candidate pairing: generating all valid permutations of identified entities
    (3) RE: identifying relations between candidate pairs

The results are exported in JSONL and CSV formats.

Usage (argument values to be adjusted by user):
    python pipeline_inference.py
        --data_dir ../data_ctg/parsedCTG_sentlevel.csv
        --output_dir ../results/v1
        --ner_model_path ../models/ner_chia_v1
        --re_model_path ../models/re_chia_v1
"""

# --- imports --- #
import os
import json
import torch
import logging
import argparse
import sys
import itertools
from pathlib import Path
import ast

import pandas as pd
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer
)


# --- logging setup --- #
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)



# --- functions --- #

def run_ner_module(text: str, ner_pipe):
    """
    Applies the NER model to an eligibility criterion sentence
    to identify clinical entities.

    Args:
        text (str): The raw eligibility criterion sentence string
        ner_pipe: HuggingFace token-classification pipeline

    Returns:
        predicted_entities (list): List of predicted entitity dictionaries
        with keys "entity_group", "word", "start", "end", "confidence_score"
    """
    # apply NER
    predictions = ner_pipe(text)
    
    # format predictions
    # for JSONL serialization
    predicted_entities = list()
    for pred in predictions:
        clean_pred = {
            "entity_group": pred["entity_group"],
            "word": pred["word"],
            "start": int(pred["start"]),
            "end": int(pred["end"]),
            "confidence_score": float(pred["score"])
        }
        predicted_entities.append(clean_pred)
        
    return predicted_entities



def format_re_input(text: str, entity1: dict, entity2: dict):
    """
    Formats a sentence for RE inference by injecting entity markers.
    Performs right-to-left injeciton to ensure that index matching.

    Args:
        text (str): Sentence string
        entity1 (dict): First argument/entity of the relation
        entity2 (dict): Second argument/entity of the relation

    Returns:
        marked_text (str): Sentence string with injected entity markers
    """
    # entity/argument indices
    e1_start, e1_end = entity1["start"], entity1["end"]
    e2_start, e2_end = entity2["start"], entity2["end"]
    insertions = [
        (e1_start, "[E1]"), (e1_end, "[/E1]"),
        (e2_start, "[E2]"), (e2_end, "[/E2]")
    ]
    # sort descending by position
    # --> to insert markers right-to-left    
    insertions.sort(key=lambda x: x[0], reverse=True)

    # entity marker injecton
    marked_text = text
    for pos, marker in insertions:
        # safety check for string boundaries
        if pos > len(marked_text):
            continue
        marked_text = marked_text[:pos] + marker + marked_text[pos:]
    
    return marked_text



def run_re_module(
    text: str, 
    entities: list, 
    re_tokenizer: PreTrainedTokenizer, 
    re_model: PreTrainedModel,
    device: torch.device
    ):
    """
    Applies an RE model to a text and a pair of entities
    to identify relations between entities.
    Performs entity marker injection, generates all directed permutations
    of entity paris, filters out overlapping entities.

    Args:
        text (str): Sentence string
        entities (list): List of entity dictionaries identified by the NER module
        re_tokenizer (PreTrainedTokenizer): Tokenizer with added entity marker tokens.
        re_model (PreTrainedModel): Trained RE model
        device (torch.deevice): Torch device (CPU/GPU) for inference

    Returns:
        found_relations (list): List of predicted relations, excluding 'NO_RELATION'
    """
    if len(entities) < 2:
        return []

    # generate all directed permutations
    # from all entities in sentence
    # (A->B and B->A)
    pairs = list(itertools.permutations(entities, 2))
    found_relations = list()
    
    # set model in eval-mode
    # to only keep model-layers relevant for classification
    re_model.eval()
    
    for e1, e2 in pairs:
        # filter overlapping spans
        # --> to avoid ambiguous marker injection
        if max(e1["start"], e2["start"]) < min(e1["end"], e2["end"]):
            continue
        
        # entity marker injection
        try:
            marked_text = format_re_input(text, e1, e2)
        except Exception as e:
            logger.warning(f"Formatting error for pair {e1['word']}/{e2['word']}: {e}")
            continue
        
        # tokenization and move device for inference
        inputs = re_tokenizer(
            marked_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        ).to(device)
        
        # relation prediction
        with torch.no_grad():
            outputs = re_model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
        pred_label = re_model.config.id2label[pred_id]
        
        # collect predicted relations
        # only keep positive relations
        if pred_label != "NO_RELATION":
            confidence = torch.softmax(logits, dim=1)[0][pred_id].item()
            found_relations.append({
                "arg1": e1["word"],
                "arg1_type": e1["entity_group"],
                "arg2": e2["word"],
                "arg2_type": e2["entity_group"],
                "relation": pred_label,
                "confidence_score": round(confidence, 4),
                "marked_context": marked_text
            })
    
    return found_relations



def save_predictions(data: list, output_dir: str, filename: str):
    """
    Saves inference results in JSONL and CSV formats.

    Args:
        data (list): List of processed sentence dictionaries
        output_dir (str): Directory to save results
        filename (str): Filename for predicitions files, generates .jsonl- and .csv-endings (e.g. "NER_predictions")
    Returns:
        None
    """
    
    jsonl_path = os.path.join(output_dir, f"{filename}.jsonl")
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    
    # save as JSONL
    with open(jsonl_path, "w", encoding="UTF-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    # save as CSV
    pd.DataFrame(data).to_csv(csv_path, index=False)
    logger.info(f"Saved {filename} results to {output_dir}")
    
    return None



# --- main execution --- #
def main(args):
    """
    Executes the complete NER + RE pipeline.
    Arguments provided via command line.
    
    Args:
        data_dir (str): Path to the preprocessed ctg dataset/data to apply pipeline to, default="../data_ctg/parsedCTG_sentlevel"
        output_dir(str): Directory to save the predictions, default="../results"
        ner_model_path (str): Path to NER model
        re_model_path (str): Path to RE model
        mode (str): Inference mode, whether to apply NER + RE inference, NER only, or RE only. ["ner+re", "ner", "re"], default="ner+re"
        sample_size (optional, int): Limit number of sentences for testing, not applied when not defined, default=None
    """
    # (1) environment setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # (2) load data to perform inference on
    logger.info(f"Inference input data: {args.data_dir}")
    df = pd.read_csv(args.data_dir)
    # sample_size to limit test samples for development
    if args.sample_size:
        df = df.head(args.sample_size)
    
    # (3) load models
    logger.info("Initializing models...")
    # NER model and tokenizer
    # device --> 0=cpu, 1=gpu
    if args.mode == "ner+re" or args.mode=="ner":
        ner_pipeline = pipeline(
            "token-classification",
            model=args.ner_model_path,
            tokenizer=args.ner_model_path,
            aggregation_strategy="simple",
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
    
    # RE model and tokenizer
    if args.mode == "ner+re" or args.mode=="re":
        re_tokenizer = AutoTokenizer.from_pretrained(args.re_model_path)
        re_model = AutoModelForSequenceClassification.from_pretrained(args.re_model_path).to(device)

    # (4) Named Entity Recognition
    if args.mode == "ner+re" or args.mode=="ner":
        logger.info("Running NER Module...")
        ner_results = list()
        # extract entities for each sentence
        for idx, row in df.iterrows():
            text = str(row["text"])
            entities = run_ner_module(text, ner_pipeline)
        
            ner_results.append({
                "sentence_id": idx,
                "nct_id": row.get("studyNCTid", "unknown"),
                "criteria_type": row.get("criteria_type", "unknown"),
                "text": text,
                "entities": entities
            })
    
        # save predicted entities to disk
        save_predictions(ner_results, args.output_dir, "ner_predictions")

    # (5) Relation Extraction
    
    # relation extraction with previous NER
    if args.mode == "ner+re":
        logger.info("Running RE Module...")
        full_pipeline_results = list()
        # identify relation between entities from NER module
        for entry in ner_results:
            text = entry["text"]
            entities = entry["entities"]
        
            relations = run_re_module(text, entities, re_tokenizer, re_model, device)
        
            # combine everything for final output
            full_pipeline_results.append({
                **entry,
                "relations": relations
            })
    
        # save predicted relations to disk
        save_predictions(full_pipeline_results, args.output_dir, "re_predictions")
    
    # relation extraction without NER
    # input data must contain "entities"-column
    elif args.mode == "re":
        logger.info("Running RE Module...")
        full_pipeline_results = list()
        # load input data
        if args.data_dir.endswith(".jsonl"):
            df = pd.read_json(args.data_dir, lines=True)
        else:
            df = pd.read_csv(args.data_dir)
        
        re_results = list()
        for idx, row in df.iterrows():
            # get text and entities from input data
            text = str(row["text"])
            entities = row.get("entities", [])
            # convert input to list
            # as entities-list from input is loaded as str
            entities = ast.literal_eval(entities)
            # identify relations
            relations = run_re_module(text, entities, re_tokenizer, re_model, device)
            
            re_results.append({
                "sentence_id": idx,
                "nct_id": row.get("studyNCTid", "unknown"),
                "criteria_type": row.get("criteria_type", "unknown"),
                "text": text,
                "entities": entities,
                "relations": relations
                })

        save_predictions(re_results, args.output_dir, "re_predictions")
    
    logger.info("Pipeline inference completed successfully.")



# boilerplate
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete NER + RE Pipeline.")
    
    # get system paths
    # and then define default paths
    # __file__ = src/pipeline_inference.py
    # .parent  = src/
    # .parent.parent = project_root/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_DATA_DIR = PROJECT_ROOT / "data_ctg" / "parsedCTG_sentlevel.csv"
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"

    # data paths
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Path to the preprocessed ctg dataset/data to apply pipeline to.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to save the predictions. Automatically adds 'ner_predictions' and 're_predictions' as file names.")
    
    # model paths
    parser.add_argument("--ner_model_path", type=str, help="Path to NER model.")
    parser.add_argument("--re_model_path", type=str, help="Path to RE model.")
    
    # inference mode
    parser.add_argument("--mode", type=str, default="ner+re", choices=["ner+re", "ner", "re"],
                        help="Inference mode, whether to apply NER + RE inference, NER only, or RE only.")
    
    # sample size for testing purposes
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Limit number of sentences for testing. Not applied when not defined.")

    # check if interactive environment (e.g. jupyter notebook)
    # or command line
    if "ipykernel" in sys.modules:
        # notebook mode
        args = parser.parse_args([
            "--ner_model_path", "../models/ner_chia_v1",
            "--re_model_path", "../models/re_chia_v1"
            ])
    else:
        # command line mode
        args = parser.parse_args()

    main(args)