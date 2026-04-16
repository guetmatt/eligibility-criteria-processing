"""
NER Training Module

This script handles the training and hyperparameter optimization
of a BERT-based Named Entity Recognition (NER) model using the HuggingFace Transformers library.

Usage (argument values to be adjusted by user):
    python ner_training.py --data_dir ../data/chia_without_scope_parsedNER_v1 --output_dir ../models/NER_chia_v1
"""


# --- imports --- #
import os
import json
import logging
import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import evaluate
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizer
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers.trainer_utils import EvalPrediction
import optuna



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
        id2label, label2id (tuple[dict[str, int], dict[int, str]]): A tuple containing label mappings (label2id, id2label).
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
    
    return label2id, id2label



def compute_metrics_seq(p: EvalPrediction, id2label: dict[int, str]):
    """
    Computes precision, recall, F1, and accuracy using the seqeval library
    Sequence-based evalaution of named entity spans (not single tokens)

    Args:
        p (EvalPrediction): Predictions and labels (tuple) from the Trainer
        id2label (dict[int, str]): Mapping from label_id to label_name

    Returns:
        results_dict (dict[str, float]: Dictionary containing overall precision, recall, f1, and accuracy.
    """
    metric = evaluate.load("seqeval")
    
    # predictions -> logits, model prodictions
    # true_labels -> ground truth labels
    predictions, true_labels = p.predictions, p.label_ids
    
    # convert logits to predicted label_ids
    predictions = np.argmax(predictions, axis=2)

    # convert label_ids to label_names
    # and filter out special tokens (-100)
    predictions_named = list()
    true_labels_named = list()
    for prediction, label in zip(predictions, true_labels):
        sentence_predictions = list()
        sentence_true_labels = list()
        for p, l in zip(prediction, label):
            if l != -100:
                sentence_predictions.append(id2label[p])
                sentence_true_labels.append(id2label[l])
        predictions_named.append(sentence_predictions)
        true_labels_named.append(sentence_true_labels)
    
    # calculate metrics
    # zero_division=0 handles cases with no predicted entities
    results = metric.compute(
        predictions=predictions_named, 
        references=true_labels_named, 
        zero_division=0
    )
    # metrics overall
    results_dict = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    } 
    
    # metrtics per entity type
    for key, val in results.items():
        if isinstance(val, dict):
            results_dict[f"entity_{key}_f1"] = val["f1"]
            results_dict[f"entity_{key}_precision"] = val["precision"]
            results_dict[f"entity_{key}_recall"] = val["recall"]
    
    return results_dict



def compute_metrics_tok(p: EvalPrediction, id2label: dict[int, str]):
    """
    Computes precision, recall, F1, and accuracy using the scikit-learn library.
    Token-based evalaution of named entity tokens (not spans).

    Args:
        p (EvalPrediction): Predictions and labels (tuple) from the Trainer.
        id2label (dict[int, str]): Mapping from label_id to label_name.

    Returns:
        results_dict (dict[str, float]: Dictionary containing overall precision, recall, f1, and accuracy.
    """
    # predictions -> logits, model prodictions
    # true_labels -> ground truth labels
    predictions, true_labels = p.predictions, p.label_ids
    
    # convert logits to predicted label_ids
    predictions = np.argmax(predictions, axis=2)
    
    # flatten batches, filter special tokens (-100)
    predictions_flat = list()
    true_labels_flat = list()
    for prediction, label in zip(predictions, true_labels):
        for p, l in zip(prediction, label):
            if l != -100:
                predictions_flat.append(p)
                true_labels_flat.append(l)
    
    # merge B- and I- labels into one entity label
    def get_base_name(label_id):
        name = id2label[label_id]
        if name.startswith("B-") or name.startswith("I-"):
            return name[2:]
        return name 
    # convert predictions and true labels to base entity names
    predictions_flat_merged = [get_base_name(p) for p in predictions_flat]
    true_labels_flat_merged = [get_base_name(l) for l in true_labels_flat]
    
    # calculate metrics
    # average='weighted' accounts for label imbalance (e.g. many 'O' labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels_flat_merged, predictions_flat_merged, average="weighted", zero_division=0)
    accuracy = accuracy_score(true_labels_flat_merged, predictions_flat_merged)
    
    results_dict = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }
    
    # metrics per entity type
    labels_set = sorted(list(set(true_labels_flat_merged)))
    precision_per_ent, recall_per_ent, f1_per_ent, _ = precision_recall_fscore_support(
        true_labels_flat_merged, predictions_flat_merged, average=None, labels=labels_set, zero_division=0
    )
    for idx, label_name in enumerate(labels_set):
        results_dict[f"entity_{label_name}_f1"] = f1_per_ent[idx]
        results_dict[f"entity_{label_name}_precision"] = precision_per_ent[idx]
        results_dict[f"entity_{label_name}_recall"] = recall_per_ent[idx]
    
    return results_dict



def model_init(trial: optuna.Trial, model_checkpoint: str, label2id: dict[str, int], id2label: dict[int, str]):
    """
    Initializes a model for for hyperparameter optimization.

    Args:
        trial (optuna.Trial): The optuna trial object from hyperparamter optimization.
        model_checkpoint (str): HuggingFace model.
        label2id (Dict): Mapping of label names to label_ids.
        id2label (Dict): Mapping of label_ids to label names.

    Returns:
        AutoModelForTokenClassification: The re-initialized model.
    """
    
    return AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )



def hyperparameter_space(trial: optuna.Trial):
    """
    Defines the hyperparameter search space for optuna.
    Sticking to standard narrow ranges for NER.
    
    Args:
        trial (optuna.Trial): The current Optuna trial object

    Returns:
        A dictionary of hyperparameter suggestions.
    """
    
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.3),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 4, 8)
    }



def check_overwrite(output_dir: str):
    """
    Checks if the output directory already contains a trained model 
    and asks the user for confirmation before overwriting.
    
    Args:
        output_dir (str): Output directory to be checked.
    
    Returns:
        None
    """
    if os.path.exists(output_dir):
        # check for typical transformer model files
        model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        # "any" expects iterable input
        # --> list comprehension
        if any([os.path.exists(os.path.join(output_dir, f)) for f in model_files]):
            # ask user if existing model should be overwritten
            print(f"\n[WARNING] The output directory '{output_dir}' already contains a trained model.")
            user_input = input("Do you really want to overwrite the existing model? (y/n): ").lower().strip()
            if user_input != "y":
                print("Aborting script to prevent overwriting.")
                sys.exit(0)
    return None



def main(args):
    """
    Main training pipeline execution.
    Loads parsed dataset for NER, initializes tokenizer, data collator and model,
    performs hyperparameter optimization, trains an NER model, 
    evaluates trained model, saves trained model and tokenizer.
    Arguments provided via command line.

    Args:
        data_dir (str): Path to the preprocessed chia dataset directory, default="./data/chia_without_scope_parsedNER_v1"
        output_dir (str): Directory to save the trained model and tokenizer, default="./models/NER_chia_v1"
        model_checkpoint (str): HuggingFace model checkpoint of the model that will be trained for NER, default="emilyalsentzer/Bio_ClinicalBERT"
        eval_method (str): Evaluation method for model training/testing: 'seq' for span-based or 'tok' for token-based
        do_hpo (bool): Whether to run hyperparameter optimization or not, default=True
        hpo_trials (int): Number of hyperparameter optimization trials to run, default=10
    """
    # (0) check for existing model
    check_overwrite(args.output_dir)
    
    
    logger.info(f"Starting training with data from: {args.data_dir}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    
    # (1) load dataset
    try:
        dataset = load_from_disk(args.data_dir)
        logger.info(f"Dataset loaded successfully. Splits:{list(dataset.keys())}")
    except FileNotFoundError:
        logger.critical(f"Dataset not found at {args.data_dir}. Check the path.")
        sys.exit(1)

    # (2) load label mappings
    label2id, id2label = load_label_map(args.data_dir)
    logger.info(f"Loaded {len(label2id)} labels.")
    
    # (3) load tokenizer and data collator (for padding)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # (4) training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        fp16=True,
        disable_tqdm=False,
        report_to="none"
    )
    
    # (5) initialize trainer
    # functools.partial --> to inject arguments into model_init and compute_metrics
    init_func = partial(
        model_init, 
        model_checkpoint=args.model_checkpoint, 
        label2id=label2id, 
        id2label=id2label
    )
    
    # choose evaluation methods based on input
    if args.eval_method == "tok":
        logger.info("Using Token-based evaluation.")
        metrics_func = partial(compute_metrics_tok, id2label=id2label)
    else:
        logger.info("Using Sequence-based evaluation.")
        metrics_func = partial(compute_metrics_seq, id2label=id2label)
    
    trainer = Trainer(
        model_init=init_func,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_func
    )
    
    # (6) hyperparameter optimization
    if args.do_hpo:
        logger.info("Starting hyperparameter optimization...")
        best_run = trainer.hyperparameter_search(
            direction="maximize",
            hp_space=hyperparameter_space,
            backend="optuna",
            n_trials=args.hpo_trials,
            study_name="ner_hpo_search"
            )
        logger.info(f"Best hyperparameters: {best_run}")
    
    # update trainer with best hyperparameters
    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)
    
    # (7) start training
    logger.info("Starting model training...")
    trainer.train()
    
    # (8) final evaluation
    logger.info("Evaluating the trained model on the test set...")
    metrics = trainer.evaluate(dataset["test"])
    # metrics overall and per entity type
    metrics_overall = {k: v for k, v in metrics.items() if not k.startswith("eval_entity_")}
    metrics_entity = {k: v for k, v in metrics.items() if k.startswith("eval_entity_")}   
    
    # print metrics
    print("\n" + "="*30)
    print("FINAL TEST METRICS - OVERALL")
    print("="*30)
    for key, value in metrics_overall.items():
        print(f"{key}: {value:.4f}")
    print("="*30 + "\n")
    
    print("\n" + "="*30)
    print("FINAL TEST METRICS - PER ENTITY TYPE")
    print("="*30)
    for key, value in metrics_entity.items():
        print(f"{key}: {value:.4f}")
    print("="*30 + "\n")
    
    
    # (9) save trained model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Trained model and tokenizer saved to {args.output_dir}")
    

# boilerplate
if __name__ == "__main__":
    # argparse for command line arguments
    parser = argparse.ArgumentParser(description="Train an NER model with HPO support.")
    
    # get system paths
    # and then define default paths
    # __file__ = src/ner_parseChia.py
    # .parent  = src/
    # .parent.parent = project_root/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "chia_without_scope_parsedNER_v1"
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "NER_chia_v1"
    
    
    # required paths
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to the preprocessed chia dataset directory.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save the trained model and tokenizer.")
    
    # model configuration
    parser.add_argument("--model_checkpoint", type=str, default="emilyalsentzer/Bio_ClinicalBERT",
                        help="HuggingFace model checkpoint of the model that will be trained for NER.")
    
    # evaluation method
    parser.add_argument("--eval_method", type=str, default="seq", choices=["seq", "tok"],
                        help="Evaluation method for model training/testing: 'seq' for span-based or 'tok' for token-based.")
    
    # hyperparameter optimization configuration
    parser.add_argument("--do_hpo", action="store_true", default=True,
                        help="Whether to run hyperparameter optimization or not.")
    parser.add_argument("--hpo_trials", type=int, default=10,
                        help="Number of hyperparameter optimization trials to run.")

    # check if interactive environment (e.g. jupyter notebook)
    # or command line
    if "ipykernel" in sys.modules and len(sys.argv) < 2:
        # notebook mode
        # falls back to default arguments, if < 2 arguments provided
        args = parser.parse_args([])
    else:
        # command line mode
        args = parser.parse_args()
    
    main(args)