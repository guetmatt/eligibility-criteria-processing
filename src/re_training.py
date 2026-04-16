"""
RE Training Module

This script handles the training and hyperparameter optimization
of a BERT-based Relation Extraction (RE) model using the HuggingFace Transformers library.
- injection of entity markers [E1]...[/E1] to identify relation arguments
- tokenization with added entity marker tokens
- weighted loss function for class imbalance (NO_RELATION "Explosion")

Usage (argument values to be adjusted by user):
    python re_training.py --data_dir ../data/chia_without_scope_parsedRE_all_v1 --output_dir ../models/RE_model_v1
"""

# --- imports --- #
import os
import json
import logging
import argparse
import sys
from pathlib import Path
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import CrossEntropyLoss
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
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
    Loads label mappings from the dataset directory.

    Args:
        input_dir (str): Path to directory containing the label mapping file
        filename (str): Name of the laebl mapping JSON file

    Returns:
        tuple[dict, dict]:
        - label2id: mapping from label names to label_id
        - id2label: mapping from label_id to label names
    """
    # system path
    path = os.path.join(input_dir, filename)
    if not os.path.exists(path):
        logger.error(f"Label mapping file not found at {path}")
        raise FileNotFoundError(path)

    # open and read file        
    with open(path, "r", encoding="UTF-8") as f:
        data = json.load(f)
    
    # JSON saves keys as strings
    # --> convert to int for id2label
    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}
    
    return label2id, id2label



def format_re_input(example: dict):
    """
    Injects entity markers [E1], [/E1], [E2], [/E2] into the text.
    Markers are inserted right-to-left based on character indices to prevent 
    index shifting.
    Markers are used to identify arguments for a relation.

    Args:
        example (dict): A single dataset sample containing entity indices

    Returns:
        dict: Updated sample dictionary with injected entity markers ("text_with_markers")
    """
    text = example["text"]
    
    # collect indices for injection
    insertions = [
        (example['e1_start'], "[E1]"), 
        (example['e1_end'],   "[/E1]"),
        (example['e2_start'], "[E2]"), 
        (example['e2_end'],   "[/E2]")
    ]
    # sort indices descending 
    # --> for right-to-left injection
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    # entity marker injection
    marked_text = text
    for pos, marker in insertions:
        marked_text = marked_text[:pos] + marker + marked_text[pos:]
    
    return {"text_with_markers": marked_text}



def compute_metrics(eval_pred):
    """
    Calculates classification performance metrics precision, recall, f1 and accuracy.
    Uses "weighted" average to account for class imbalance (NO_RELATION "explosion")
    
    Args:
        eval_pred (tuple): Tuple containing "predictions" (logits) and "label_ids", model relation predictions

    Returns:
        results (dict): Dictionary with performance metrics
    """
    # get model predictions and ground_truth
    logits, true_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(true_labels, predictions)
    
    results = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    } 
    
    return results



def compute_weighted_loss(
    outputs,
    labels,
    num_items_in_batch=None, 
    class_weights: torch.Tensor = None
    ):
    """
    Custom weighted loss function to handle class imbalance.
    Cross-Entropy loss weighted by class frequency.

    Args:
        outputs: SequenceClassififerOutput containing prediction logits
        labels: Tensor containing ground truth
        num_items_in_batch (int): Argument required by newer Trainer versions (unused here)
        class_weights (torch.Tensor): A tensor of weights for each class, computed on the training split.

    Returns:
        loss (torch.Tensor): The weighted loss calculated by the weighted loss function
    """
    # extract logits from model output
    logits = outputs.get("logits")

    # move class_weights to correct device
    # needed for library use
    if class_weights is not None:
        class_weights = class_weights.to(logits.device)
    
    # define and compute weighted loss
    loss_func = CrossEntropyLoss(weight=class_weights)
    loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
    
    return loss



def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, id2label: dict, output_dir: str):
    """
    Generates and saves a normalized confusion matrix heatmap to visualize model errors.
    Shows the proportion of instances of a class that was classified into each 
    available category.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        id2label (dict): Label mappings to from label_ids to label names
        output_dir (str): Path to save the resulting PNG image
    
    Returns:
        None
    """
    
    # sort labels by id
    # --> to match axis alignment with matrix indices
    label_ids = sorted(id2label.keys())
    label_names = [id2label[idx] for idx in label_ids]

    # compute matrix
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids, normalize="true")
    
    # plot matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=label_names, 
                yticklabels=label_names, cmap="Blues", cbar=True)
    
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.title("Normalized Confusion Matrix for Relation Extraction", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    
    # save plot-png to disk
    save_path = os.path.join(output_dir, "confusion_matrix_re.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return None



# --- model and hyperparameter optimization setup --- #

def model_init(trial: optuna.Trial,
               model_checkpoint: str,
               label2id: dict,
               id2label: dict,
               tokenizer_len: int):
    """
    Initializes a fresh model instance for training and hyperparameter optimization.
    The tokenizer vocabulary gets extended by entity marker tokens previously,
    so that the models embedding layer has to be resized to accommodate
    the extended vocabulary.

    Args:
        trial (optuna.Trial): The current optuna trial object (from a hyperparameter optimization study)
        model_checkpoint (str): The HuggingFace path to the model
        label2id (dict): Label mapping from label names to label_ids
        id2label (dict): Label mapping from label_ids to label names
        tokenizer_len (int): The length of the tokenizer vocabulary (including entity markers)

    Returns:
        model (AutoModelForSequenceClassification): An initialized AutoModelForSequenceClassification from HuggingFace
    """
    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # resize model embeddings
    # to adjust for entity marker tokens
    model.resize_token_embeddings(tokenizer_len)
    
    return model



def hyperparameter_space(trial: optuna.Trial):
    """
    Defines the search space for hyperparameter optimization.

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
        if any([os.path.exists(os.path.join(output_dir, f)) for f in model_files]):
            # ask user if existing model should be overwritten
            print(f"\n[WARNING] The output directory '{output_dir}' already contains a trained model.")
            user_input = input("Do you really want to overwrite the existing model? (y/n): ").lower().strip()
            if user_input != "y":
                print("Aborting script to prevent overwriting.")
                sys.exit(0)
    return None

    

# --- main execution ---#
def main(args):
    """
    Main training pipeline execution.
    Loads parsed dataset for RE, injects entity markers,
    adds entity marker tokens to tokenizer vocabulary,
    performs tokenization, calculates weighted loss function,
    performs hyperparameter optimization, trains an RE model,
    evaluates trained model, saves trained model and tokenizer.
    Arguments provided via command line.

    Args:
        - data_dir (str): Path to the preprocessed chia dataset directory, default="./data/chia_without_scope_parsedRE_v1"
        - output_dir (str): Directory to save the trained model and tokenizer, default="./models/RE_chia_v1"
        - model_checkpoint (str): HuggingFace model checkpoint of the model that will be trained for RE, default="emilyalsentzer/Bio_ClinicalBERT"
        - max_len (int): Maximum sequence length after token injection, default=256
        - do_hpo (bool): Whether to run hyperparameter optimization or not, default=True
        - hpo_trials (int): Number of hyperparameter optimization trials to run, default=10
    """
    # (0) check for existing model
    check_overwrite(args.output_dir)
    
    # (1) load dataset
    logger.info(f"Loading dataset from {args.data_dir}...")
    dataset = load_from_disk(args.data_dir)
    # rename "label" to "labels" for Trainer functionality
    dataset = dataset.rename_column("label", "labels")
    label2id, id2label = load_label_map(args.data_dir)

    # (2) entity marker injection
    logger.info("Injecting entity markers...")
    dataset = dataset.map(format_re_input)

    # (3) tokenization and vocabulary extension
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)

    def tokenize_func(batch):
        return tokenizer(batch["text_with_markers"], truncation=True, 
                         padding="max_length", max_length=args.max_len)

    tokenized_dataset = dataset.map(tokenize_func, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # (4) class weight calculation to handle class imbalance
    train_labels = dataset["train"]["labels"]
    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    
    # dampening of class weighting to reduce "aggressiveness" of weighting
    # --> uses square-root for dampening
    # --> to prevent model from unstable/radical gradient updates
    weights = np.sqrt(weights)
    weights /= np.mean(weights)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # (4) Trainer initialization
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        report_to="none"
    )
    
    # functools.partial
    # --> to inject arguments into model_init and compute_metrics
    init_func = partial(
        model_init, 
        model_checkpoint=args.model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        tokenizer_len=len(tokenizer)
    )

    # initialized weighted loss function
    weighted_loss_func = partial(compute_weighted_loss, class_weights=class_weights_tensor)

    trainer = Trainer(
        model_init=init_func,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        compute_loss_func=weighted_loss_func
    )

    # (6) hyperparameter optimization
    if args.do_hpo:
        logger.info("Starting hyperparameter optimization...")
        best_run = trainer.hyperparameter_search(
            direction="maximize",
            hp_space=hyperparameter_space,
            n_trials=args.hpo_trials
        )
        logger.info(f"Best hyperparameters: {best_run}")

        # apply optimal hyperparameters to trainer configuration
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)

    # (7) model training
    logger.info("Starting training...")
    trainer.train()

    # (8) evaluation and metric reporting
    logger.info("Final evaluation on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    
    # classification report
    preds_output = trainer.predict(tokenized_dataset["test"])
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_true = tokenized_dataset["test"]["labels"]

   
    # extract all ids and names from label mappings
    # --> to pass to classification_report for full functionality
    all_label_ids = sorted(id2label.keys())
    all_label_names = [id2label[i] for i in all_label_ids]

    print("\n" + "="*50)
    print("DETAILED TEST SET CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true,
                                y_pred,
                                labels=all_label_ids,
                                target_names=all_label_names,
                                zero_division=0
                                ))
    
    # create confusion matrix
    plot_confusion_matrix(y_true, y_pred, id2label, args.output_dir)

    # (9) save trained model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model and tokenizer successfully saved to {args.output_dir}")
    


# boilerplate
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RE model with HPO support.")
    
    # get system paths
    # and then define default paths
    # __file__ = src/ner_parseChia.py
    # .parent  = src/
    # .parent.parent = project_root/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "chia_without_scope_parsedRE_v1"
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "RE_chia_v1"

    # required paths
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Path to the preprocessed chia dataset directory.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to save the trained model and tokenizer.")
    
    # model configuration
    parser.add_argument("--model_checkpoint", type=str, default="emilyalsentzer/Bio_ClinicalBERT",
                        help="HuggingFace model checkpoint of the model that will be trained for RE.")
    parser.add_argument("--max_len", type=int, default=256,
                        help="Maximum sequence length after token injection.")
    
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