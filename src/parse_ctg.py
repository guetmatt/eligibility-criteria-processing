"""
Data Parsing Module for raw data from ClinicalTrials.gov

Provides functionality to import csv-files downloaded from ClinicalTrials.gov
and to transform the "EligibilityCriteria" text blocks
into a structured, sentence-level format. 
- identifies criteria type (inclusion, exclusion)
- parses list items within eligibility criteria content
- exctracts metadata (NCT ids)

Usage (argument values to be adjusted by user):
    python parse_ctg.py --input_path ../data_ctg/Studies_with_id_and_EligibilityCriteria.csv --output_path ../data_ctg/parsedCTG_sentlevel.csv
"""


# --- imports --- #
# --- imports --- #
import os
import re
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd


# --- logging setup --- #
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)



# --- functions --- #
def parse_criteria_text(text: str):
    """
    Parses a text block of eligibility criteria into a list of sentences
    and detects criteria type (inclusion, exclusion) for each sentence.
    Accepted bullet point markers are "*" and "-".

    Args:
        text (str): Eligibility criteria text content

    Returns:
        parsed_items (list): A list of dictionaries, each dictionary represents a single bullet point (criterion), with keys:
            - "criteria_type": "inclusion" or "exclusion"
            - "text": Text content of the criterion

    """
    # no text provided
    if not isinstance(text, str):
        return []

    # split text into lines
    lines = text.split("\n")
    parsed_items = []
    criteria_type = None
    
    # regex patterns
    # - section headers (criteria type)
    inclusion_pattern = re.compile(r"^\s*inclusion\s+criteria:?", re.IGNORECASE)
    exclusion_pattern = re.compile(r'^\s*exclusion\s+criteria:?', re.IGNORECASE)
    # - bulleted list content
    # --> text following bullet markers (* or -)
    bullet_pattern = re.compile(r'^\s*[*-]\s+(.*)')

    for line in lines:
        line_stripped = line.strip()
        
        # ignore empty lines
        if not line_stripped:
            continue
        
        # criteria type detection
        if inclusion_pattern.search(line_stripped):
            criteria_type = "inclusion"
            continue
        elif exclusion_pattern.search(line_stripped):
            criteria_type = "exclusion"
            continue
        
        # bullet point content detection
        # only when criteria_type has been found
        # --> ensures correct format
        if criteria_type:
            match = bullet_pattern.match(line_stripped)
            if match:
                content = match.group(1).strip()
                if content:
                    parsed_items.append({
                        "criteria_type": criteria_type,
                        "text": content
                    })
    
    return parsed_items



def process_dataset(data_dir: str, output_dir: str):
    """
    Loads and parses the ClinicalTrials.gov dataset.
    - row-wise iteration over the input CSV
    - parses eligibility criteria for each clinical trial 

    Args:
        data_dir (str): Path to the source CSV file
        output_dir (str): Directory where the processed sentence-level CSV will be saved

    Returns:
        df_processed (pd.DataFrame): DataFrame with processed sentence-level data
    """
    # load data
    logger.info(f"Loading raw dataset from {data_dir}...")
    if not os.path.exists(data_dir):
        logger.error(f"Input file not found: {data_dir}")
        raise FileNotFoundError(f"Missing input CSV at {data_dir}")

    try:
        df = pd.read_csv(data_dir)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise
    
    # process data
    processed_data = []
    logger.info("Starting criteria text parsing...")

    for idx, row in df.iterrows():
        # ignore malformed csv lines
        if "EligibilityCriteria" not in row or "StudyNCTid" not in row:
            continue

        criteria_list = parse_criteria_text(row["EligibilityCriteria"])
        
        for item in criteria_list:
            processed_data.append({
                "studyNCTid": row["StudyNCTid"],
                "criteria_type": item["criteria_type"],
                "text": item["text"]
            })

    # convert processed data to DataFrame
    df_processed = pd.DataFrame(processed_data)
    
    # save processed data to disk
    output_dir_full = os.path.dirname(output_dir)
    if output_dir_full:
        os.makedirs(output_dir_full, exist_ok=True)
    logger.info(f"Saving {len(df_processed)} parsed sentences to {output_dir}...")
    df_processed.to_csv(output_dir, index=False)
    
    return df_processed



# --- main execution --- #
def main(args):
    """
    Main parsing pipeline execution. Loads raw data downloaded from
    ClinicalTrials.gov and parses it into a sentence-level csv-file.
    Arguments provided via command line.
    
    Args:
        data_dir (str): Path to the raw ClinicalTrials.gov csv, default="../data_ctg/Studies_with_id_and_EligibilityCriteria.csv",
        output_dir (str): Path to save the structured sentencelevel csv, default="../data_ctg/parsedCTG_sentlevel.csv"
    """
    # load and process raw data
    try:
        process_dataset(args.data_dir, args.output_dir)
        logger.info("CTG Parsing pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")
        sys.exit(1)
        


# boilerpalte
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ClinicalTrials.gov eligibility criteria blocks into sentence-level data.")
    
    # get system paths
    # and then define default paths
    # __file__ = src/ner_parseChia.py
    # .parent  = src/
    # .parent.parent = project_root/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_DATA_DIR = PROJECT_ROOT / "data_ctg" / "Studies_with_id_and_EligibilityCriteria.csv"
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_ctg" / "parsedCTG_sentlevel.csv"

    # required paths
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Path to the raw ClinicalTrials.gov csv.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Path to save the structured sentencelevel csv.")

    # check if interactive environment (e.g. jupyter notebook)
    # or command line
    if "ipykernel" in sys.modules:
        # notebook mode
        args = parser.parse_args([])
    else:
        # command line mode
        args = parser.parse_args()
    
    main(args)