# %%
import pandas as pd
import re
import torch
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM

# preprocessing -- groß-klein-schreibung ?
# headlines sometimes "Key inclusion criteria", ...
# %%
# read complete csv
# df = pd.read_csv('data_by_lea/Studies_with_id_and_EligibilityCriteria.csv')


# %%
# reads only the first five csv-entries
chunksize = 5
with pd.read_csv('data_by_lea/Studies_with_id_and_EligibilityCriteria.csv', chunksize=chunksize) as reader:
    for chunk in reader:
        df = chunk
        break


# %%
# separate ElgibilityCriteria into Inclusion and Exclusion Criteria
def split_criteria(ec):
    # Clean formatting vs windows newlines
    ec = ec.replace("\r", "").strip()

    re_pattern = re.compile(
        r"Inclusion Criteria:(.*)Exclusion Criteria:(.*)",
        re.DOTALL | re.IGNORECASE
    )

    matches = re_pattern.search(ec)
    if matches:
        inclusion = matches.group(1).strip()
        exclusion = matches.group(2).strip()
    else:
        inclusion = str()
        exclusion = str()
    
    return pd.Series([inclusion, exclusion])

# %%
df[["Inclusion", "Exclusion"]] = df["EligibilityCriteria"].apply(split_criteria)
df = df[["StudyNCTid", "Inclusion", "Exclusion"]]
df


# model test - biogpt
#try https://huggingface.co/clinicalnlplab/me-llama ? 
# taken from: https://huggingface.co/microsoft/biogpt
# %%
model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large")

# %%
tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large")

# %%
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# %%
set_seed(42)
# example_incl_criteria = "Male or female of non-childbearing potential, aged 18-75 years (both inclusive) at the time of signing the informed consent."
example_incl_criteria = "Male or female of non-childbearing potential, aged 18-75 years (both inclusive)"
len(example_incl_criteria)
# %%
# male or female of non-childbearing potential
# aged 18-75, inclusive at the time of signing informed consent
input_text = f"""Inclusion Criteria: {example_incl_criteria}
The inclusion criteria for this clinical study require the participant to be"""
generator(input_text, max_new_tokens = 200, num_return_sequences=3, do_sample=True)













# %%
"""
Examples Eligibility Criteria from first line of data

Inclusion Criteria:

* Male or female of non-childbearing potential, aged 18-75 years (both inclusive) at the time of signing the informed consent.
* Body mass index (BMI) between 20.0 and 39.9 kilogram per square metre (kg/m\^2) (both inclusive) at screening.
* Meeting the pre-defined glomerular filtration rate (GFR) criteria using estimated GFR (eGFR) based on the chronic kidney disease epidemiology collaboration (CKD-EPI) Creatinine Equation (2021) adjusted for estimated individual body surface area (BSA) for any of the renal function groups:
* For participants with normal renal function: eGFR of greater than or equal to 90 millilitres per minute (mL/min)
* Stage 2: For participants with mild renal impairment: eGFR of 60-89 mL/min
* Stage 3: For participants with moderate renal impairment: eGFR of 30-59 mL/min
* Stage 4: For participants with severe renal impairment: eGFR of 15-29 mL/min not requiring dialysis
* Stage 5: For participants with kidney failure: eGFR of less than 15 mL/min and requiring dialysis treatment


Exclusion Criteria:
* Any disorder which in the investigator's opinion might jeopardise participant's safety or compliance with the protocol.
* Presence or history of any clinically relevant respiratory, metabolic, renal, hepatic, cardiovascular, gastrointestinal, or endocrinological conditions (except conditions associated with renal impairment or kidney failure) as judged by the investigator.
* Use of drugs known to affect creatinine clearance including cephalosporin and aminoglycoside, antibiotics, flucytosine, cisplatin, cimetidine, trimethoprim, cibenzoline and nitrofurantoin within 14 days or 5 half-lives, whichever is greater, before planned dosing of the investigational medicinal product (IMP). 
"""

