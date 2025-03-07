"""
FOR TERMINAL:
pip install pandas
pip install numpy
pip install transformers
pip install datasets
pip install sentencepiece
pip install evaluate
pip install rouge_score
pip install evaluate
pip install torch

pip install accelerate -U
pip install protobuf


"""

# default libraries (no need extra import)
import pandas as pd
import numpy as np

#library for loading dataset (not absolute path) 
import os

# other libraries
import transformers
import datasets
import sentencepiece
import torch

# libs for evaluation
import evaluate
import rouge_score
rouge = evaluate.load("rouge")

from datasets import Dataset

# Import modules from transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")

"""
PATH TO THE FILE 
"""

current_directory = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_directory, 'very_huge_ds.csv')

#file_path = '/Users/liza/PycharmProjects/title-generator/very_huge_ds.csv'

"""
FILE READING
TEST/TRAIN SPLITTING
"""

title_df = pd.read_csv(file_path)
title_dataset = Dataset.from_pandas(title_df)
title_dataset = title_dataset.train_test_split(test_size=0.2)


"""
TEXT FOR MODEL PREPROCESSING:
TOKENIZING with T5Tokenizer
"""

def preprocess_function(item):
   inputs = ["summarize: " + doc for doc in item["text"]]
   model_inputs = tokenizer(inputs, max_length=530, truncation=True)
   labels = tokenizer(text_target=item["clickbate_title"], max_length=30, truncation=True)
   model_inputs["labels"] = labels["input_ids"]
   return model_inputs


tokenized_title_dataset = title_dataset.map(preprocess_function, batched=True)

"""
FOR EVALUATION
"""

def compute_metrics(eval_pred):
   # Unpacks the evaluation predictions tuple into predictions and labels.
   predictions, labels = eval_pred


   # Decodes the tokenized predictions back to text, skipping any special tokens (e.g., padding tokens).
   decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)


   # Replaces any -100 values in labels with the tokenizer's pad_token_id.
   # This is done because -100 is often used to ignore certain tokens when calculating the loss during training.
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)


   # Decodes the tokenized labels back to text, skipping any special tokens (e.g., padding tokens).
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


   # Computes the ROUGE metric between the decoded predictions and decoded labels.
   # The use_stemmer parameter enables stemming, which reduces words to their root form before comparison.
   result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)


   # Calculates the length of each prediction by counting the non-padding tokens.
   prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]


   # Computes the mean length of the predictions and adds it to the result dictionary under the key "gen_len".
   result["gen_len"] = np.mean(prediction_lens)


   # Rounds each value in the result dictionary to 4 decimal places for cleaner output, and returns the result.
   return {k: round(v, 4) for k, v in result.items()}


"""
PARAMETERS FOR MODEL
"""


model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

output_dir = "t5_small_clickbait"
os.makedirs(output_dir, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
   output_dir= output_dir,
   evaluation_strategy="epoch",
   learning_rate=2e-5,
   per_device_train_batch_size=2,
   per_device_eval_batch_size=2,
   weight_decay=0.01,
   save_total_limit=3,
   num_train_epochs=3,
   predict_with_generate=True,
   fp16=False,  # Disable mixed precision training
)

"""
MODEL TRAIN
"""

trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_title_dataset["train"],
   eval_dataset=tokenized_title_dataset["test"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()

"""
MODEL SAVE
"""

trainer.save_model(output_dir)


"""TEST ON EXAMPLE"""

text = title_dataset['test'][4]['text']
text = "summarize: " + text
text

from transformers import pipeline

summarizer = pipeline("summarization", model="t5_small_clickbait")
pred = summarizer(text, max_length=15, min_length=5)

print(pred)
