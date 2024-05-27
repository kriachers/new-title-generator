import pandas as pd
import datasets
from datasets import Dataset
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_directory, 'very_huge_ds.csv')

title_df = pd.read_csv(file_path)
title_dataset = Dataset.from_pandas(title_df)
title_dataset = title_dataset.train_test_split(test_size=0.2)

"""TEST ON EXAMPLE"""

text = title_dataset['test'][4]['text']
text = "summarize: " + text
text

from transformers import pipeline

summarizer = pipeline("summarization", model="t5_small_clickbait")
pred = summarizer(text, max_length=30, min_length=5)

print(pred)