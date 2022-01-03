import pandas as pd
import random


def sample_quora_question(pool=10000):
    data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t', nrows=pool)
    return data['question1'][random.randint(0, pool - 1)]