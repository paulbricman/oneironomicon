import pandas as pd
import random
import numpy as np


def sample_quora_question(pool=10000):
    data = pd.read_csv('data/quora_duplicate_questions.tsv',
                       sep='\t', nrows=pool)
    return data['question1'][random.randint(0, pool - 1)]


def softmax(x, temperature):
    e_x = np.exp(x / temperature)
    return e_x / e_x.sum()
