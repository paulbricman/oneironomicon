import pandas as pd
from sentence_transformers import SentenceTransformer, util

query = 'How can I develop a chatbot? How can AI be used in education? Why does GPT-2 work?'

encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = encoder_model.encode(query, convert_to_tensor=True)

data = pd.read_csv('data/quora_duplicate_questions.tsv',
                   sep='\t', nrows=100000)

all_questions = list(set([e.strip() for e in data['question1'].tolist()]))
corpus_embeddings = encoder_model.encode(all_questions, convert_to_tensor=True)

results = util.semantic_search(query_embedding, corpus_embeddings, top_k=1000)

valid_questions = []

for result in results[0]:
    valid_questions += [all_questions[result['corpus_id']]]

open('data/filtered_quora.txt', 'w').write('\n'.join(valid_questions))
