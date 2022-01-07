import json
import random
import pickle
from numpy import dtype
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import torch


class SingleMindedAgent():
    def reply(self, prompt):
        return 'Interesting...'


class RandomAgent():
    def reply(self, prompt):
        replies = json.load(open('data/prompts.json')).values()
        replies = [e for sublist in replies for e in sublist]
        reply = random.choice(replies)
        return reply


class QLearningAgent():
    def __init__(self, encoder_model):
        self.encoder_model = encoder_model

    def quantize(self, reply):
        centroids = pickle.load(open('data/replies_centroids.pickle', 'rb'))
        centroids = torch.Tensor(centroids).float()

        db_embs = pickle.load(open('data/embs.pickle', 'rb'))
        current_emb = db_embs.get(reply, self.encoder_model.encode(
            reply, convert_to_tensor=True))
        db_embs[reply] = current_emb
        pickle.dump(db_embs, open('data/embs.pickle', 'wb'))
        
        cluster_index = util.semantic_search(current_emb, centroids)[0][0]['corpus_id']
        print(cluster_index)
        return cluster_index

