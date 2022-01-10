import json
import random
import pickle
from sentence_transformers import util
import torch
import numpy as np
from util import sample_quora_question, softmax


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
    def __init__(self, encoder_model, n_states=100):
        self.encoder_model = encoder_model

        self.replies = json.load(open('data/prompts.json')).values()
        self.replies = [e for sublist in self.replies for e in sublist][:3] + \
            ['Great, now please try to answer our original question again.']

        self.n_actions = len(self.replies)
        self.n_states = n_states
        self.q_table = np.zeros((n_states, self.n_actions))

    def quantize(self, reply):
        centroids = pickle.load(open('data/replies_centroids.pickle', 'rb'))
        centroids = torch.Tensor(centroids).float()

        db_embs = pickle.load(open('data/embs.pickle', 'rb'))
        current_emb = db_embs.get(reply, self.encoder_model.encode(
            reply, convert_to_tensor=True))
        db_embs[reply] = current_emb
        pickle.dump(db_embs, open('data/embs.pickle', 'wb'))

        cluster_index = util.semantic_search(current_emb, centroids)[
            0][0]['corpus_id']
        return cluster_index

    def train(self, sandbox, task, epochs=5, turns=5, gamma=0.5, alpha=0.5):
        print(self.q_table)

        for epoch in range(epochs):
            print('[*] Epoch', epoch)
            dialog_history = [
                {
                    'agent_turn': True,
                    'content': sample_quora_question()
                }
            ]

            sandbox.dialog_history = dialog_history
            sandbox.simulation_reply()
            next_state_index = None

            for turn in range(turns):
                if not next_state_index:
                    initial_state_index = self.quantize(
                        sandbox.dialog_history[-1]['content'])
                else:
                    initial_state_index = next_state_index

                probability_distribution = softmax(
                    self.q_table[initial_state_index], 1 - (epoch / epochs))

                greedy_action_index = np.random.choice(
                    range(self.n_actions), p=probability_distribution)
                greedy_action_reply = self.replies[greedy_action_index]
                sandbox.agent_reply(greedy_action_reply)

                sandbox.simulation_reply()
                reward = task.compute_reward(sandbox.dialog_history)
                next_state_index = self.quantize(
                    sandbox.dialog_history[-1]['content'])
                max_q_value = np.max(self.q_table[next_state_index])

                self.q_table[initial_state_index][greedy_action_index] += \
                    alpha * (reward + gamma * max_q_value -
                             self.q_table[initial_state_index][greedy_action_index])

            print(sandbox.render_prompt(append_new=False))
            print(self.q_table)
