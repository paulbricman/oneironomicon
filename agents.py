import json
import random
import pickle
from sentence_transformers import util
import torch
import numpy as np
from util import sample_quora_question, softmax
from tqdm import tqdm
import os


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
        self.replies = [
            e for sublist in self.replies for e in sublist] + [None]

        self.n_actions = len(self.replies)
        self.n_states = n_states
        self.q_table = np.zeros((n_states, self.n_actions))

    def quantize(self, reply):
        centroids = pickle.load(open('data/replies_centroids.pickle', 'rb'))
        centroids = torch.Tensor(centroids).float()

        if not os.path.exists('data/embs.pickle'):
            pickle.dump({}, open('data/embs.pickle', 'wb'))
        db_embs = pickle.load(open('data/embs.pickle', 'rb'))
        current_emb = db_embs.get(reply, self.encoder_model.encode(
            reply, convert_to_tensor=True))
        db_embs[reply] = current_emb
        pickle.dump(db_embs, open('data/embs.pickle', 'wb'))

        cluster_index = util.semantic_search(current_emb, centroids)[
            0][0]['corpus_id']
        return cluster_index

    def train(self, sandbox, task, episodes=20000, turns=10, gamma=0.5, alpha=0.5):
        dialog_histories = []
        reward_history = []

        for episode in tqdm(range(episodes)):
            pickle.dump({}, open('data/embs.pickle', 'wb'))
            pickle.dump({}, open('data/qaness.pickle', 'wb'))

            print('[*] Episode', episode)
            target_question = sample_quora_question()
            dialog_history = [
                {
                    'agent_turn': True,
                    'content': target_question
                }
            ]

            self.replies[-1] = 'Great, now please try to answer our original question again. ' + target_question

            sandbox.dialog_history = dialog_history
            sandbox.simulation_reply()
            next_state_index = None
            episode_rewards = []

            for turn in range(turns):
                if not next_state_index:
                    initial_state_index = self.quantize(
                        sandbox.dialog_history[-1]['content'])
                else:
                    initial_state_index = next_state_index

                probability_distribution = softmax(
                    self.q_table[initial_state_index], 1 - (episode / episodes))

                softmax_action_index = np.random.choice(
                    range(self.n_actions), p=probability_distribution)
                softmax_action_reply = self.replies[softmax_action_index]
                sandbox.agent_reply(softmax_action_reply)

                sandbox.simulation_reply()
                reward = task.compute_reward(sandbox.dialog_history)
                episode_rewards += [reward]
                next_state_index = self.quantize(
                    sandbox.dialog_history[-1]['content'])
                max_q_value = np.max(self.q_table[next_state_index])

                self.q_table[initial_state_index][softmax_action_index] += \
                    alpha * (reward + gamma * max_q_value -
                             self.q_table[initial_state_index][softmax_action_index])

            dialog_histories += [sandbox.dialog_history]

            print(sandbox.render_prompt(append_new=False))
            print(np.mean(episode_rewards))
            reward_history += [np.mean(episode_rewards)]

            pickle.dump([reward_history, dialog_histories, self.q_table], open(
                'data/training_outputs.pickle', 'wb'))

    def reply(self, dialog_history):
        user_reply = dialog_history[-1]['content']
        state = self.quantize(user_reply)
        self.replies[-1] = 'Great, now please try to answer our original question again. ' + dialog_history[0]['content']
        greedy_index = np.argmax(self.q_table[state])
        # print(greedy_index, self.q_table[state])
        return self.replies[greedy_index]
