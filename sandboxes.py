import re
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, PrefixConstrainedLogitsProcessor
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
import torch


class DiscreteSandbox():
    def __init__(self, dialog_history=[], agent=None, states=1000, model='distilgpt2', encoder_model='all-MiniLM-L6-v2', turns=5, simulation_persona='A', agent_persona='Q'):
        self.dialog_history = dialog_history
        self.agent = agent
        self.states = states
        self.turns = turns
        self.model = model
        self.simulation_persona = simulation_persona
        self.agent_persona = agent_persona
        self.encoder_model = encoder_model

    def render_prompt(self, append_new=True):
        prompt = ''

        # print(self.dialog_history)
        for contribution in self.dialog_history:
            if contribution['agent_turn']:
                prompt += self.agent_persona + ': ' + \
                    contribution['content'] + '\n'
            else:
                prompt += self.simulation_persona + \
                    ': ' + contribution['content'] + '\n'

        if append_new:
            if not self.dialog_history[-1]['agent_turn']:
                prompt += self.agent_persona + ':'
            else:
                prompt += self.simulation_persona + ':'

        return prompt

    def simulation_reply(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if isinstance(self.model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model).to(device)

        prompt = self.render_prompt()
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_token_length = len(inputs[0])

        outputs = self.model.generate(inputs, max_length=prompt_token_length+40, no_repeat_ngram_size=3,
                                      do_sample=False, prefix_allowed_tokens_fn=lambda x, y: self.force_one_paragraph(x, y, prompt_token_length), forced_eos_token_id=50256)
        reply = self.tokenizer.decode(outputs[0][prompt_token_length:])
        reply = reply.replace('<|endoftext|>', '').strip()

        self.dialog_history += [{'agent_turn': False, 'content': reply}]

    def agent_reply(self, reply=None):
        if not reply:
            reply = self.agent.reply(self.dialog_history)
        self.dialog_history += [{'agent_turn': True, 'content': reply}]

    def converse(self, user=False, task=None):
        rewards = []
        for _ in range(self.turns):
            if not user:
                self.simulation_reply()
            else:
                print('---')
                print(self.render_prompt())
                self.dialog_history += [{'agent_turn': False, 'content': input()}]
                if task is not None:
                    rewards += [task.compute_reward(self.dialog_history)]
            self.agent_reply()
        return rewards

    def force_one_paragraph(self, batch_id, previous_token_ids, prompt_token_length):
        max_sentences = 1
        previous_token_ids = previous_token_ids.tolist()[prompt_token_length:]
        generated_text = self.tokenizer.decode(previous_token_ids)

        if '\n' in generated_text:
            return [50256]

        if len([e for e in generated_text if e in ['.', '!', '?']]) == max_sentences:
            return [50256]

        return range(0, 50255)

    def discretize(self, dialog_histories_path, n_centroids=100):
        if isinstance(self.encoder_model, str):
            self.encoder_model = SentenceTransformer(self.encoder_model)

        if not os.path.exists('data/replies_embeddings.pickle'):
            dialog_histories = pickle.load(open(dialog_histories_path, 'rb'))

            all_simulation_replies = []
            for dialog_history in dialog_histories:
                simulation_replies = [e['content']
                                      for e in dialog_history if e['agent_turn'] == False]
                all_simulation_replies += simulation_replies

            all_embeddings = self.encoder_model.encode(all_simulation_replies)
            replies_embeddings_dict = dict(
                zip(all_simulation_replies, all_embeddings))
            pickle.dump(replies_embeddings_dict, open(
                'data/replies_embeddings.pickle', 'wb'))

        replies_embeddings_dict = pickle.load(
            open('data/replies_embeddings.pickle', 'rb'))
        kmeans = KMeans(n_centroids).fit(
            list(replies_embeddings_dict.values()))
        pickle.dump(kmeans.cluster_centers_, open(
            'data/replies_centroids.pickle', 'wb'))

        labels = kmeans.labels_
        for label in range(n_centroids):
            for reply_idx, reply in enumerate(list(replies_embeddings_dict.keys())):
                if label == labels[reply_idx]:
                    print(label, reply)

            input()
