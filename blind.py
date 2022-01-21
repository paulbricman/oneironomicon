from agents import QLearningAgent, RandomAgent
from sentence_transformers import SentenceTransformer
from sandboxes import DiscreteSandbox
from tasks import QuestionAnsweringAssistance
import torch
import pickle
import numpy as np
from util import sample_quora_question
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
encoder_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
agent = QLearningAgent(encoder_model=encoder_model)
random_agent = RandomAgent()
sandbox = DiscreteSandbox(encoder_model=encoder_model,
                          model='EleutherAI/gpt-neo-1.3B', agent=agent)
task = QuestionAnsweringAssistance(encoder_model=encoder_model)

agent.q_table = pickle.load(open(
    '20K_training_artifacts/training_outputs.pickle', 'rb'))[2]

episodes = 5
prompts = [{
        'agent_turn': True,
        'content': sample_quora_question()
    } for e in range(episodes)]

order = ['trained', 'untrained']
random.shuffle(order)
for group in order:
    if group == 'trained':
        sandbox.agent = agent
        all_trained_rewards = []
        for episode in range(episodes):
            dialog_history = prompts[episode]
            sandbox.dialog_history = [dialog_history]
            rewards = sandbox.converse(user=True, task=task)
            print(sandbox.render_prompt(append_new=False))
            all_trained_rewards += [np.mean(rewards)]
    else:
        sandbox.agent = random_agent
        all_untrained_rewards = []
        for episode in range(episodes):
            dialog_history = prompts[episode]
            sandbox.dialog_history = [dialog_history]
            rewards = sandbox.converse(user=True, task=task)
            print(sandbox.render_prompt(append_new=False))
            all_untrained_rewards += [np.mean(rewards)]

print(all_trained_rewards, all_untrained_rewards, np.mean(all_trained_rewards), np.mean(all_untrained_rewards))