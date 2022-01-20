from tkinter import dialog
from agents import QLearningAgent
from sentence_transformers import SentenceTransformer
from sandboxes import DiscreteSandbox
from tasks import QuestionAnsweringAssistance
import torch
import pickle

from util import sample_quora_question

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
encoder_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
agent = QLearningAgent(encoder_model=encoder_model)
sandbox = DiscreteSandbox(encoder_model=encoder_model,
                          model='distilgpt2', agent=agent)
task = QuestionAnsweringAssistance(encoder_model=encoder_model)

agent.q_table = pickle.load(open(
    '20K_training_artifacts/training_outputs.pickle', 'rb'))[2]

for episode in range(10):
    dialog_history = {
        'agent_turn': True,
        'content': sample_quora_question()
    }
    sandbox.dialog_history = [dialog_history]
    sandbox.converse()
    print(sandbox.render_prompt(append_new=False))
