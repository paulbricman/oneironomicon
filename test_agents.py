from agents import QLearningAgent
from sentence_transformers import SentenceTransformer
from sandboxes import DiscreteSandbox
from tasks import QuestionAnsweringAssistance
import torch


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
encoder_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
agent = QLearningAgent(encoder_model=encoder_model)
sandbox = DiscreteSandbox(encoder_model=encoder_model, model='EleutherAI/gpt-neo-1.3B')
task = QuestionAnsweringAssistance(encoder_model=encoder_model)
agent.train(sandbox, task)
