from agents import QLearningAgent
from sentence_transformers import SentenceTransformer

encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
agent = QLearningAgent(encoder_model)
while True:
    agent.quantize(input())