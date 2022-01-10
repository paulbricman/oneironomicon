from agents import QLearningAgent
from sentence_transformers import SentenceTransformer
from sandboxes import DiscreteSandbox
from tasks import QuestionAnsweringAssistance


encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
agent = QLearningAgent(encoder_model=encoder_model)
sandbox = DiscreteSandbox(encoder_model=encoder_model)
task = QuestionAnsweringAssistance(encoder_model=encoder_model)
agent.train(sandbox, task)
