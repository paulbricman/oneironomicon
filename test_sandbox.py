from sandboxes import DiscreteSandbox
from agents import RandomAgent, SingleMindedAgent
from util import sample_quora_question
import pickle


agent = RandomAgent()
dialog_histories = []
sandbox = DiscreteSandbox(None, agent, model='EleutherAI/gpt-neo-125M', turns=5)

for sample in range(3):
    dialog_history = [
        {
            'agent_turn': True,
            'content': sample_quora_question()
        }
    ]

    sandbox.dialog_history = dialog_history
    sandbox.converse()
    dialog_histories += [sandbox.dialog_history]


for dialog_history in dialog_histories:
    sandbox.dialog_history = dialog_history
    print(sandbox.render_prompt(append_new=False), '\n\n')

pickle.dump(dialog_histories, open('data/dialog_histories.pickle', 'wb'))