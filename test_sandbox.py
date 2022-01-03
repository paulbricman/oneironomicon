from sandboxes import DiscreteSandbox
from agents import RandomAgent, SingleMindedAgent
from util import sample_quora_question


agent = RandomAgent()

dialog_history = [
    {
        'agent_turn': True,
        'content': sample_quora_question()
    }
]

sandbox = DiscreteSandbox(dialog_history, agent, model='EleutherAI/gpt-neo-1.3B', turns=5)
sandbox.converse()
print(sandbox.render_prompt(append_new=False))