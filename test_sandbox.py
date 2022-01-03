from sandboxes import DiscreteSandbox
from agents import SingleMindedAgent


agent = SingleMindedAgent()
dialog_history = [
    {
        'agent_turn': True,
        'content': 'What is the meaning of life?'
    }
]

sandbox = DiscreteSandbox(dialog_history, agent, model='EleutherAI/gpt-neo-1.3B', turns=5)
sandbox.converse()
print(sandbox.render_prompt(append_new=False))