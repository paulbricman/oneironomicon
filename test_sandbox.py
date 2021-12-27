from sandboxes import DiscreteSandbox
from agents import SingleMindedAgent


agent = SingleMindedAgent()
dialog_history = [
    {
        'agent_turn': True,
        'content': 'What if we approached this from a different perspective?'
    }
]

sandbox = DiscreteSandbox(dialog_history, agent)
sandbox.converse()
print(sandbox.render_prompt(append_new=False))