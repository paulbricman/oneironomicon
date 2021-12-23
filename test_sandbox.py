from sandboxes import DiscreteSandbox

dialog_history = [
    {
        'agent_turn': True,
        'content': 'Hello! How are you?'
    }
]

sandbox = DiscreteSandbox(dialog_history=dialog_history)
sandbox.simulation_reply()
# print(sandbox.dialog_history)