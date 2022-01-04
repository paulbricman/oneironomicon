from tasks import QuestionAnsweringAssistance

dialog_history = [
    {
        'agent_turn': True,
        'content': 'What is a popular scripting language?'
    },
    {
        'agent_turn': False,
        'content': 'Python is a good one.'
    },
    {
        'agent_turn': True,
        'content': 'Are you sure?'
    },
    {
        'agent_turn': False,
        'content': 'Basketball is nice.'
    },
    {
        'agent_turn': True,
        'content': 'What is a popular scripting language?'
    },
    {
        'agent_turn': False,
        'content': 'The most widely use scripting language by developers in 2021 is Javascript.'
    },
]

task = QuestionAnsweringAssistance()
print(task.compute_reward(dialog_history))