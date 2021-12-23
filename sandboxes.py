from transformers import pipeline


class DiscreteSandbox():
    def __init__(self, states=1000, model='distilgpt2', turns=10, simulation_persona='student', agent_persona='instructor', dialog_history=[]):
        self.states = states
        self.turns = turns
        self.pipeline = pipeline('text-generation', model=model)
        self.simulation_persona = simulation_persona
        self.agent_persona = agent_persona
        self.dialog_history = dialog_history


    def render_prompt(self):
        prompt = ''

        for contribution in self.dialog_history:
            if contribution['agent_turn']:
                prompt += self.agent_persona + ': ' + contribution['content'] + '\n'
            else:
                prompt += self.simulation_persona + ': ' + contribution['content'] + '\n'
            
        if not self.dialog_history[-1]['agent_turn']:
            prompt += self.agent_persona + ':'
        else:
            prompt += self.simulation_persona + ':'

        return prompt

    
    def simulation_reply(self):
        prompt = self.render_prompt()
        reply = self.pipeline(prompt)[0]['generated_text'][len(prompt):].split('\n')[0]
        print('<' + prompt + '|' + reply + '>')
        self.dialog_history += [{'agent_turn': False, 'content': reply}]