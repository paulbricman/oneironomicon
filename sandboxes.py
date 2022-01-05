from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, PrefixConstrainedLogitsProcessor


class DiscreteSandbox():
    def __init__(self, dialog_history=[], agent=None, states=1000, model='distilgpt2', turns=2, simulation_persona='student', agent_persona='teacher'):
        self.dialog_history = dialog_history
        self.agent = agent
        self.states = states
        self.turns = turns
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.simulation_persona = simulation_persona
        self.agent_persona = agent_persona


    def render_prompt(self, append_new=True):
        prompt = ''

        for contribution in self.dialog_history:
            if contribution['agent_turn']:
                prompt += self.agent_persona + ': ' + contribution['content'] + '\n'
            else:
                prompt += self.simulation_persona + ': ' + contribution['content'] + '\n'

        if append_new: 
            if not self.dialog_history[-1]['agent_turn']:
                prompt += self.agent_persona + ':'
            else:
                prompt += self.simulation_persona + ':'

        return prompt

    
    def simulation_reply(self):
        prompt = self.render_prompt()
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        prompt_token_length = len(inputs[0])
        
        outputs = self.model.generate(inputs, max_length=prompt_token_length+40, no_repeat_ngram_size=3,
            do_sample=False, prefix_allowed_tokens_fn=lambda x, y:self.force_one_paragraph(x, y, prompt_token_length), forced_eos_token_id=50256)
        reply = self.tokenizer.decode(outputs[0][prompt_token_length:])
        reply = reply.replace('<|endoftext|>', '').strip()

        self.dialog_history += [{'agent_turn': False, 'content': reply}]


    def agent_reply(self):
        reply = self.agent.reply(self.dialog_history)
        self.dialog_history += [{'agent_turn': True, 'content': reply}]


    def converse(self):
        for _ in range(self.turns):
            self.simulation_reply()
            self.agent_reply()


    def force_one_paragraph(self, batch_id, previous_token_ids, prompt_token_length):
        max_sentences = 1
        previous_token_ids = previous_token_ids.tolist()[prompt_token_length:]
        generated_text = self.tokenizer.decode(previous_token_ids)

        if '\n' in generated_text:
            return [50256]

        if len([e for e in generated_text if e in ['.', '!', '?']]) == max_sentences:
            return [50256]

        return range(0, 50255)