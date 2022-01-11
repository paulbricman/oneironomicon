# To control logging level for various modules used in the application:
import logging
import re
from agents import RandomAgent
from util import sample_filtered_quora_question
from sandboxes import DiscreteSandbox
import tqdm
import pickle


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.
    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


set_global_logging_level(logging.ERROR)


agent = RandomAgent()
dialog_histories = []
sandbox = DiscreteSandbox(
    None, agent, model='EleutherAI/gpt-neo-125M', turns=5)

for sample in tqdm(range(10000)):
    dialog_history = [
        {
            'agent_turn': True,
            'content': sample_filtered_quora_question()
        }
    ]

    sandbox.dialog_history = dialog_history
    sandbox.converse()
    dialog_histories += [sandbox.dialog_history]


for dialog_history in dialog_histories:
    sandbox.dialog_history = dialog_history
    print(sandbox.render_prompt(append_new=False), '\n\n')

pickle.dump(dialog_histories, open(
    'data/filtered_dialog_histories.pickle', 'wb'))
