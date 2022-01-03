import json
import random


class SingleMindedAgent():
    def reply(self, prompt):
        return 'Interesting...'


class RandomAgent():
    def reply(self, prompt):
        replies = json.load(open('data/prompts.json')).values()
        replies = [e for sublist in replies for e in sublist]
        reply = random.choice(replies)
        return reply