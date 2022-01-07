from sentence_transformers import CrossEncoder, util
import pickle
import os


class QuestionAnsweringAssistance():
    def __init__(self, encoder_model, qa_model='cross-encoder/qnli-distilroberta-base'):
        self.encoder_model = encoder_model
        self.qa_model = CrossEncoder(qa_model)


    def compute_reward(self, dialog_history, qa_threshold=0.7, sim_threshold=0.5):
        question = dialog_history[0]['content']
        student_replies = [e['content'] for e in dialog_history if e['agent_turn'] == False]

        if not os.path.exists('data/qaness.pickle'):
            pickle.dump({}, open('data/qaness.pickle', 'wb'))

        db_qaness = pickle.load(open('data/qaness.pickle', 'rb'))
        past_qaness = [db_qaness.get(e) for e in student_replies[:-1]]
        current_qaness = self.qa_model.predict([(question, student_replies[-1])])[0]
        db_qaness[student_replies[-1]] = current_qaness
        good_student_replies = [e for e_idx, e in enumerate(student_replies[:-1]) if past_qaness[e_idx] > qa_threshold]
        pickle.dump(db_qaness, open('data/qaness.pickle', 'wb'))

        if not os.path.exists('data/embs.pickle'):
            pickle.dump({}, open('data/embs.pickle', 'wb'))

        db_embs = pickle.load(open('data/embs.pickle', 'rb'))
        past_embs = [db_embs.get(e) for e in good_student_replies]
        current_emb = self.encoder_model.encode(student_replies[-1], convert_to_tensor=True)
        db_embs[student_replies[-1]] = current_emb
        pickle.dump(db_embs, open('data/embs.pickle', 'wb'))
        
        if len(past_embs) == 0:
            return current_qaness

        highest_similarity = util.semantic_search(current_emb, past_embs)[0][0]['score']

        if highest_similarity < sim_threshold:
            return current_qaness
        else:
            return 0