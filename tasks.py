from sentence_transformers import CrossEncoder, SentenceTransformer, util


class QuestionAnsweringAssistance():
    def __init__(self, encoder_model='all-MiniLM-L6-v2', qa_model='cross-encoder/qnli-distilroberta-base'):
        self.encoder_model = SentenceTransformer(encoder_model)
        self.qa_model = CrossEncoder(qa_model)


    def compute_reward(self, dialog_history, qa_threshold=0.7, sim_threshold=0.5):
        question = dialog_history[0]['content']
        student_replies = [e['content'] for e in dialog_history if e['agent_turn'] == False]

        good_student_replies = [e for e in student_replies[:-1] if self.qa_model.predict([(question, e)])[0] > qa_threshold]
        
        corpus_embeddings = self.encoder_model.encode(good_student_replies, convert_to_tensor=True)
        query_embedding = self.encoder_model.encode(student_replies[-1], convert_to_tensor=True)
        
        highest_similarity = util.semantic_search(query_embedding, corpus_embeddings)[0][0]['score']

        if highest_similarity < sim_threshold:
            return self.qa_model.predict([(question, student_replies[-1])])[0]
        else:
            return 0