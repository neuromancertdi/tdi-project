import numpy

class Tokenizer:
    def __init__(self, token_list):
        self.token_to_idx = {token: i for i, token in enumerate(token_list)}
        self.idx_to_token = {i: token for i, token in enumerate(token_list)}
        self.tokens_num = len(token_list)


    def token_to_ohe(self, token):
        vector = [0 for _ in range(self.tokens_num)]

        idx = self.token_to_idx[token]
        vector[idx] = 1

        return vector


    def sequence_to_vector(self, sequence):
        vector = [0 for _ in range(self.tokens_num)]

        for item in sequence:
            idx = self.token_to_idx[item]
            vector[idx] += 1

        return vector


    def vector_to_token(self, vector):
        idx = numpy.argmax(vector)
        token = self.idx_to_token[idx]

        return token