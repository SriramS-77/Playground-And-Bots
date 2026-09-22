import tensorflow as tf
from keras.layers import Normalization
import numpy as np

MODEL_FILENAME = "big_model.keras"
MODEL_PATH = f"scoring_service/models/{MODEL_FILENAME}"   # "models/mid_model.keras"
CONTEXT_LENGTH = 100

def handle_data(data, max_length=100):
    new_data = []
    for idx, session in enumerate(data):
        session_length = len(session)
        cur = 0
        while cur * max_length + max_length <= session_length:
            new_data.append(session[cur * max_length: (cur + 1) * max_length])
            cur += 1
        last_section = session[cur * max_length:]
        last_section += [session[-1] for _ in range(max_length - len(last_section))]
        new_data.append(last_section)

    new_data = np.array(new_data)
    layer = Normalization(axis=1)
    layer.adapt(new_data)
    return new_data


class HumanityScorer:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)

    def predict(self, mouse_movement):
        data = [mouse_movement]
        data = handle_data(data=data, max_length=CONTEXT_LENGTH)   # np.array(data).reshape((-1, 2))
        humanity_scores = self.model.predict(data, verbose=0)
        humanity_scores = [score for [score] in humanity_scores]
        # print(data.shape)
        # print(humanity_scores)
        humanity_score = sum(humanity_scores) / len(humanity_scores)
        # print(humanity_score)
        return humanity_score

scorer = HumanityScorer()
