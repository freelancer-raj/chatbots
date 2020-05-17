import os
import json
from joblib import load
import numpy as np
import random
from keras.models import load_model
from train_bot import preprocess_doc_and_extract_words, models_dir, data_dir

model = load_model(os.path.join(models_dir, "chatbot_model.h5"))
intents = json.loads(open(os.path.join(data_dir, "intents.json")).read())
words = load(os.path.join(models_dir, "words.h5"))
classes = load(os.path.join(models_dir, "classes.h5"))


def build_data_for_model(query_tokens, words, encoding="binary"):
    if encoding == "binary":
        row = [[int(w in query_tokens) for w in words]]
    else:
        raise Exception(f"Encoding {encoding} not supported for training data!")
    return np.array(row)


def predict_intent(data, model, n=1):
    preds = model.predict(data)[0]
    predicted_probabilities_by_class = {i:v for i,v in enumerate(preds)}
    top_classes = [k for k, _ in sorted(
        predicted_probabilities_by_class.items(), key=lambda item: item[1], reverse=True
    )]
    intent_names = [classes[i] for i in top_classes]

    return intent_names[:n]


def generate_response(predicted_intent, random_response=True, idf=False, qtd=False):
    if random_response:
        for i in intents["intents"]:
            if i["tag"] in predicted_intent:
                return random.choice(i["responses"])


if __name__ == "__main__":
    print("Enter your query to start chatting!")
    q = str(input("Q:"))
    exit = 0
    while not exit:
        query_tokens = preprocess_doc_and_extract_words(q)
        print(query_tokens)
        data = build_data_for_model(query_tokens, words)

        intent = predict_intent(data, model)
        print(intent)
        print(f"A: {generate_response(intent)}")
        if "goodbye" in intent:
            exit = 1
        else:
            q = str(input("Q:"))

