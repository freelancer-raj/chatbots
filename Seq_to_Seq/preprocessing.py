import numpy as np
import re
import os
from joblib import dump
from tensorflow.keras import utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_LEN = 250

working_directory = os.getcwd()
data_directory = os.path.join(working_directory, 'data')
models_directory = os.path.join(working_directory, 'models')

# common contractions
common_contractions = {
    r"i'm": "i am",
    r"he's": "he is",
    r"she's": "she is",
    r"that's": "that is",
    r"what's": "what is",
    r"where's": "where is",
    r"\'ll": " will",
    r"\'ve": " have",
    r"\'re": " are",
    r"\'d": " would",
    r"won't": "will not",
    r"can't": "can not",
    r"n't": " not",
    r"&": "and",
    r"it's": "it is",
    r"how's": "how is",
    r"[$()\"#/@;:<>{}+=-`|.?,\'*%_\[\]]|(-)+": ""
}


def clean_text(text):
    # lowercase
    text = str(text).lower()

    # replace common contractions
    for contraction, replacement in common_contractions.items():
        text = re.sub(contraction, replacement, text)
    return text


def get_max_len(data, cap=MAX_LEN):
    return min(max([len(t.split()) for t in data]), cap)


def create_tokenizer(data, char_level=False, include_oov_token=False):
    tokenizer = Tokenizer(char_level=char_level, oov_token=include_oov_token)
    tokenizer.fit_on_texts(data)
    word_idx_map = tokenizer.word_index
    idx_word_map = {v:k for k,v in word_idx_map.items()}
    return tokenizer, word_idx_map, idx_word_map


def tokenize_sentences_and_pad(data, tokenizer, max_len, pad_type="post"):
    tokenized_sents = tokenizer.texts_to_sequences(data)
    return pad_sequences(tokenized_sents, padding=pad_type, maxlen=max_len)


# Read the input data
lines = open(os.path.join(data_directory, 'movie_lines.txt'), encoding='utf-8', errors='ignore').read().split('\n')
conversations = open(os.path.join(data_directory, 'movie_conversations.txt'), encoding='utf-8', errors='ignore').read().split('\n')

# Load the data from txt file to list
data_separator = " +++$+++ "

id_to_line = {}
for line in lines:
    _line = line.split(data_separator)
    if len(_line) == 5:
        id_to_line[_line[0]] = _line[-1].strip()

junk_characters = r"['\s\[\]]"
conversations_ids = [re.sub(junk_characters, "", conv.split(data_separator)[-1]).split(",") for conv in conversations[:-1]]

questions = []
answers = []

for conv in conversations_ids:
    for i in range(len(conv)-1):
        questions.append(id_to_line[conv[i]])
        answers.append(id_to_line[conv[i+1]])

cleaned_questions = [q for q in questions]
cleaned_answers = [f"%{a}$" for a in answers]  # Using % as a SOS character and $ as EOS

print("Loaded all questions and answers")

# Truncating and keeping only text shorter than max len
training_questions = []
training_answers = []
for q,a in zip(cleaned_questions, cleaned_answers):
  if (len(q) <= MAX_LEN) and (len(a) <= MAX_LEN):
    training_questions.append(q)
    training_answers.append(a)

print(f"Number of questions with length less than max len - {len(training_questions)}")
print(f"Number of answers with length less than max len - {len(training_answers)}")

MAX_LEN = get_max_len(training_questions+training_answers)
q_tokenizer, q_word_idx_map, _ = create_tokenizer(training_questions, char_level=True, include_oov_token=True)
questions_tokenized = tokenize_sentences_and_pad(training_questions, q_tokenizer, MAX_LEN)
print(f"Tokenized questions. Shape of encoder input data - {questions_tokenized.shape}")

a_tokenizer, a_word_idx_map, a_idx_word_map = create_tokenizer(training_answers, char_level=True, include_oov_token=True)
answers_tokenized = tokenize_sentences_and_pad(training_answers, a_tokenizer, MAX_LEN)
print(f"Tokenized answers. Shape of decoder input data - {answers_tokenized.shape}")

num_output_tokens = len(a_word_idx_map) + 1
print(f"Vocabulary size - {num_output_tokens}")

target_text = []
for ans in training_answers:
    target_text.append(ans[1:])

decoder_output_tokenized = tokenize_sentences_and_pad(target_text, a_tokenizer, MAX_LEN)
decoder_targets_ohe = utils.to_categorical(decoder_output_tokenized, num_output_tokens)
decoder_target_data = np.array(decoder_targets_ohe)
print(f"Decoder output shape - {decoder_target_data.shape}")

dump(questions_tokenized, os.path.join(data_directory, "encoder_input.h5"))
dump(answers_tokenized, os.path.join(data_directory, "decoder_input.h5"))
dump(decoder_targets_ohe, os.path.join(data_directory, "decoder_output_onehot.h5"))
dump(q_word_idx_map, os.path.join(data_directory, "input_word_idx_map.h5"))
dump(a_word_idx_map, os.path.join(data_directory, "output_word_idx_map.h5"))


