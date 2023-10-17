import pandas as pd
import numpy as np
from pythainlp import word_tokenize
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, pad_sequences

from keras.layers import LSTM, Embedding, Dense, Bidirectional
from keras.models import Sequential
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
import copy


# from gensim.models import KeyedVectors
# word2vec_model = KeyedVectors.load_word2vec_format('LTW2V_v0.1.bin', binary=True, unicode_errors='ignore')

from flask import Flask,request,abort,jsonify
import requests
from app.Config import *
import json


def tokenize(sentence) :
   return word_tokenize(sentence, engine='newmm')

# def map_word_index(word_seq) :
#   indices = []
#   for word in word_seq:
#     if word in word2vec_model.key_to_index:
#       indices.append(word2vec_model.key_to_index[word] + 1)
#     else :
#       indices.append(1)
#   return indices

question_data = pd.read_csv('NLP_Elder_Companion.csv', encoding='utf8')
answer_data = pd.read_csv('NLP_Elder_Answer.csv')
data = np.array(question_data.values)
text = np.array([word_tokenize(text, engine="newmm", keep_whitespace=False) for text in data[:, 1]])

# _class = np.array(data['Class'].values)
_class = np.array(question_data['Class'].values)

words_list = {}
max_len_sentence = 0
i = 1

for sentence in text:
    if max_len_sentence < len(sentence) :
        max_len_sentence = len(sentence)

    for word in sentence:
        if words_list.get(word) == None:
            words_list[word] = i
            i += 1

max_vocab = len(words_list)
word_sequence = copy.deepcopy(text)

for i in range(len(word_sequence)) :
    for j in range(len(word_sequence[i])) :
        word_sequence[i][j] = words_list[text[i][j]]

x = pad_sequences(word_sequence, maxlen=max_len_sentence, padding='post')
y = to_categorical(_class, num_classes=9)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)

model = Sequential()
model.add(Embedding(input_dim=max_vocab,
                    output_dim=500,
                    input_length=max_len_sentence))
model.add(Bidirectional(LSTM(500)))
model.add(Dense(9, activation="softmax"))

lost_func = CategoricalCrossentropy()
otm = Adam(learning_rate=0.001)
model.compile(optimizer=otm, loss=lost_func, metrics=["accuracy"])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=69, epochs=20, verbose=1)



app=Flask(__name__)

@app.route('/webhook', methods=['POST', 'GET'])

def webhook():
    if request.method == 'POST':
        payload =request.json
        Reply_token=payload['events'][0]['replyToken']
        message=payload['events'][0]['message']['text']


        input_sequence = word_tokenize(message, engine="newmm", keep_whitespace=False)

        for i in range(len(input_sequence)) :
            if words_list.get(input_sequence[i]) != None :
                input_sequence[i] = words_list[input_sequence[i]]
            else :
                input_sequence[i] = 0

        input_sequence = np.array([input_sequence])

        input_data = pad_sequences(input_sequence, maxlen=max_len_sentence, padding='post')
        logit = model.predict(input_data, batch_size=32)
        predict = [pred for pred in np.argmax(logit,axis=1)][0]

        if (np.max(logit)) > 0.5:
            label = predict[0]
            Reply_text = answer_data["Answer"][label]
        else:
            Reply_text = "ขออภัยค่ะ ฉันไม่เข้าใจคำถาม กรุณาถามคำถามใหม่ค่ะ"
        
        print(Reply_text, flush=True)

        ReplyMessage(Reply_token,Reply_text,Channel_access_token)
        return jsonify({'success': True}), 200

    elif request.method == 'GET':
        return "this is method GET!!!", 200
    else:
        abort(400)


def ReplyMessage(Reply_token,TextMessage,Line_Acees_Token):
    LINE_API='https://api.line.me/v2/bot/message/reply/'
    
    Authorization='Bearer {}'.format(Line_Acees_Token)
    print(Authorization)
    headers={
        'Content-Type':'application/json; char=UTF-8',
        'Authorization':Authorization
    }

    data={
        "replyToken":Reply_token,
        "messages":[{
            "type":"text",
            "text":TextMessage
        }
        ]
    }
    data=json.dumps(data) # ทำเป็น json
    r=requests.post(LINE_API,headers=headers,data=data)
    return 200