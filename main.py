### Author: Riya Nakarmi ###
### College Project ###


import random
import json
import pickle
import numpy as np
import tensorflow as tf 

import re
from nltk.corpus import stopwords

import nltk
# nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
#     return sentence_words

def preprocess_text(text):
    # text = text.lower()
    # print('1 :',text)
    # text = re.sub(r'\d+','',text)
    # print('2 :',text)
    # text = re.sub(r'[^\w\s]','',text)
    # print('3 :',text)
    tokens = nltk.word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    print('8 :',len(stop_words))
    new_stopwords = ['microsoft', 'dynamic','dynamics','provide']
    new_stopwords_list = stop_words.union(new_stopwords)
    print('9 :',len(new_stopwords_list))
    filtered_tokens = [word for word in tokens if word not in new_stopwords_list]
    return filtered_tokens

def perform_lemmatization(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def clean_up_sentence(text):
    tokens = preprocess_text(text)
    filtered_tokens = remove_stopwords(tokens)
    lemmaized_tokens = perform_lemmatization(filtered_tokens)
    # lemmaized_tokens = perform_lemmatization(tokens)
    print('4 :',lemmaized_tokens)
    clean_text=' '.join(lemmaized_tokens)
    print('5 :',clean_text)
    if " " in  clean_text:
        clean_text =  clean_text.replace(" ", "*")
        print('12:', clean_text)
    # sentence_words = nltk.word_tokenize(clean_text)
    sentence_words = [clean_text]
    print('8 :',sentence_words)
    
    return sentence_words

def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
# def get_response(sentence):
    bow = bag_of_words(sentence)
    print('6 :',bow)
    res = model.predict(np.array([bow]))[0]
    print('7 :',res)
    ERROR_THRESHOLD = 0.05
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # print(results)

    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# def get_respon(intents_list,intents_json):
def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    # tag= intents_list
    list_of_intents =intents_json['intents']
    # print(list_of_intents)
    result = 'hello'
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result 

# # while True:
#     message = input("| You: ")
#     if message == "bye" or message == "Goodbye":
#         ints = predict_class(message)
#         print(ints)
#         res = get_response(ints, intents)
#         print(res)
#         print("| Bot:", res)
#         print("|===================== The Program End here! =====================|")
#         exit()

#     else:
#         ints = predict_class(message)
#         print(ints)
#         res = get_response(ints, intents)
#         print("| Bot:", res)

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break
    
        ints = predict_class(sentence)
        print(ints)
        resp = get_response(ints,intents)
        # resp = get_response(sentence)
        print(resp)
        
