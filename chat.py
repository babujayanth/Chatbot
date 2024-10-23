import random
import json

import torch

from nltk.corpus import stopwords

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)


input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
print(all_words)
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    print('a:',sentence)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in sentence if word not in stop_words]
    print('b:',filtered_tokens)
    # X = bag_of_words(sentence, all_words)
    X = bag_of_words( filtered_tokens, all_words)
    print('c:',X)
    X = X.reshape(1, X.shape[0])
    # print(X)
    X = torch.from_numpy(X).to(device)
    print(X)

    output = model(X)
    # print(output)
    _, predicted = torch.max(output, dim=1)
    print(predicted)

    
    
    tag = tags[predicted.item()]
    print(tag)

    probs = torch.softmax(output, dim=1)
    # print(probs)
    prob = probs[0][predicted.item()]
    print(prob)
    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I am Sorry .I do not understand your question."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

