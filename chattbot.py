import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle

stemmer = LancasterStemmer() # takes only the words(no question marks or commas etc)

with open("intents.json") as file:
    data = json.load(file)
#print(data) # data["intents"'] see json file

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f) # rb -> read bytes
    
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern) # A sentence or data can be split into words -> tokenize
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words))) # set -> no duplicates

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]  # create an empty list, same size as labels. Later gets modified on output_row

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1) # Doing one hot encoding
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f) # wb -> write byte

tensorflow.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net) # choosing DNN model

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Now bag the user's words
def bag_of_words(sentc, words):

    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(sentc)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Chatbot is ready!!(type exit to stop)")

    while True:
        inp = input("You:")
        if inp.lower() == "exit":
            break

        result = model.predict([bag_of_words(inp, words)]) # result basically shows some no., the greatest no. means more chance from that class[labels = class1, class2,.... diff tags]
        result_index = np.argmax(result)
        tag = labels[result_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()




















 
