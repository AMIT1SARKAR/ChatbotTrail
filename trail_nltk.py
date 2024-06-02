import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)



def stem(word):
    return stemmer.stem(word.lower())



def bag_of_words(tokenize_sentence,all_words):
    
    tokenize_sentence = [stem(w) for w in tokenize_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx , w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx] = 1.0
    
    return bag

'''sentence = [ "hello", "how ", "are", "you"]
words = ["hi","hello","i","you","bye"]
b= bag_of_words(sentence , words)
print(b)'''





'''
#tokenizition test
a = "How long does shipping takes?"
print("\n",a,"\n")
b=tokenize(a)
print(b,"\n")

#steming test

words=["Organize","organizes","organizing"]
print("\n",words)
stemmed_word = [stem(i) for i in words]
print("\n",stemmed_word,"\n")'''