from collections import Counter, defaultdict
from machine_learning import split_data

import math, random, re, glob
import numpy as np
import pandas as pd

##  --------------------------------------------------------
##  Adicionado nltk para Stemming da palavra
##  --------------------------------------------------------
import nltk
nltk.download()
##  d
##  punkt
##  words

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import words

ps = PorterStemmer()

def tokenize(message):
    message = message.lower() # convert to lowercase
    all_words = re.findall("[a-z0-9']+", message) # extract the words
    return set(all_words) # remove duplicates

def count_words(training_set):
    """training set consists of pairs (message, is_spam)"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam)"""
    return [(w,
        (spam + k) / (total_spams + 2 * k),
        (non_spam + k) / (total_non_spams + 2 * k))
            for w, (spam, non_spam) in counts.items()]


def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for word, prob_if_spam, prob_if_not_spam in word_probs:

        # for each word in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # for each word that's not in the message
        # add the log probability of _not_ seeing it
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    result = 0 
##  --------------------------------------------------------
##  probleminha da divisao por 0, entao um try para evita-lo
##  -------------------------------------------------------- 
    try:
       result = prob_if_spam / (prob_if_spam + prob_if_not_spam)
    except:
       print('spam_probability',message, prob_if_spam , prob_if_spam , prob_if_not_spam)
    
    return result


class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):

        # count spam and non-spam messages
        num_spams = len([is_spam for message, is_spam in training_set if is_spam])
        num_non_spams = len(training_set) - num_spams

        # run training data through our "pipeline"
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)

##  --------------------------------------------------------
##  funcao para limpar um pouco do lixo html que vem no email
##  é uma piada mas removeu 90% do lixo
##  --------------------------------------------------------
def is_html(line):
    line.lower()
    batRegex = re.compile(r'(<[^>]+>|^\s|@|^=|^#|^http:|style=|^__|^<|^colspan|^valign|^background|^%|^face=|&nbsp;|size=|^table|^meta|^http|^x-|^--|^errors-|th=|^content-|^mime-|^sender|^charset|^date:)')
    tag = batRegex.search(line)
    #print(line, tag)
    return tag

##  --------------------------------------------------------
##  funcao para fazer o stemming das palavras e aplicar filtro extra
##  --------------------------------------------------------
def filters(line):   
    line.lower()
    arr_words = word_tokenize(line)
    
    for word in arr_words :
        if len(word) <= 1 or word == '':
            word = ps.stem(word)
            arr_words.remove(word)

    clear = ','.join(arr_words)
    clear = re.sub(',', ' ', clear)

    return clear   


def get_subject_data(path):

    data = []

    # regex for stripping out the leading "Subject:" and any spaces after it
    subject_regex = re.compile(r"^Subject:\s+")

    # glob.glob returns every filename that matches the wildcarded path
    for fn in glob.glob(path):
        is_spam = "ham" not in fn

        with open(fn,'r',encoding='ISO-8859-1') as file:
            start = 0
            for line in file:   
##  --------------------------------------------------------
##  diversas alteracoes para realizar a leitura do contúdo da 
##  mensagem após o titulo
##  -------------------------------------------------------- 
                if line.startswith("Subject:"):
                    subject = subject_regex.sub("", line).strip()
                    subject = filters(subject)
                    data.append((subject, is_spam))                    
                    start = 1
                elif line.startswith(" ") and start == 1:
                    start = 2
                elif start == 2  and is_html(line) == None:
                    subject = filters(line)
                    data.append((subject, is_spam))                    

    return data

def p_spam_given_word(word_prob):
    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

def train_and_test_model(path):

    data = get_subject_data(path)
    
    random.seed(0)      # just so you get the same answers as me
    train_data, test_data = split_data(data, 0.75)

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    classified = [(subject, is_spam, classifier.classify(subject))
              for subject, is_spam in test_data]

    counts = Counter((is_spam, spam_probability > 0.5) # (actual, predicted)
                     for _, is_spam, spam_probability in classified)

    print() 
    print(counts)
    print()
    
    classified.sort(key=lambda row: row[2])
    spammiest_hams = list(filter(lambda row: not row[1], classified))[-5:]
    hammiest_spams = list(filter(lambda row: row[1], classified))[:5]

    print("spammiest_hams", spammiest_hams)
    print()
    print("hammiest_spams", hammiest_spams)
    print()
    
    words = sorted(classifier.word_probs, key=p_spam_given_word)

    spammiest_words = words[-5:]
    hammiest_words = words[:5]

    print("spammiest_words", spammiest_words)
    print()    
    print("hammiest_words", hammiest_words)
    print()

if __name__ == "__main__":
    train_and_test_model(r"./emails/*/*")
    #train_and_test_model(r"./test/*/*")