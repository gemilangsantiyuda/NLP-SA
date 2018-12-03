import nltk
from nltk.corpus import brown
from collections import Counter
brown_words = brown.words()

unigram_counts = Counter(brown_words)
bigrams = []
for sent in brown.sents():
    bigrams.extend(nltk.trigrams(sent, pad_left=True, pad_right=True))
bigram_counts = Counter(bigrams)

trigrams = []
for sent in brown.sents():
    trigrams.extend(nltk.trigrams(sent, pad_left=True, pad_right=True))
trigram_counts = Counter(trigrams)

def bigram_LM(sentence_x, smoothing=0.0):
    unique_words = len(unigram_counts.keys()) + 2
    x_bigrams = nltk.bigrams(sentence_x, pad_left=True, pad_right=True)
    prob_x = 1.0
    for bg in x_bigrams:
        if bg[0] == None:
            prob_bg = (bigram_counts[bg]+smoothing)/(len(brown.sents())+smoothing*unique_words)
        else :
            prob_bg = (bigram_counts[bg]+smoothing)/(unigram_counts[bg[0]]+smoothing*unique_words)
        prob_x = prob_x * prob_bg
        print(str(bg)+":"+str(prob_bg))
    return prob_x

sentence = ["i","want","to","go","home"]
print(bigram_LM(sentence,1))
