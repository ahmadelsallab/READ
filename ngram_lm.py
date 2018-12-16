from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import random

first_sentence = reuters.sents()[0]
print(first_sentence)  # [u'ASIAN', u'EXPORTERS', u'FEAR', u'DAMAGE', u'FROM' ...

# Get the bigrams
print(list(bigrams(first_sentence)))  # [(u'ASIAN', u'EXPORTERS'), (u'EXPORTERS', u'FEAR'), (u'FEAR', u'DAMAGE'), (u'DAMAGE', u'FROM'), ...

# Get the padded bigrams
print(list(bigrams(first_sentence, pad_left=True, pad_right=True)))  # [(None, u'ASIAN'), (u'ASIAN', u'EXPORTERS'), (u'EXPORTERS', u'FEAR'), (u'FEAR', u'DAMAGE'), (u'DAMAGE', u'FROM'),

# Get the trigrams
print(list(trigrams(first_sentence)))  # [(u'ASIAN', u'EXPORTERS', u'FEAR'), (u'EXPORTERS', u'FEAR', u'DAMAGE'), (u'FEAR', u'DAMAGE', u'FROM'), ...

# Get the padded trigrams
print(list(trigrams(first_sentence, pad_left=True, pad_right=True)))  # [(None, None, u'ASIAN'), (None, u'ASIAN', u'EXPORTERS'), (u'ASIAN', u'EXPORTERS', u'FEAR'), (u'EXPORTERS', u'FEAR', u'DAMAGE'), (u'FEAR', u'DAMAGE', u'FROM') ...

def build_trigram_lm(corpus):
    model = defaultdict(lambda: defaultdict(lambda: 0))

    for sentence in corpus:
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1

    '''
    print(model["what", "the"]["economists"])  # "economists" follows "what the" 2 times
    print(model["what", "the"]["nonexistingword"])  # 0 times
    print(model[None, None]["The"])  # 8839 sentences start with "The"
    '''
    # Let's transform the counts to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
    return model

model = build_trigram_lm(reuters.sents())
print(model["what", "the"]["economists"])  # 0.0434782608696
print(model["what", "the"]["nonexistingword"])  # 0.0
print(model[None, None]["The"])  # 0.161543241465

def sample_lm(model):
    text = [None, None]
    prob = 1.0  # <- Init probability

    sentence_finished = False

    while not sentence_finished:
        r = random.random()
        accumulator = .0

        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]

            if accumulator >= r:
                prob *= model[tuple(text[-2:])][word]  # <- Update the probability with the conditional probability of the new word
                text.append(word)
                break

        if text[-2:] == [None, None]:
            sentence_finished = True

    return text, prob

'''    
    text = [None, None]

    sentence_finished = False

    while not sentence_finished:
        r = random.random()
        accumulator = .0

        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]

            if accumulator >= r:
                text.append(word)
                break

        if text[-2:] == [None, None]:
            sentence_finished = True
'''


text = sample_lm(model)
print(' '.join([t for t in text if t]))