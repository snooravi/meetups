import re
from string import ascii_lowercase
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import choice

con = open('war_and_peace.txt', 'r')       # load data
train = con.read()

train = train.lower()                      # make lowercase
train = re.sub(r'\s+', ' ', train)         # get rid of extra whitespace and newlinesx1

type(train)
train[5000:5300]                           # example

letters = [i for i in ascii_lowercase]
letters.append('_')                        # _ presents any non-letter character

print(re.search('_', train))                # make sure '_' not in the training data

trans_matrix = pd.DataFrame(np.zeros((27, 27), dtype=int), index=letters, columns=letters)
trans_matrix


for i in range(len(train) - 1):

    if i % 100000 == 0:
        print('processed first %i characters out of 3210812' % i)

    if train[i] in letters:
        current_letter = train[i]                  # the preceding letter
    else:
        current_letter = '_'                       # if not letter, assign non-letter character # if not letter, assign non-letter character

    if train[i + 1] in letters:
        last_letter = train[i + 1]                  # the succeeding letter
    else:
        last_letter = '_'                           # if not letter, assign non-letter character # if not letter, assign non-letter character

    trans_matrix.loc[current_letter, last_letter] = trans_matrix.loc[current_letter, last_letter] + 1


trans_matrix = trans_matrix + 1             # allowing any combination of letters

pickle.dump(trans_matrix, open('trans_matrix.p', 'wb'))    # save for future use

trans_matrix = pickle.load(open('trans_matrix.p', 'rb'))   # load the pickled file

for i in range(trans_matrix.shape[0]):
    trans_matrix.iloc[i, :] = trans_matrix.iloc[i, :] / trans_matrix.iloc[i, :].sum()

sns.heatmap(trans_matrix)
plt.yticks(rotation=0)
plt.savefig('heatmap.png')
print('created the heatmap')


def decode(mapping, coded):                             # function to decode text

    text = []

    for i in coded:
        if i in letters:
            text.append(letters[mapping.index(i)])
        else:
            text.append(letters[mapping.index('_')])

    return ''.join(text)


def encode(mapping, text):

    encoded = []

    for i in text:
        if i in letters:
            encoded.append(mapping[letters.index(i)])
        else:
            encoded.append(mapping[letters.index('_')])

    return ''.join(encoded)


def loglike(decoded):

    loglike = 0
    last_letter = '_'

    for i in decoded:
        current_letter = i
        loglike = loglike + np.log(trans_matrix.loc[last_letter, current_letter])
        last_letter = current_letter

    loglike = loglike + np.log(trans_matrix.loc[last_letter, '_'])

    return loglike


# np.random.seed(12345)
mapping_true = list(choice(letters, 27, replace=False))              # randomly scrample to create coding(mapping)
correct_txt = ('Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation.'
               ' This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames'
               ' of withering injustice. It came as a joyous daybreak to end the long night of their captivity').lower()
encoded = encode(mapping_true, correct_txt)


mapping0 = list(choice(letters, 27, replace=False))      # set the initial mapping
current_loglik = loglike(decode(mapping0, encoded))
max_loglik = current_loglik                              # best loglikehood so far
max_decode = decode(mapping0, encoded)                   # best decoded text so far

while i <= 5000:                                        # Metrapolis Begins Here

    tmp = choice(letters, size=2, replace=False)

    pos1, pos2 = mapping0.index(tmp[0]), mapping0.index(tmp[1])  # randomly swap 2 letters

    mapping_proposed = mapping0[:]                              # copy a list
    mapping_proposed[pos1], mapping_proposed[pos2] = mapping_proposed[pos2], mapping_proposed[pos1]

    proposed_loglik = loglike(decode(mapping_proposed, encoded))   # loglikehood of proposed mapping

    if (np.random.uniform() < np.exp(proposed_loglik - current_loglik)):
        mapping0 = mapping_proposed
        current_loglik = proposed_loglik

        if current_loglik > max_loglik:
            max_loglik = current_loglik
            max_decode = decode(mapping_proposed, encoded)

            i += 1

        print(i, decode(mapping0, encoded), '\n')
