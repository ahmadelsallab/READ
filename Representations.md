# Evolution of text representation in NLP

# NLP tasks
- Classification
Sentiment, Spam
- Sequence2Sequence (same as semantic segmentation)
MT, QA

# Text representation
Computer and Neural Nets can only understand digital forms.

Images are represented as pixel values, mapped in 2D spatial map, with each pixel value represent color intensity as that location in the 2D image space.

Speech signals are represented as the voltage activation of the microphone signal recording after being digitized with an ADC.

_What about text?_

Words are digitized as indices in vocabulary vectors.

Words as indices is not good since higher index means higher value, which is not true.


## One-Hot-Encoding (OHE)

Each index is mapped to a OHE vector

No similarity relation between words vectors
[0 0 0 1 0 0] T . [1 0 0 0 0 0 0] = 0

Also, its a very sparse vecor; vocab can reach 13M words.

## Bad of Words (BoW

More compact representation is to encode the vocab vector for each sentence, and mark the locations of the words of vocab appearing in that sentence as 1, while all other locations are 0.

It's still a sparse vector, but the sentence representation is more compact (one 13M vector for sentence/document instead of word).

However, the order of words is lost.

## Word vectors

Learn a dense representation of words, say a vector of 50 or 100 dimensions.

This vector is called an "Embedding" for reasons that will become clear later.

If we stack all the vectors in one table we have an "Embedding table", or a Look-up-table (LUT).

If we represent the word as OHE, then dot product of the word with the LUT gives the word vector.
In that sense, the LUT can be thought as weight matrix.

_How to find the values in the LUT?_

As we said the LUT can be thought as weights matrix.

We want to have vectors that encode similarity, in a way that synonyms are near to each others, same as in wordnet.

To do that, we come up with an artificial task that encode similarity. The most trivial task is the "validity" task as follows: stick every 2 neighboring words vecotrs together, and train a network to produce "1" for that pair, meaning that this pair can occur together. And then flip one of them with any random word, and train the net to produce "0" for the new pair, marking that as "invalid". 

Of course such trivial task by itself is not useful in anything, but after we finish, we have a LUT that encodes words that occur together in near vectors in the embedding space.

There are two famous other tasks (at least in word2vec):
- Skip-gram
- CBOW

The skip-gram is near to the validity task, we ask the net to predict the precedding and secceeding n words in a window, given a center word.
In some sense this encodes the context.

## ELMO
The word2vec provides the same vector regardless the context of the word. This prevents word sense disambiguation
ELMO provides contextualized vecotrs, learned by mixing the current word vecotr and its negithbors

ELMO can work at char level, which reduces the OOV rate.
 
# Document representation
We are not interested in words represntation, but in sequence of words; document or at least sentence represenations.

## Bag of words vectors
The BoW idea words for sequence of words in the same manner described above. 
A more advanced version is to average the words vectors of the sentence words:

S = 1/N(W1 + W2 + ....+WN)

Still order is missing

## RNN


For sequence learning problems, either for classification or sequence production (seq2seq), what we want to do is to summarize the sequence in one vector. It is also desirable to encode the order of words somehow.

RNN's can act as a state machine, keeping an internal hidden state that starts with some value with the start of the sequence. The state evolves every time a word of the sequence is fed to the RNN, in addition to its previous state (recurrence). In some sense, the state evolution is conditioned on the previous state, like in Bayes filters. After all the sequence is fed, the final state encodes and summarizes the whole sequence information in order.

RNN's remained the SoTA till 2017 and still widely used. The main disadv is their tendency to forget and diverge with longer sequences, which is handled with gated units like in LSTMs and GRU's and state reset.


## RAE


NLP community is familiar to "parse trees, which originate from linguistic origins, where a sentence is read/parsed according to a certain order that encodes it's syntactic struture.
In the same fashion, RAE's aim to compute a sentence representation by computing a representation of 2 vectors at a time. THe choice of which words vectors to parse is done according to a binary parse tree.
The tree could be a known syntactic tree obtained by linguistic rules, or it can be "guessed".
The way "guessing" happens is through an AE. The AE takes 2 vecors, encodes them, then reconstruct them again, scoring an error between the original and reconstructed vecrtors. This process is repeated for the whole sequence. The pair giving min reconstruction error is picked.
When a pair is parsed, they are replaced with their AE bottle neck representation. This make the tree shrinks with every parsing step.
This process is repeated in a tree fashion until only one vector remains, that represent the sequence.

Unlike RNN's, Recursive models do not keep internal state, but instead, the same computational block is re-called recursively on every new input.

## Convs2s
The recursive and recurrent models are sequential by nature; we need to wait the output of the previous step to start the next. Such sequential nature is not suitable for parallelization, which prevents us from using GPU's. 
ConvS2S models aims at using conv1d kernels to summarize the sequence. Conv kernel has a "FoV". the FoV of deeper feature maps neurons span larger part of the input. This provides a good hierarichal representation, with the depther controls the span of the representation.
This technique is way faster than recurrent or recursive models.

However, since the sequence length requires deeper models, it makes it difficult to learn dependencies between distant words (see https://arxiv.org/pdf/1706.03762.pdf)

## Attention
A trending techniue is to add gates that "Selects" to encode or discard vectors in the final representation.
Like in the BoW vectors, where the vectors are averaged, we can think of a weighted average.
The weights are "attention" gates. This is called "attention mechanism".
The weights can be "soft" or "hard" attention.
In "hard" attention, we keep the max weight vector
In "soft" attention, we use a "softmax" over the weights.

V = softmax(W's).V's



The two main techniques of attention are Bahddanau attention https://arxiv.org/abs/1409.0473 or Luong attention https://arxiv.org/pdf/1508.04025.pdf. While the first acts on the hiddent LSTM states, the latter works on the LSTM outputs.

There are few ways to learn the weights https://arxiv.org/pdf/1508.04025.pdf:
- Dot: W_i = W_i.W's: the higher the simialiry of word vectors, the higher the weight
- General: W_i.W_learnable.W's: learn the weights
- Concat: W_learnable.[W_i;W_j]

Like in BoW, the order is lost. That's why attention is mixed with recurrence as in https://arxiv.org/abs/1409.0473.

Another way is to use positional encoding (Trans and Convs2s); either learnable or constant (Trans=sine/cosine): http://jalammar.github.io/illustrated-transformer/

In Bahdanau, seq2seq is done by summarizing the input sequence in one vector coming out of the encoder. The decoder task is to produce unaralled tokens, conditioned on the sumamrized encoder state, and the previous decoded token. Attention comes into play to increase dependency not only on the last state of the encoder, but also the previous ones, in addition to the decoded states, not only the previous one.

## HATT
Text is a sequence of words, but this sequence has a hierarichal structure beyond only a flat sequence. We can have document, structured into paragraphs, then into sentences.
In HATT, similar idea to Bahddanau attention is used, but first the sequence is tokenized into sentences, then words.
High score on IMDB

## Transformer
http://jalammar.github.io/illustrated-transformer/
To get rid of recurrence, the transformer fully relies on attention gates. The attention weights are calculated using dot, in parallel for all sequence words, producing vectors for all the words in parallel.
The process is repeated multiple steps as needed (layers).
The final representation is a number of refined vectors. 
To summarize into one vectors, we can add 1 attention weight gates at the end. In BERT a special symbol is introduced for that.

The decoder also depends on attention, and condition on the encoder states in addition to "only" the previous decoded tokens, same as in seq2seq with RNN.

## Universal transformer
<TBD>

## ULMFiT
An intuitive, but new trend in NLP is transfer learning. ULMFiT started this wave by training an encoder for language modeling on wiki text, then fine tune by adding classification or decoder layers for the target tasks.
The idea is well established almost 5 years earlier in the CV community, where networks like VGG, ResNet,...etc are widely used after being trained on imagenet. Those are called backbones.

## GPT
The GPT mixes the idea of transfer learning with Transformer



## BERT
BERT is based on the Trans and GPT, with the introduction of Bi-dir LM, with two tasks: MLM and Next-sentence prediction
BERT is said to be the imagenet moment
http://jalammar.github.io/illustrated-bert/
