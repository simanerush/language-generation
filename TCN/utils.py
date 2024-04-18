import re
import string
import torch
from torch.utils.data import Dataset


vocab = string.ascii_lowercase + ' .'
vocab_size = len(vocab)

class SpeechDataset(Dataset):
    """
    Creates a dataset of strings from a text file.
    All strings will be of length max_len and padded with '.' if needed.

    By default this dataset will return a string, this string is not directly readable by pytorch.
    Use transform (e.g. one_hot) to convert the string into a Tensor.
    """

    def __init__(self, dataset_path, transform=None, max_len=250):
        with open(dataset_path) as file:
            st = file.read()
        st = st.lower()
        reg = re.compile('[^%s]' % vocab)
        period = re.compile(r'[ .]*\.[ .]*')
        space = re.compile(r' +')
        sentence = re.compile(r'[^.]*\.')
        self.data = space.sub(' ',period.sub('.',reg.sub('', st)))
        if max_len is None:
            self.range = [(m.start(), m.end()) for m in sentence.finditer(self.data)]
        else:
            self.range = [(m.start(), m.start()+max_len) for m in sentence.finditer(self.data)]
            self.data += self.data[:max_len]
        if transform is not None:
            self.data = transform(self.data)

    def __len__(self):
        return len(self.range)

    def __getitem__(self, idx):
        s, e = self.range[idx]
        if isinstance(self.data, str):
            return self.data[s:e]
        return self.data[:, s:e]

class LanguageModel:
    """
    An abstract class representing a language model.
    """
    def predict_all(self, some_text):
        """
        Given some_text, predict the likelihoods of the next character for each substring from 0..i
        The resulting tensor is one element longer than the input, as it contains probabilities for all sub-strings
        including the first empty string (probability of the first character)

        :param some_text: A string containing characters in vocab, may be an empty string!
        :return: torch.Tensor((len(vocab), len(some_text)+1)) of log-probabilities
        """
        raise NotImplementedError('Abstract function LanguageModel.predict_all')

    def predict_next(self, some_text):
        """
        Given some_text, predict the likelihood of the next character

        :param some_text: A string containing characters in vocab, may be an empty string!
        :return: a Tensor (len(vocab)) of log-probabilities
        """
        return self.predict_all(some_text)[:, -1]

class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    """
    Use beam search to find the highest likelihood generations.

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """

    top_sequences = TopNHeap(n_results)

    # Sequence, log_likelihood, length
    beam = [("", 0.0, 0)]

    for _ in range(max_length):
        new_beam = []
        for seq, log_likelihood, length in beam:
            if length < max_length and not seq.endswith('.'):
                log_probs = model.predict_next(seq)

                for idx, log_prob in enumerate(log_probs):
                    next_char = vocab[idx]
                    new_seq = seq + next_char
                    new_log_likelihood = log_likelihood + log_prob.item()
                    new_length = length + 1
                    if next_char == '.' or new_length >= max_length:
                        score = (new_log_likelihood / new_length) if (average_log_likelihood and new_length) > 0 else new_log_likelihood
                        top_sequences.add((score, new_seq))
                    else:
                        new_beam.append((new_seq, new_log_likelihood, new_length))
                    
            else:
                score = (log_likelihood / length) if (average_log_likelihood and length) > 0 else log_likelihood
                top_sequences.add((score, seq))

        new_beam.sort(key=lambda x: (x[1]/x[2]) if (average_log_likelihood and x[2] > 0) else x[1], reverse=True)
        beam = new_beam[:beam_size]

    final_results = sorted(top_sequences.elements, key=lambda x: x[0], reverse=True)[:n_results]

    return ' '.join([seq for _, seq in final_results])


def one_hot(s: str):
    """
    Converts a string into a one-hot encoding
    :param s: a string with characters in vocab (all other characters will be ignored!)
    :return: a once hot encoding Tensor r (len(vocab), len(s)), with r[j, i] = (s[i] == vocab[j])
    """
    import numpy as np
    if len(s) == 0:
        return torch.zeros((len(vocab), 0))
    return torch.as_tensor(np.array(list(s.lower()))[None, :] == np.array(list(vocab))[:, None]).float()
