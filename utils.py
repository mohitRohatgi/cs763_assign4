import numpy as np
import pickle as pickle
import os
import statistics

vocab_path = os.path.join(os.getcwd(), 'vocab.pkl')


# 0 signifies no sys call (padding), <UNK> signifies unknown sys call.
def load_data(train_path, mode='train', saved=False):
    data_file = open(train_path, 'r')

    i = 2
    if mode != 'train' or saved:
        with open(vocab_path, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
    else:
        vocab = {'0': 0, '<UNK>': 1}

    sys_calls = []
    for line in data_file.readlines():
        strip_line = line.strip().split(" ")
        for j, call in enumerate(strip_line):
            if call not in vocab:
                if mode == 'train' and not saved:
                    vocab[call] = i
                    i += 1
                    strip_line[j] = int(vocab[call])
                else:
                    strip_line[j] = int(vocab['<UNK>'])
            else:
                strip_line[j] = int(vocab[call])
        sys_calls.append(strip_line)

    if mode == 'train':
        with open(vocab_path, 'wb') as vocab_file:
            pickle.dump(vocab, vocab_file, pickle.HIGHEST_PROTOCOL)

    data_file.close()
    return sys_calls


def load_label_data(labels_path):
    label_file = open(labels_path, 'r')
    labels = []
    for line in label_file.readlines():
        labels.append(int(line.strip()))
    return np.array(labels)


def pad_seq(sys_call, seq_length):
    while len(sys_call) < seq_length:
        sys_call.append(0)
    return sys_call


def pad_sys_calls(sys_calls, seq_length):
    for i, sys_call in enumerate(sys_calls):
        sys_calls[i] = pad_seq(sys_call, seq_length)
    return np.array(sys_calls)


def batch_iter(n_epoch, data, batch_size, shuffled=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for _ in range(n_epoch):
        if shuffled:
            shuffled_data = data[np.random.permutation(np.arange(len(data)))]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(data))
            if end_index - start_index < batch_size:
                start_index = end_index - batch_size
            batch_data = shuffled_data[start_index:end_index]
            yield batch_data


def get_freq(data, bin_size=50):
    freqs = {}
    for sys_call in data:
        freq = int(np.ceil(len(sys_call) / float(bin_size)))
        if freq not in freqs:
            freqs[freq] = 0
        freqs[freq] += 1
    median = int(statistics.median(freqs.values()))
    for k, v in freqs.items():
        if v == median:
            return k, freqs


def preprocess(data, median_freq):
    pre_processed_data = []
    for sys_call in data:
        start = 0
        while start < len(sys_call) - median_freq:
            pre_processed_data.append(sys_call[start: min(start + median_freq, len(sys_call))])
            start += median_freq
    return pre_processed_data


def get_batch_data_iterator(n_epoch, data_path, seq_length, batch_size, label_path=None, mode='train', saved=False):
    data = load_data(train_path=data_path, mode=mode, saved=saved)
    # freq = get_freq(data)
    data = np.array(pad_sys_calls(data, seq_length))

    if mode != 'train':
        return batch_iter(n_epoch, data, batch_size, shuffled=False), len(data)

    indices = np.random.permutation(np.arange(len(data)))
    shuffled_data = data[indices]
    boundary = int(len(data) * 0.8)
    train_labels, valid_labels = None, None

    if label_path is not None:
        labels = load_label_data(label_path)
        shuffled_label = labels[indices]
        train_labels, valid_labels = shuffled_label[:boundary], shuffled_label[boundary:]

    train_data, valid_data = shuffled_data[:boundary], shuffled_data[boundary:]
    train_iterator = batch_iter(n_epoch, train_data, batch_size)
    valid_iterator = batch_iter(n_epoch, valid_data, batch_size)
    train_label_iter = batch_iter(n_epoch, train_labels, batch_size)
    valid_label_iter = batch_iter(n_epoch, valid_labels, batch_size)
    return train_iterator, train_label_iter, valid_iterator, valid_label_iter
