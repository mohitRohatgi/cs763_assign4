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
    trim_sys_call = []

    if len(sys_call) > seq_length:
        for i in range(seq_length):
            trim_sys_call.append(sys_call[i])
        return trim_sys_call

    while len(sys_call) < seq_length:
        sys_call.append(0)
    return sys_call


def pad_sys_calls(sys_calls, max_train_length, bins):
    padding_length = min(bins, key=lambda x:abs(x-max_train_length))
    for i, sys_call in enumerate(sys_calls):
        sys_calls[i] = pad_seq(sys_call, bins[-1])
    return np.array(sys_calls), padding_length


def train_valid_split(data, labels, split_ratio):
    num_valid = int(np.ceil(split_ratio * len(data)))
    valid_indices = np.random.randint(0, len(data), num_valid)
    valid_data = []
    valid_label = []
    train_data = []
    train_label = []
    for i in range(len(data)):
        if i in valid_indices:
            valid_data.append(data[i])
            valid_label.append(labels[i])
        else:
            train_label.append(labels[i])
            train_data.append(data[i])
    return train_data, valid_data, train_label, valid_label


def get_batch_data(data, label, batch_size):
    assert len(data) == len(label)
    indices = np.random.randint(0, len(data), batch_size)
    data_batch = []
    label_batch = []
    max_len = 0
    for index in indices:
        if max_len < len(data[index]):
            max_len = len(data[index])
        data_batch.append(data[index])
        label_batch.append(label[index])
    return data_batch, label_batch, max_len


def get_batch_data_iterator(n_epoch, data_path, seq_length, batch_size, bins,
                            label_path=None, mode='train', saved=False):
    data = load_data(train_path=data_path, mode=mode, saved=saved)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    if label_path is not None:
        labels = load_label_data(label_path)
        for i in range(n_epoch):
            train_data, valid_data, train_label, valid_label = train_valid_split(data, labels, 0.8)
            for batch_num in range(num_batches_per_epoch):
                train_data_batch, train_label_batch, max_train_length = get_batch_data(train_data, train_label, batch_size)
                valid_data_batch, valid_label_batch, max_valid_length = get_batch_data(valid_data, valid_label, batch_size)
                train_data_batch_pad = pad_sys_calls(train_data_batch, max_train_length, bins)
                valid_data_batch_pad = pad_sys_calls(valid_data_batch, max_train_length, bins)
                yield train_data_batch_pad, np.array(train_label_batch), valid_data_batch_pad, np.array(valid_label_batch)
    else:
        for i in range(len(data)):
            yield pad_sys_calls([data[i]], len(data[i]), bins)
