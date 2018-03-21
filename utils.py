import numpy as np


# in vocab 0 signifies no sys call (padding)
def load_train_data(train_path):
    train_file = open(train_path, 'r')
    sys_calls = []
    vocab = {'0': 0}
    i = 1
    for line in train_file.readlines():
        strip_line = line.strip().split(" ")
        for j, call in enumerate(strip_line):
            if call not in vocab:
                vocab[call] = i
                i += 1
            strip_line[j] = int(vocab[call])
        sys_calls.append(strip_line)
    train_file.close()
    return sys_calls, vocab


def load_label_data(labels_path):
    label_file = open(labels_path, 'r')
    labels = []
    i = 1
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


def get_batch_label_iterator(label_path, batch_size):
    labels = load_label_data(label_path)
    batch_len = int(np.ceil(len(labels) / float(batch_size)))
    labels_batch = np.empty(batch_len, dtype=object)
    for i in range(batch_len):
        labels_batch[i] = labels[i * batch_size: min((i + 1) * batch_size, len(labels))]
    del labels
    for label_batch in labels_batch:
        yield label_batch


def get_batch_data_iterator(train_path, seq_length, batch_size):
    sys_calls, vocab = load_train_data(train_path)
    batch_len = int(np.ceil(len(sys_calls) / float(batch_size)))
    sys_calls = pad_sys_calls(sys_calls, seq_length)
    sys_calls_batches = np.empty(batch_len, dtype=object)
    for i in range(batch_len):
        sys_calls_batches[i] = sys_calls[i * batch_size: min((i + 1) * batch_size, len(sys_calls))]

    for sys_calls_batch in sys_calls_batches:
        yield sys_calls_batch


def get_data_iterator(train_path, label_path, seq_length, batch_size):
    sys_calls_batch = get_batch_data_iterator(train_path, seq_length, batch_size)
    label_batch = get_batch_label_iterator(label_path, batch_size)
    return sys_calls_batch, label_batch
