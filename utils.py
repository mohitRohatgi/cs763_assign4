import numpy as np


# in vocab 0 signifies no sys call (padding)
def load_train_data(train_path):
    train_file = open(train_path, 'r')
    sys_calls = []
    vocab = {'0': 0}
    i = 1
    for line in train_file.readlines():
        strip_line = line.strip().split(" ")
        for call in strip_line:
            if int(call) not in vocab:
                vocab[int(call)] = i
                i += 1
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


def get_one_hot(call, vocab: dict):
    x = np.zeros(len(vocab))
    x[vocab[int(call)]] = 1
    return x


def convert_to_one_hot(sys_calls, seq_length, vocab: dict):
    one_hot_sys_calls = []
    for sys_call in sys_calls:
        one_hot_sys_call = np.zeros((seq_length, len(vocab)))
        for i, call in enumerate(sys_call):
            one_hot_sys_call[i] = get_one_hot(call, vocab)
        one_hot_sys_calls.append(one_hot_sys_call)
    return np.array(one_hot_sys_calls)


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
    one_hot_sys_calls = convert_to_one_hot(sys_calls, seq_length, vocab)
    del sys_calls
    batch_len = int(np.ceil(len(one_hot_sys_calls) / float(batch_size)))
    sys_calls_batches = np.empty(batch_len, dtype=object)
    for i in range(batch_len):
        sys_calls_batches[i] = one_hot_sys_calls[i * batch_size: min((i + 1) * batch_size, len(one_hot_sys_calls))]
    del one_hot_sys_calls

    for sys_calls_batch in sys_calls_batches:
        yield sys_calls_batch


def get_data_iterator(train_path, label_path, seq_length, batch_size):
    sys_calls_batch = get_batch_data_iterator(train_path, seq_length, batch_size)
    label_batch = get_batch_label_iterator(label_path, batch_size)
    return sys_calls_batch, label_batch
