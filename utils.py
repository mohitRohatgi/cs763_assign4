import numpy as np


# in vocab 0 signifies no sys call (padding)
def load_data(train_path):
    data_file = open(train_path, 'r')
    sys_calls = []
    vocab = {'0': 0}
    i = 1
    for line in data_file.readlines():
        strip_line = line.strip().split(" ")
        for j, call in enumerate(strip_line):
            if call not in vocab:
                vocab[call] = i
                i += 1
            strip_line[j] = int(vocab[call])
        sys_calls.append(strip_line)
    data_file.close()
    return sys_calls, vocab


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


def get_batch_label_iterator(n_epoch, label_path, batch_size):
    labels = load_label_data(label_path)
    batch_len = int(np.ceil(len(labels) / float(batch_size)))
    num_batches_per_epoch = int((len(labels) - 1) / batch_size) + 1
    for _ in range(n_epoch):
        shuffled_labels = labels[np.random.permutation(np.arange(len(labels)))]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(labels))
            batch_data = shuffled_labels[start_index:end_index]
            yield batch_data


def get_batch_data_iterator(n_epoch, data_path, seq_length, batch_size):
    data, vocab = load_data(data_path)
    data = np.array(pad_sys_calls(data, seq_length))
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for _ in range(n_epoch):
        shuffled_data = data[np.random.permutation(np.arange(len(data)))]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(data))
            batch_data = shuffled_data[start_index:end_index]
            yield batch_data


def get_data_iterator(n_epoch, data_path, label_path, seq_length, batch_size):
    sys_calls_batch = get_batch_data_iterator(n_epoch, data_path, seq_length, batch_size)
    label_batch = get_batch_label_iterator(n_epoch, label_path, batch_size)
    return sys_calls_batch, label_batch
