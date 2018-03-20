import tensorflow as tf
from src.model import Model
from src.criterion import Criterion
from utils import get_data_iterator
from config import Config

# TODO: add dropouts in the model.
# TODO: add valid step.
# TODO: make directory model_name and save model and loss in the folder.


def main():
    # model_name = sys.argv[sys.argv.index('-modelName') + 1]
    # train_path = sys.argv[sys.argv.index('-data') + 1]
    # labels_path = sys.argv[sys.argv.index('-target') + 1]
    model_name = 'model_name'
    train_path = '/Users/mohitrohatgi/Downloads/assign4/train_data.txt'
    labels_path = '/Users/mohitrohatgi/Downloads/assign4/train_labels.txt'
    config = Config()
    train_data, label_data = get_data_iterator(train_path, labels_path, config.seq_length, config.batch_size)

    criterion = Criterion()
    model = Model(config.n_layers, config.hidden_dim, config.vocab_size, config.input_dimension)
    with tf.Session().as_default() as sess:
        step = 0
        for train_batch_data, label_batch_data in zip(train_data, label_data):
            loss, accuracy, prediction = model.run_batch(sess, criterion, train_batch_data, label_batch_data)
            print("step = ", step, " loss = ", loss, " accuracy = ", accuracy)
            step += 1

    print(model_name, train_path, labels_path)


if __name__ == '__main__':
    main()
