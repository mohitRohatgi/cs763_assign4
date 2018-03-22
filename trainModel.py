import tensorflow as tf
from src.model import Model
from utils import get_batch_data_iterator
from config import Config

# TODO: make directory model_name and save model and loss(train and valid) data in the folder.
# TODO: save best model data(graph details).
# TODO: save models.
# TODO: fit the data.


def main():
    # model_name = sys.argv[sys.argv.index('-modelName') + 1]
    # train_path = sys.argv[sys.argv.index('-data') + 1]
    # labels_path = sys.argv[sys.argv.index('-target') + 1]
    model_name = 'model_name'
    train_path = '/Users/mohitrohatgi/Downloads/assign4/train_data.txt'
    labels_path = '/Users/mohitrohatgi/Downloads/assign4/train_labels.txt'
    config = Config()
    train_data, train_label, valid_data, valid_label = get_batch_data_iterator(config.n_epoch, train_path, labels_path,
                                                                               config.seq_length, config.batch_size)

    model = Model(config.n_layers, config.hidden_dim, config.vocab_size, config.embed_size)
    with tf.Session().as_default() as sess:
        step = 0
        sess.run(tf.global_variables_initializer())
        for train_batch_data, train_label_batch, valid_batch_data, valid_batch_label in zip(train_data, train_label,
                                                                                            valid_data, valid_label):
            train_loss, train_accuracy, train_prediction = model.run_batch(sess, train_batch_data, train_label_batch)
            print("train_step = ", step, " loss = ", train_loss, " accuracy = ", train_accuracy)
            step += 1
            if step % config.evaluate_every == 0:
                model.isTrain = False
                valid_loss, valid_accuracy, valid_prediction = model.run_batch(sess, valid_batch_data, valid_batch_label)
                print("valid_loss = ", valid_loss, " accuracy = ", valid_accuracy)
                model.isTrain = True

    print(model_name, train_path, labels_path)


if __name__ == '__main__':
    main()
