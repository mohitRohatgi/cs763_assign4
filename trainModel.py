import tensorflow as tf

from history_logger import HistoryLogger
from src.model import Model
from utils import get_batch_data_iterator
from config import Config

import os
import time


def main():
    # model_name = sys.argv[sys.argv.index('-modelName') + 1]
    # train_path = sys.argv[sys.argv.index('-data') + 1]
    # labels_path = sys.argv[sys.argv.index('-target') + 1]

    start = time.time()
    model_name = os.path.join(os.getcwd(), 'model_name')
    train_path = os.path.join(os.getcwd(), 'data/train_data.txt')
    labels_path = os.path.join(os.getcwd(), 'data/train_labels.txt')
    save_meta_graph = True

    model_no = int(time.time())
    model_name = os.path.join(model_name, str(model_no))
    if model_name[-1] != '/':
        model_name += '/'
    config = Config()
    train_data, train_label, valid_data, valid_label = get_batch_data_iterator(
        n_epoch=config.n_epoch, data_path=train_path, label_path=labels_path, seq_length=config.seq_length,
        batch_size=config.batch_size, mode='train', saved=False)
    model = Model(config.n_layers, config.hidden_dim, config.vocab_size, config.embed_size)

    with tf.Session().as_default() as sess:
        step = 0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10000)

        if not os.path.exists(model_name):
            os.makedirs(model_name)

        logger_path = os.path.join(model_name, str(model_no))
        logger = HistoryLogger(config)
        for train_batch_data, train_label_batch in zip(train_data, train_label):
            train_loss, train_accuracy, train_prediction = model.run_batch(sess, train_batch_data, train_label_batch)
            print("train_step = ", step, " loss = ", train_loss, " accuracy = ", train_accuracy, " dir = ", model_name)
            step += 1
            if step % config.evaluate_every == 0:
                model.isTrain = False
                valid_batch_data, valid_batch_label = valid_data.__next__(), valid_label.__next__()
                valid_loss, valid_accuracy, valid_prediction = model.run_batch(sess, valid_batch_data, valid_batch_label)
                logger.add(train_loss, train_accuracy, valid_loss, valid_accuracy, step)
                print("valid_loss = ", valid_loss, " accuracy = ", valid_accuracy, " config = ", config)
                model.isTrain = True
                saver.save(sess, os.path.join(logger_path + '_' + str(step)), write_meta_graph=save_meta_graph)
                save_meta_graph = False
                logger.save(logger_path)

    end = time.time() - start

    print(model_name, train_path, labels_path, " time = ", end)


if __name__ == '__main__':
    main()
