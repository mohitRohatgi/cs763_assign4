import csv
import os

import numpy as np
import tensorflow as tf

from history_logger import HistoryLogger


# TODO: load best model data.
# TODO: load best model.
# TODO: output the test data predicted labels.
# TODO: find the best model.
# TODO: copy best model to best model.


def test():
    logger_path = '/Users/mohitrohatgi/PycharmProjects/cs763_assign4/model_name/1521720220/1521720220'
    logger = HistoryLogger.load(logger_path)
    best_model = logger_path + '_' + str(10)
    print(logger.best_model)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(best_model))
            saver.restore(sess, logger.best_model)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("model/input").outputs[0]

            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("model/dropout").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("model/prediction").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testPrediction.bin'), 'w') as f:
        csv.writer(f).writerows(all_predictions)
    # model_name = sys.argv[sys.argv.index('-modelName') + 1]
    # test_path = sys.argv[sys.argv.index('-data') + 1]
    # print(model_name, test_path)


if __name__ == '__main__':
    test()
