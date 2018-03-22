import os
import tensorflow as tf

from config import Config
from history_logger import HistoryLogger
from utils import get_batch_data_iterator

# TODO: ix the argument.
# TODO: find the best model.
# TODO: copy best model to best model.
# TODO: cross check submission.


def test():
    logger_path = '/Users/mohitrohatgi/PycharmProjects/cs763_assign4/model_name/1521725203/1521725203'
    model_name = '/Users/mohitrohatgi/Downloads/assign4/test_data.txt'
    logger = HistoryLogger.load(logger_path)
    best_model = logger_path + '_' + str(10)
    print(logger.best_model)
    graph = tf.Graph()
    config = Config()
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
            batches, data_len = get_batch_data_iterator(1, data_path=model_name, label_path=None,
                                                        seq_length=config.seq_length, batch_size=config.batch_size,
                                                        mode='test')

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions.append(batch_predictions)

    index1 = data_len - config.batch_size - 1
    index2 = data_len - data_len % config.batch_size
    predictions = []

    index = 0
    for prediction_batch in all_predictions:
        for prediction in prediction_batch:
            if not (index1 < index < index2):
                predictions.append(prediction[0])
            index += 1
    del all_predictions

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testPrediction.bin'), 'w') as f:
        f.write("id,label\n")
        for line_id, prediction in enumerate(predictions):
            f.write(str(line_id) + "," + str(prediction) + "\n")
    # model_name = sys.argv[sys.argv.index('-modelName') + 1]
    # test_path = sys.argv[sys.argv.index('-data') + 1]
    # print(model_name, test_path)


if __name__ == '__main__':
    test()
