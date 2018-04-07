import tensorflow as tf

from history_logger import HistoryLogger
from utils import get_batch_data_iterator
import os

# TODO: copy best model to best model.


def find_latest(model_name):
    dirs = [int(o) for o in os.listdir(model_name) if os.path.isdir(os.path.join(model_name, o))]
    latest_dir = max(dirs)
    return os.path.join(model_name, str(latest_dir), str(latest_dir))


def test():
    model_name = os.path.join(os.getcwd(), 'model_name')
    model_name = find_latest(model_name)
    test_path = os.path.join(os.getcwd(), 'data/test_data.txt')
    logger = HistoryLogger.load(model_name)
    best_model = logger.best_model
    # best_model = '/home/mrohatgi/cs763_assign4/model_name/1523010750/1523010750_30200'
    graph = tf.Graph()
    config = logger.config

    # Generate batches for one epoch
    data_gen = get_batch_data_iterator(1, data_path=test_path, label_path=None, mode='test',
                                       seq_length=config.seq_length, batch_size=config.batch_size, bins=config.bins)
    with graph.as_default():
        sess = tf.Session(graph=graph)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            graph_path = best_model[:best_model.rfind('_')] + '_' + str(config.evaluate_every)
            saver = tf.train.import_meta_graph("{}.meta".format(graph_path))
            saver.restore(sess, logger.best_model)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input").outputs[0]

            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout").outputs[0]

            # Collect the predictions here
            all_predictions = []

            while True:
                try:
                    batches, data_len = data_gen.__next__()
                    # Tensors we want to evaluate
                    predictions = graph.get_operation_by_name("prediction_" + str(data_len - 1)).outputs[0]
                    batch_predictions = sess.run(predictions, {input_x: batches, dropout_keep_prob: 1.0})
                    all_predictions.append(batch_predictions[0])
                except:
                    break

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testPrediction.bin'), 'w') as f:
        f.write("id,label\n")
        for line_id, prediction in enumerate(all_predictions):
            f.write(str(line_id) + "," + str(prediction) + "\n")


if __name__ == '__main__':
    test()
