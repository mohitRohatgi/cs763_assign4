import numpy as np
import pickle
import os
import copy


class HistoryLogger:
    def __init__(self, config, mode='loss'):
        self.step = []
        self.valid_loss = []
        self.train_loss = []
        self.valid_accuracy = []
        self.train_accuracy = []
        self.best_model = None
        self.mode = mode
        self.config = copy.deepcopy(config)

    def add(self, train_loss, train_accuracy, valid_loss, valid_accuracy, step):
        self.step.append(step)
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)
        self.valid_loss.append(valid_loss)
        self.valid_accuracy.append(valid_accuracy)

    def save(self, dir_path):
        logger = copy.deepcopy(self)
        logger.step = np.array(self.step)
        logger.valid_loss = np.array(self.valid_loss)
        logger.valid_accuracy = np.array(self.valid_accuracy)
        logger.train_loss = np.array(self.train_loss)
        logger.train_accuracy = np.array(self.train_accuracy)
        logger.best_model = self._find_best_model(dir_path)
        path = os.path.join(dir_path)
        output_file = open(path + '.pkl', 'wb')
        pickle.dump(logger, output_file, pickle.HIGHEST_PROTOCOL)
        output_file.close()

    def _find_best_model(self, dir_path):
        if self.mode == 'accuracy':
            best_step = self.step[int(np.argmax(self.valid_accuracy))]
        else:
            best_step = self.step[int(np.argmin(self.valid_loss))]
        return dir_path + '_' + str(best_step)

    @staticmethod
    def load(path):
        with open(path + ".pkl", 'rb') as input_file:
            return pickle.load(input_file)
