import sys

# TODO: load best model data.
# TODO: load best model.
# TODO: output the test data predicted labels.
# TODO: get the best accuracy.


def test():
    model_name = sys.argv[sys.argv.index('-modelName') + 1]
    test_path = sys.argv[sys.argv.index('-data') + 1]
    print(model_name, test_path)


if __name__ == '__main__':
    test()
