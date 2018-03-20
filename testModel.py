import sys


def test():
    model_name = sys.argv[sys.argv.index('-modelName') + 1]
    test_path = sys.argv[sys.argv.index('-data') + 1]
    print(model_name, test_path)


if __name__ == '__main__':
    test()
