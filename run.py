import sys
import src.data
import src.utils
import src.main


def main(targets):
    testfile, d1, d2, labels, targets, src.main.labels_distr, src.main.le = src.data.make_dataset(targets)
    # X, A, train_idx, test_idx, labels = src.data.preprocess_data(d1, d2, labels, targets)
    src.main.X, src.main.A, src.main.train_idx, src.main.test_idx, src.main.labels = src.data.preprocess_data(d1, d2,
                                                                                                              labels,
                                                                                                              targets)
    src.main.testfile = testfile

    try:
        epochs = int(targets[1])
    except IndexError:
        print('No epochs listed using ' + epochs + ' epochs')
        epochs = 3

    src.main.run_LPA_GCN(epochs)
    src.main.run_GCN(epochs)
    src.main.run_GS(epochs)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
