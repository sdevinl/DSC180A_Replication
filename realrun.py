import sys
import src.data
import src.utils
from src.main import *


def main(targets):
    testfile, d1, d2 = src.data.make_dataset(targets)
    X, A, train_idx, test_idx, labels, labels_distr, le = src.data.preprocess_data(d1, d2)

    init_variables(testfile, X, A, train_idx, test_idx, labels_distr, labels, le)
    run_LPA_GCN()


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)