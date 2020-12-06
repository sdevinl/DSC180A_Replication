import sys
import src.data
import src.utils
import src.main


def main(targets):
    testfile, d1, d2 = src.data.make_dataset(targets)
    X, A, train_idx, test_idx, labels, labels_distr, le = src.data.preprocess_data(d1, d2)
    src.main.X, src.main.A, src.main.train_idx, src.main.test_idx, src.main.labels, src.main.labels_distr, src.main.le = src.data.preprocess_data(d1, d2)
    src.main.testfile = testfile


    #print(src.main.X)
    src.main.run_LPA_GCN()
    src.main.run_GCN()


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)