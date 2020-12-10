from src.models import *
import time
from src.data import *
from src.utils import *

#testfile, X, A, train_idx, test_idx, labels_distr, labels, le = 0, 0, 0, 0, 0, 0, 0, 0

def run_LPA_GCN(epochs=10, Lambda=.4):
    testfile.write('---------------LPA-GCN---------------------- \n')
    print('---------------LPA-GCN----------------------')
    GCN_LPA_model = GCN_LPA(X.shape[1], 300, len(le.classes_), A)
    optimizer = torch.optim.SGD(GCN_LPA_model.parameters(), lr=.1)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in np.arange(epochs):
        t = time.time()
        GCN_LPA_model.train()
        optimizer.zero_grad()
        output, Y_hat = GCN_LPA_model(X, A, labels_distr)

        loss_gcn = criterion(output[train_idx], labels[train_idx])
        loss_lpa = criterion(Y_hat[train_idx], labels[train_idx])

        acc = accuracy(output[train_idx], labels[train_idx])
        loss_train = loss_gcn + Lambda * loss_lpa

        loss_train.backward()
        optimizer.step()

        testfile.write('Epoch: {:04d}'.format(epoch + 1) + ' ' +
                       'loss_train: {:.4f}'.format(loss_train.item()) + ' ' +
                       'acc_train: {:.4f}'.format(acc.item()) + ' ' +
                       'time: {:.4f}s'.format(time.time() - t) + '\n')

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc.item()),
              'time: {:.4f}s'.format(time.time() - t))

    # Test
    GCN_LPA_model.eval()
    output, Y_hat = GCN_LPA_model(X, A, labels_distr)

    loss_gcn = criterion(output[test_idx], labels[test_idx])
    loss_lpa = criterion(Y_hat[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])

    loss_test = loss_gcn + Lambda * loss_lpa

    testfile.write('Optimization Finished! \n')
    testfile.write("Test set results:" + ' ' +
                   "loss= {:.4f}".format(loss_test.item()) + ' ' +
                   "accuracy= {:.4f}".format(acc_test.item()) + '\n')

    print("Optimization Finished!")
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    testfile.write('\n')
    print('\n')

    # Model and Optimizer

    # GCN takes in number of papers, number hidden layers, and number of classes


# Train model
def run_GCN(epochs=10, prep_A=None):
    testfile.write('---------GCN----------- \n')
    print('---------GCN-----------')
    model = GCN(X.shape[1], 300, len(le.classes_))

    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    criterion = torch.nn.CrossEntropyLoss()

    # Train and Test functions
    def train(epoch, prep_A=None):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(X, A, prep_A)
        loss = criterion(output[train_idx], labels[train_idx])
        acc = accuracy(output[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        testfile.write('Epoch: {:04d}'.format(epoch + 1) + ' '
                       'loss_train: {:.4f}'.format(loss.item()) + ' '
                       'acc_train: {:.4f}'.format(
            acc.item()) + ' '
                          'time: {:.4f}s'.format(time.time() - t) + '\n')
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(acc.item()),
              'time: {:.4f}s'.format(time.time() - t))

    def test(prep_A=None):
        model.eval()
        output = model(X, A, prep_A)
        loss_test = criterion(output[test_idx], labels[test_idx])
        acc_test = accuracy(output[test_idx], labels[test_idx])

        testfile.write("Test set results:" + ' '
                       "loss= {:.4f}".format(loss_test.item()) + ' '
                       "accuracy= {:.4f}".format(acc_test.item()) + '\n')
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        testfile.write('\n')
        print('\n')

    t_total = time.time()
    for epoch in range(epochs):
        train(epoch, prep_A)

    testfile.write('Optimization Finished! \n')
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test(prep_A)


# GraphSAGE model
def run_GS(epochs=10, Lambda=10):
    testfile.write('----------------GraphSAGE--------------- \n')
    print('-----------GraphSAGE------------------')
    GS_model = GS(X.shape[1], 300, len(le.classes_))
    optimizer = torch.optim.SGD(GS_model.parameters(), lr=.1)
    criterion = torch.nn.CrossEntropyLoss()

    # Train and Test functions
    def train(epoch, prep_A=None):
        t = time.time()
        GS_model.train()
        optimizer.zero_grad()
        output = GS_model(X, A)
        loss = criterion(output[train_idx], labels[train_idx])
        acc = accuracy(output[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        testfile.write('Epoch: {:04d}'.format(epoch + 1) + ' '
                        'loss_train: {:.4f}'.format(loss.item()) + ' '
                        'acc_train: {:.4f}'.format(acc.item()) + ' '
                        'time: {:.4f}s'.format(time.time() - t) + '\n')
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(acc.item()),
              'time: {:.4f}s'.format(time.time() - t))

    def test(prep_A=None):
        GS_model.eval()
        output = GS_model(X, A)
        loss_test = criterion(output[test_idx], labels[test_idx])
        acc_test = accuracy(output[test_idx], labels[test_idx])

        testfile.write("Test set results:" + ' '
                        "loss= {:.4f}".format(loss_test.item()) + ' '
                        "accuracy= {:.4f}".format(acc_test.item()) + '\n')
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        testfile.write('\n')
        print('\n')

    t_total = time.time()
    for epoch in range(epochs):
        train(epoch)

    testfile.write('Optimization Finished! \n')
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()

