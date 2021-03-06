from src.models import *
import time
from src.data import *
from src.utils import *
from src.data import *

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


# New
def GCN_Train_Tester(data='cora', epoch=50, lr=0.01, weight_decay=5e-4, hidden=16, seed=51, norm=True, output_path='output/data.txt'):
    print('\nGCN')
    output_path.write('\nGCN\n')

    if data == 'cora':
        adj, features, labels, idx_train, idx_val, idx_test, labels_lpa = cora_ingestion('data/')
    elif data == 'ogb':
        adj, features, labels, idx_train, idx_val, idx_test, labels_lpa = arxiv_ingestion()

    np.random.seed(seed)
    torch.manual_seed(seed)

    model = GCN(feature_size=features.shape[1], hidden_size=hidden, class_size=labels.max().item() + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(epoch):
        for epoch in range(epoch):
            model.train()
            optimizer.zero_grad()

            output = model(features, adj, norm)

            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            loss_train.backward()
            optimizer.step()

            model.eval()
            output = model(features, adj, norm)

            acc_val = accuracy(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            print('Epoch:' + str((epoch + 1)).zfill(4),
                  '| train loss: ' + str(round(float(loss_train), 4)),
                  '| train acc: ' + str(round(float(acc_train), 4)),
                  '| valid loss: ' + str(round(float(loss_val), 4)),
                  '| valid acc: ' + str(round(float(acc_val), 4)))
            print('')

        output_path.write(
              'train acc: ' + str(round(float(acc_train), 4)) +
              '| valid acc: ' + str(round(float(acc_val), 4)))
        output_path.write(' ')

    def results():
        model.eval()
        output = model(features, adj, norm)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print('******************************************************')
        print("Results: ",
              "| test loss: " + str(round(float(loss_test), 8)),
              "| test accuracy: " + str(round(float(acc_test), 8)))
        print('******************************************************')
        print('')

        output_path.write("Results: " +
              "| test accuracy: " + str(round(float(acc_test), 8)) + '\n')
        output_path.write('******************************************************\n')
        output_path.write('')

    # Train & Test
    start = time.time()
    train(epoch)
    print("Time Elapsed: " + str(round((time.time() - start), 2)) + ' seconds')
    results()


def GCN_LPA_Train_Tester(data='cora', epoch=50, lr=0.01, weight_decay=5e-4, hidden=16, seed=54, Lambda=1, norm=True, output_path='output/data.txt'):
    print('\nLPA-GCN')
    output_path.write('\nLPA-GCN\n')

    if data == 'cora':
        adj, features, labels, idx_train, idx_val, idx_test, labels_lpa = cora_ingestion('data/')
    elif data == 'ogb':
        adj, features, labels, idx_train, idx_val, idx_test, labels_lpa = arxiv_ingestion()

    np.random.seed(seed)
    torch.manual_seed(seed)
    labels_lpa_transformed = torch.from_numpy(labels_lpa).type(torch.FloatTensor)

    model = GCN_LPA(feature_size=features.shape[1], hidden_size=hidden, class_size=labels.max().item() + 1, adj=adj)

    optimizer = torch.optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    def train(epoch):
        for epoch in range(epoch):
            model.train()
            optimizer.zero_grad()
            output, y_hat = model(features, adj, labels_lpa_transformed, norm=norm)
            loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
            loss_lpa = F.nll_loss(y_hat, labels)
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train = loss_gcn + Lambda * loss_lpa
            loss_train.backward(retain_graph=True)
            optimizer.step()

            model.eval()
            output = (model(features, adj, labels_lpa_transformed, norm=norm))[0]

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch:' + str((epoch + 1)).zfill(4),
                  '| train loss: ' + str(round(float(loss_train), 4)),
                  '| train acc: ' + str(round(float(acc_train), 4)),
                  '| valid loss: ' + str(round(float(loss_val), 4)),
                  '| valid acc: ' + str(round(float(acc_val), 4)))

        output_path.write(
            'train acc: ' + str(round(float(acc_train), 4)) +
            '| valid acc: ' + str(round(float(acc_val), 4)))
        output_path.write(' ')

    def results():
        model.eval()
        output = (model(features, adj, labels_lpa_transformed, norm=norm))[0]
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print('******************************************************')
        print("Results: ",
              "| test loss: " + str(round(float(loss_test), 4)),
              "| test accuracy: " + str(round(float(acc_test), 4)))
        print('******************************************************')
        print('')

        output_path.write("Results: " +
              "| test accuracy: " + str(round(float(acc_test), 8)) + '\n')
        output_path.write('******************************************************\n')
        output_path.write('')

    start = time.time()
    train(epoch)
    print("Time Elapsed: " + str(round((time.time() - start), 2)) + ' seconds')
    results()


def GS_Train_Tester(data='cora', epoch=100, lr=0.01, weight_decay=5e-4, hidden=16, seed=51, output_path='', norm=False):
    print('\nGraphSAGE')
    output_path.write('\nGraphSAGE\n')

    if data == 'cora':
        adj, features, labels, idx_train, idx_val, idx_test, labels_lpa = cora_ingestion('data/')
    elif data == 'ogb':
        adj, features, labels, idx_train, idx_val, idx_test, labels_lpa = arxiv_ingestion()

    np.random.seed(seed)
    torch.manual_seed(seed)

    model = GS(nfeat=features.shape[1], nhid=hidden, nclass=labels.max().item() + 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(epoch):
        for epoch in range(epoch):
            model.train()
            optimizer.zero_grad()

            output = model(features, adj, norm)

            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            loss_train.backward()
            optimizer.step()

            model.eval()
            output = model(features, adj, norm)

            acc_val = accuracy(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            print('Epoch:' + str((epoch + 1)).zfill(4),
                  '| train loss: ' + str(round(float(loss_train), 4)),
                  '| train acc: ' + str(round(float(acc_train), 4)),
                  '| valid loss: ' + str(round(float(loss_val), 4)),
                  '| valid acc: ' + str(round(float(acc_val), 4)))
            print('')

        output_path.write(
            'train acc: ' + str(round(float(acc_train), 4)) +
            '| valid acc: ' + str(round(float(acc_val), 4)))
        output_path.write(' ')

    def results():
        model.eval()
        output = model(features, adj, norm)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print('******************************************************')
        print("Results: ",
              "| test loss: " + str(round(float(loss_test), 8)),
              "| test accuracy: " + str(round(float(acc_test), 8)))
        print('******************************************************')
        print('')

        output_path.write("Results: " +
                          "| test accuracy: " + str(round(float(acc_test), 8)) + '\n')
        output_path.write('******************************************************\n')
        output_path.write('')

    # Train & Test
    start = time.time()
    train(epoch)
    print("Time Elapsed: " + str(round((time.time() - start), 2)) + ' seconds')
    results()