import sys
import src.data
import src.utils
import src.main


def main(targets):
    #testfile, d1, d2, labels, targets, src.main.labels_distr, src.main.le = src.data.make_dataset(targets)
    # X, A, train_idx, test_idx, labels = src.data.preprocess_data(d1, d2, labels, targets)
    ##src.main.X, src.main.A, src.main.train_idx, src.main.test_idx, src.main.labels = src.data.preprocess_data(d1, d2,
     #                                                                                                         labels,
      #                                                                                                        targets)
    #src.main.testfile = testfile


    #src.main.run_LPA_GCN(epochs)
    #src.main.run_GCN(epochs)
    #src.main.run_GS(epochs)



    try:
        if targets[0] not in ['cora', 'ogb', 'test']:
            targets.append('cora')

    except IndexError:
        print('No dataset specified using default (cora)')
        targets.append('cora')

    try:
        targets[1] = int(targets[1])
    except IndexError:
        targets.append(3)
        print('No epochs listed using ' + str(targets[1]) + ' epochs')

    try:
        if targets[2].lower() == 'false':
            targets[2] = False
        else:
            targets[2] = True
    except IndexError:
        targets.append(False)
        print('Normalization not specified, will not normalize adjacency matrix')

    try:
            targets[3] = float(targets[3])
    except:
        print('Learning rate not specified, using .01')
        targets.append(.01)

    # Clear Text File
    testfile = open('output/' + targets[0] + 'data.txt', 'w')
    testfile.write('')
    testfile.close()
    # New Output
    testfile = open('output/' + targets[0] + 'data.txt', 'a')
    testfile.write('Data: ' + targets[0] + '| Epochs: ' + str(targets[1]) + '| Normalize A: ' + str(targets[2]) + '| Learning Rate: ' + str(targets[3]) + '\n')

    print('Targets: ',  targets)

    src.main.GCN_Train_Tester(data=targets[0], epoch=targets[1], norm=targets[2], lr=targets[3], output_path=testfile)

    src.main.GCN_LPA_Train_Tester(data=targets[0], epoch=targets[1], norm=targets[2], lr=targets[3], output_path=testfile)

    src.main.GS_Train_Tester(data=targets[0], epoch=targets[1], norm=targets[2], lr=targets[3], output_path=testfile)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
