import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import argparse

from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info, dictionary_info_single
from tree import DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier


def load_args():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--county_dict', default=1, type=int)
    parser.add_argument('--decision_tree', default=1, type=int)
    parser.add_argument('--random_forest', default=1, type=int)
    parser.add_argument('--ada_boost', default=1, type=int)
    parser.add_argument('--root_dir', default='../data/', type=str)
    args = parser.parse_args()

    return args


def county_info(args):
    county_dict = load_dictionary(args.root_dir)
    dictionary_info(county_dict)


def decision_tree_testing(x_train, y_train, x_test, y_test):
    print('Decision Tree\n\n')
    clf = DecisionTreeClassifier(max_depth=20)
    clf.fit(x_train, y_train)
    preds_train = clf.predict(x_train)
    preds_test = clf.predict(x_test)
    train_accuracy = accuracy_score(preds_train, y_train)
    test_accuracy = accuracy_score(preds_test, y_test)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = clf.predict(x_test)
    print('F1 Test {}'.format(f1(y_test, preds)))


def ada_boost_testing(x_train, y_train, x_test, y_test):
    print('AdaBoost\n\n')
    weak = AdaBoostClassifier(n_trees=1)
    weak.fit(x_train, y_train)
    preds_train = weak.predict(x_train)
    preds_test = weak.predict(x_test)
    train_accuracy = accuracy_score(preds_train, y_train)
    test_accuracy = accuracy_score(preds_test, y_test)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = weak.predict(x_test)
    print('F1 Test {}'.format(f1(y_test, preds)))


def decision_tree_various_depth(x_train, y_train, x_test, y_test):
    print('Decision Tree with depths 1-25 (inclusive)\n')

    # these will keep our points
    graphTrain = []
    graphTest = []
    graphF1 = []

    # perform decision tree testing for each depth
    # i'd like to use the decision_tree_testing function here, but we need to set the proper depth for each iteration
    for layer in range(1, 26):
        print('Current depth: ', layer)
        clf = DecisionTreeClassifier(max_depth=layer)
        clf.fit(x_train, y_train)
        preds_train = clf.predict(x_train)
        preds_test = clf.predict(x_test)
        graphTrain.append(accuracy_score(preds_train, y_train))
        graphTest.append(accuracy_score(preds_test, y_test))
        print('Train {}'.format(accuracy_score(preds_train, y_train)))
        print('Test {}'.format(accuracy_score(preds_test, y_test)))
        preds = clf.predict(x_test)
        print('F1 Test {}\n'.format(f1(y_test, preds)))
        graphF1.append(f1(y_test, preds))

    table = pd.DataFrame({
        "Max Depth": [item for item in range(1, 26)],
        "Train Accuracy": graphTrain,
        "Test Accuracy": graphTest,
        "F1 Accuracy": graphF1
    })
    print(table)

    # plot our graph and output to a file
    plt.xlabel('Depth')
    plt.ylabel('Performance')
    plt.title('Accuracy & F1 Score vs Number of Trees')
    plt.plot('Max Depth', 'Train Accuracy', data=table, color='blue')
    plt.plot('Max Depth', 'Test Accuracy', data=table, color='green')
    plt.plot('Max Depth', 'F1 Accuracy', data=table, color='red')
    plt.legend()
    plt.savefig('q1.png')

    # get best depth in terms of validation accuracy
    topAccuracy = max(graphF1)
    print("The depth that gives the best validation accuracy is: ",
          [item for item in range(1, 26)][graphF1.index(topAccuracy)], "which has an F1 accuracy of ", topAccuracy)

    # get the most important feature for making a prediction
    clfMVP = DecisionTreeClassifier(max_depth=[item for item in range(1, 26)][graphF1.index(topAccuracy)])
    clfMVP.fit(x_train, y_train)
    print("The most important feature for making a prediction is: ", clfMVP.root.feature)
    print("The threshold to split on for this feature is: ", clfMVP.root.split)

    # return the most important feature for use in main
    return clfMVP.root.feature


def random_forest_testing(x_train, y_train, x_test, y_test):
    print('Random Forest\n\n')
    rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=50)
    rclf.fit(x_train, y_train)
    preds_train = rclf.predict(x_train)
    preds_test = rclf.predict(x_test)
    train_accuracy = accuracy_score(preds_train, y_train)
    test_accuracy = accuracy_score(preds_test, y_test)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = rclf.predict(x_test)
    print('F1 Test {}'.format(f1(y_test, preds)))


def random_forest_various_trees(x_train, y_train, x_test, y_test):
    graphTrain = []
    graphTest = []
    graphF1 = []

    # let the user know which test this is
    print("== Beginning test for various n_trees.\n")

    # plot accuracies for the number of trees specified in part b
    for i in range(10, 210, 10):
        print("n_trees: ", i)
        rclf = RandomForestClassifier(
            max_depth=7, max_features=11, n_trees=i
        )
        rclf.fit(x_train, y_train)
        preds_train = rclf.predict(x_train)
        preds_test = rclf.predict(x_test)
        graphTrain.append(accuracy_score(preds_train, y_train))
        graphTest.append(accuracy_score(preds_test, y_test))
        print('Train {}'.format(accuracy_score(preds_train, y_train)))
        print('Test {}'.format(accuracy_score(preds_test, y_test)))
        preds = rclf.predict(x_test)
        print('F1 Test {}\n'.format(f1(y_test, preds)))
        graphF1.append(f1(y_test, preds))

    # table for easily reading data
    table = pd.DataFrame({
        "n_trees": [i for i in range(10, 210, 10)],
        "Train Accuracy": graphTrain,
        "Test Accuracy": graphTest,
        "F1 Accuracy": graphF1
    })
    print(table)

    # plot our graph and output to a file
    plt.figure(2)
    plt.xlabel('Number of trees')
    plt.ylabel('Performance')
    plt.title('Accuracy & F1 Score vs Number of Trees in the Forest')
    plt.plot('n_trees', 'Train Accuracy', data=table, color='blue')
    plt.plot('n_trees', 'Test Accuracy', data=table, color='green')
    plt.plot('n_trees', 'F1 Accuracy', data=table, color='red')
    plt.legend()
    plt.savefig('q2pb.png')

    # return our best n__trees value for use in main
    return [i for i in range(10, 210, 10)][graphF1.index(max(graphF1))]


def random_forest_various_features(x_train, y_train, x_test, y_test):
    # keep our values to use for max_features
    useFeatures = [1, 2, 5, 8, 10, 20, 25, 35, 50]

    # for whatever reason, same variable names cause issues despite being within local scope
    # so we have to make sure there are no matching variable names even between functions

    graphTrain2 = []
    graphTest2 = []
    graphF12 = []

    # let the user know which test this is
    print("== Beginning test for various max_features.\n")

    for features in useFeatures:
        print("max_features: ", features)
        rclf = RandomForestClassifier(
            max_depth=7, max_features=features, n_trees=50
        )
        rclf.fit(x_train, y_train)
        preds_train = rclf.predict(x_train)
        preds_test = rclf.predict(x_test)
        graphTrain2.append(accuracy_score(preds_train, y_train))
        graphTest2.append(accuracy_score(preds_test, y_test))
        print('Train {}'.format(accuracy_score(preds_train, y_train)))
        print('Test {}'.format(accuracy_score(preds_test, y_test)))
        preds = rclf.predict(x_test)
        graphF12.append(f1(y_test, preds))
        print('F1 Test {}\n'.format(f1(y_test, preds)))

    # print lengths for debugging
    print("== Length of Train", len(graphTrain2))
    print("== Length of Test", len(graphTest2))
    print("== Length of F1", len(graphF12))

    # table for easily reading data
    table2 = pd.DataFrame({
        "max_features": [i for i in useFeatures],
        "Train Accuracy": graphTrain2,
        "Test Accuracy": graphTest2,
        "F1 Accuracy": graphF12
    })
    print(table2)

    # plot our graph and output to a file
    plt.figure(3)
    plt.xlabel('Max Features')
    plt.ylabel('Performance')
    plt.title('Accuracy & F1 Score vs Max Features')
    plt.plot('max_features', 'Train Accuracy', data=table2, color='blue')
    plt.plot('max_features', 'Test Accuracy', data=table2, color='green')
    plt.plot('max_features', 'F1 Accuracy', data=table2, color='red')
    plt.legend()
    plt.savefig('q2pd.png')

    # return best value for max_features to use in main
    return [feature for feature in useFeatures][graphF12.index(max(graphF12))]


def random_forest_various_seeds(x_train, y_train, x_test, y_test, best_max_features, best_n_trees):
    # let the user know which test this is
    print("== Beginning test for best result with random seeds.\n")

    # to hold data points
    randseedTrain = []
    randseedTest = []
    randseedF1 = []
    averageSeeds = []
    averageTrain = []
    averageTest = []
    averageF1 = []
    usedSeeds = []

    rclf = RandomForestClassifier(
        max_depth=7, max_features=best_max_features, n_trees=best_n_trees
    )

    for item in [i for i in range(10)]:
        rclf.seed = np.random.randint(1, 1000)
        usedSeeds.append(rclf.seed)
        rclf.fit(x_train, y_train)
        preds_train = rclf.predict(x_train)
        preds_test = rclf.predict(x_test)
        randseedTrain.append(accuracy_score(preds_train, y_train))
        randseedTest.append(accuracy_score(preds_test, y_test))
        print('Train {}'.format(accuracy_score(preds_train, y_train)))
        print('Test {}'.format(accuracy_score(preds_test, y_test)))
        preds = rclf.predict(x_test)
        randseedF1.append(f1(y_test, preds))
        print('F1 Test {}\n'.format(f1(y_test, preds)))

    # get averages
    averageSeeds.append("Average")
    averageTrain.append(sum(randseedTrain) / len(randseedTrain))
    averageTest.append(sum(randseedTest) / len(randseedTest))
    averageF1.append(sum(randseedF1) / len(randseedF1))

    # get table for data + add averages at the end
    table3 = pd.DataFrame({
        "Seed": [i for i in usedSeeds] + averageSeeds,
        "Train Accuracy": randseedTrain + averageTrain,
        "Test Accuracy": randseedTest + averageTest,
        "F1 Score": randseedF1 + averageF1
    })
    print(table3)


###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
    args = load_args()
    x_train, y_train, x_test, y_test = load_data(args.root_dir)
    if args.county_dict == 1:
        county_info(args)

    if args.decision_tree == 1:
        decision_tree_testing(x_train, y_train, x_test, y_test)
        print('\n')
        mvpFeature = decision_tree_various_depth(x_train, y_train, x_test, y_test)
        county_dict = load_dictionary(args.root_dir)
        dictionary_info_single(county_dict, mvpFeature)

    if args.random_forest == 1:
        print("== Running random forest testing.\n")
        random_forest_testing(x_train, y_train, x_test, y_test)
        best_trees = random_forest_various_trees(x_train, y_train, x_test, y_test)
        best_features = random_forest_various_features(x_train, y_train, x_test, y_test)
        random_forest_various_seeds(x_train, y_train, x_test, y_test, best_features, best_trees)

    if args.ada_boost == 1:
        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1
        ada_boost_testing(x_train, y_train, x_test, y_test)

        print('Done')
