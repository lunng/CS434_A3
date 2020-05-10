import numpy as np
import math


class Node():
    """
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""

    def __init__(self, prediction, feature, split, left_tree, right_tree):
        self.prediction = prediction
        self.feature = feature
        self.split = split
        self.left_tree = left_tree
        self.right_tree = right_tree


class DecisionTreeClassifier():
    """
	Decision Tree Classifier. Class for building the decision tree and making predictions

	Parameters:
	------------
	max_depth: int
		The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
	"""

    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features

    # take in features X and labels y
    # build a tree
    def fit(self, X, y):
        self.num_classes = len(set(y))
        self.root = self.build_tree(X, y, depth=1)

    # make prediction for each example of features X
    def predict(self, X):
        preds = [self._predict(example) for example in X]

        return preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):
        node = self.root
        while node.left_tree:
            if example[node.feature] < node.split:
                node = node.left_tree
            else:
                node = node.right_tree
        return node.prediction

    # accuracy
    def accuracy_score(self, X, y):
        preds = self.predict(X)
        accuracy = (preds == y).sum() / len(y)
        return accuracy

    # function to build a decision tree
    def build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on
        self.features_idx = np.arange(0, X.shape[1])

        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_split = None
        best_gain = 0.0
        best_left_X = None
        best_left_y = None
        best_right_X = None
        best_right_y = None

        # what we would predict at this node if we had to
        # majority class
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        prediction = np.argmax(num_samples_per_class)

        # if we haven't hit the maximum depth, keep building
        if depth <= self.max_depth:
            # consider each feature
            for feature in self.features_idx:
                # consider the set of all values for that feature to split on
                possible_splits = np.unique(X[:, feature])
                for split in possible_splits:
                    # get the gain and the data on each side of the split
                    # >= split goes on right, < goes on left
                    gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
                    # if we have a better gain, use this split and keep track of data
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split = split
                        best_left_X = left_X
                        best_right_X = right_X
                        best_left_y = left_y
                        best_right_y = right_y

        # if we haven't hit a leaf node
        # add subtrees recursively
        if best_gain > 0.0:
            left_tree = self.build_tree(best_left_X, best_left_y, depth=depth + 1)
            right_tree = self.build_tree(best_right_X, best_right_y, depth=depth + 1)
            return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree,
                        right_tree=right_tree)

        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)

    # gets data corresponding to a split by using numpy indexing
    def check_split(self, X, y, feature, split):
        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]

        # calculate gini impurity and gain for y, left_y, right_y
        gain = self.calculate_gini_gain(y, left_y, right_y)
        return gain, left_X, right_X, left_y, right_y

    def calculate_gini_gain(self, y, left_y, right_y):
        # not a leaf node
        # calculate gini impurity and gain
        gain = 0
        if len(left_y) > 0 and len(right_y) > 0:

            ########################################
            #       YOUR CODE GOES HERE            #
            ########################################

            # get weights
            weightL = len(left_y) / len(y)
            weightR = len(right_y) / len(y)

            # calculate impurities
            positiveY = (y == 1).sum() / len(y)
            negativeY = (y == 0).sum() / len(y)
            impurityY = 1 - (positiveY ** 2) - (negativeY ** 2)

            positiveL = (left_y == 1).sum() / len(left_y)
            negativeL = (left_y == 0).sum() / len(left_y)
            impurityL = 1 - (positiveL ** 2) - (negativeL ** 2)

            positiveR = (right_y == 1).sum() / len(right_y)
            negativeR = (right_y == 0).sum() / len(right_y)
            impurityR = 1 - (positiveR ** 2) - (negativeR ** 2)

            # calculate the gain from the impurities and weights
            gain = impurityY - (weightL * impurityL) - (weightR * impurityR)

            # return the gain
            return gain
        # we hit leaf node
        # don't have any gain, and don't want to divide by 0
        else:
            return 0


class RandomForestClassifier():
    """
	Random Forest Classifier. Build a forest of decision trees.
	Use this forest for ensemble predictions

	YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

	Parameters:
	-----------
	n_trees: int
		Number of trees in forest/ensemble
	max_features: int
		Maximum number of features to consider for a split when feature bagging
	max_depth: int
		Maximum depth of any decision tree in forest/ensemble
	"""

    def __init__(self, n_trees, max_features, max_depth):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth

        ##################
        # YOUR CODE HERE #
        ##################

        self.trees = []
        self.seed = None

        for i in range(self.n_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, max_features=self.max_features
            )
            self.trees.append(tree)

    # fit all trees
    def fit(self, X, y):
        bagged_X, bagged_y = self.bag_data(X, y)
        print('Fitting Random Forest...\n')
        for i in range(self.n_trees):
            print(i + 1, end='\t\r')
            ##################
            # YOUR CODE HERE #
            ##################

            self.trees[i].fit(bagged_X[i], bagged_y[i])

        print()

    def bag_data(self, X, y, proportion=1.0):
        bagged_X = []
        bagged_y = []
        for i in range(self.n_trees):
            ##################
            # YOUR CODE HERE #
            ##################

            tree_x = []
            tree_y = []

            for j in range(len(X)):
                random = np.random.randint(len(X) - 1)
                tree_x.append(X[random])
                tree_y.append(y[random])

            bagged_X.append(tree_x)
            bagged_y.append(tree_y)

        # ensure data is still numpy arrays
        return np.array(bagged_X), np.array(bagged_y)

    def predict(self, X):
        preds = np.zeros(len(X))

        # remove this one \/
        # preds = np.ones(len(X)).astype(int)
        # ^that line is only here so the code runs

        ##################
        # YOUR CODE HERE #
        ##################

        for i in range(self.n_trees):
            singlePred = np.array(self.trees[i].predict(X))
            preds = preds + singlePred

        preds = preds.tolist()

        for i in range(len(preds)):
            if (preds[i] > self.n_trees / 2):
                preds[i] = 1
            else:
                preds[i] = 0

        return preds


################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################
class AdaBoostClassifier:
    """
    AdaBoost Classifier. Build boosted decision trees.
    Use this for ensemble predictions

    Parameters:
    -----------
    n_trees: int
        Number of trees in ensemble
    max_features: int
        Maximum number of features to consider for a split when feature bagging
    """

    def __init__(self, n_trees, max_features=None):
        self.n_trees = n_trees
        self.max_features = max_features
        self.classifiers = []

    def fit(self, X, y):
        ab = None
        d = np.full(len(y), 1/len(y))
        for _ in range(self.n_trees):
            ab = DecisionTreeAdaBoost(self.max_features, d)
            ab.fit(X, y)

            preds = ab.predict(X)
            correct_idx = np.where(preds == y)
            incorrect_idx = np.where(preds != y)

            err = d[incorrect_idx].sum()

            a_t = .5 * math.log((1 - err) / err)

            d[correct_idx] *= (math.e ** -a_t)
            d[incorrect_idx] *= (math.e ** a_t)
            d /= d.sum()

            classifier = {'a_t': a_t, 'tree': ab}

            self.classifiers.append(classifier)

    def predict(self, X):
        preds = None
        for classifier in self.classifiers:
            temp_preds = np.asarray(classifier['tree'].predict(X))
            if preds is None:
                preds = np.zeros(len(temp_preds))
            temp_preds = (temp_preds * classifier['a_t'])
            preds += temp_preds

        preds[preds < 0] = -1
        preds[preds > 0] = 1
        return preds

class DecisionTreeAdaBoost(DecisionTreeClassifier):
    def __init__(self, max_features=None, d=None):
        super().__init__(max_features=max_features)
        self.max_depth = 1
        self.d = d

    def calculate_gini_gain(self, y, left_y, right_y):
        raise (NotImplementedError, 'In DecisionTreeAdaBoost do not use calculate_gini_gain,'
                                    ' instead use calculate_gini_gain_ab')

    def calculate_gini_gain_ab(self, y, left_y, right_y, left_w, right_w, d):
        # implement a different gain calculation using the weights
        # Assumption: Since we are using stumping len(y) will always be equal to len(d)

        assert(len(y) == len(self.d))

        # calculate gini impurity and gain
        if len(left_y) > 0 and len(right_y) > 0:

            ########################################
            #       YOUR CODE GOES HERE            #
            ########################################

            # get weights
            weightL = left_w.sum() / d.sum()
            weightR = right_w.sum() / d.sum()

            # calculate impurities
            positiveY_idx = np.where(y == 1)
            positiveY = d[positiveY_idx].sum() / d.sum()
            negativeY_idx = np.where(y == -1)
            negativeY = d[negativeY_idx].sum() / d.sum()
            impurityY = 1 - (positiveY ** 2) - (negativeY ** 2)

            positiveL_idx = np.where(left_y == 1)
            positiveL = left_w[positiveL_idx].sum() / left_w.sum()
            negativeL_idx = np.where(left_y == -1)
            negativeL = left_w[negativeL_idx].sum() / left_w.sum()
            impurityL = 1 - (positiveL ** 2) - (negativeL ** 2)

            positiveR_idx = np.where(right_y == 1)
            positiveR = right_w[positiveR_idx].sum() / right_w.sum()
            negativeR_idx = np.where(right_y == -1)
            negativeR = right_w[negativeR_idx].sum() / right_w.sum()
            impurityR = 1 - (positiveR ** 2) - (negativeR ** 2)

            # calculate the gain from the impurities and weights
            gain = impurityY - (weightL * impurityL) - (weightR * impurityR)

            # return the gain
            return gain
        # we hit leaf node
        # don't have any gain, and don't want to divide by 0
        else:
            return 0

    # gets data corresponding to a split by using numpy indexing
    def check_split_ab(self, X, y, feature, split, d):
        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]

        # need to know which weights to use!
        left_d = d[left_idx]
        right_d = d[right_idx]

        # calculate gini impurity and gain for y, left_y, right_y
        gain = self.calculate_gini_gain_ab(y, left_y, right_y, left_d, right_d, d)
        return gain, left_X, right_X, left_y, right_y, left_d, right_d

        # function to build a decision tree
    def build_tree_ab(self, X, y, depth, d):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on
        self.features_idx = np.arange(0, X.shape[1])

        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_split = None
        best_gain = 0.0
        best_left_X = None
        best_left_y = None
        best_right_X = None
        best_right_y = None
        best_left_d = None
        best_right_d = None

        # what we would predict at this node if we had to
        # weighted predictor
        # TODO: Make this work for non-binary classifier, I'm lazy and tired
        sign = np.sum(np.multiply(y, d))
        prediction = -1 if sign < 0 else 1

        # if we haven't hit the maximum depth, keep building
        if depth <= self.max_depth:
            # consider each feature
            for feature in self.features_idx:
                # consider the set of all values for that feature to split on
                possible_splits = np.unique(X[:, feature])
                for split in possible_splits:
                    # get the gain and the data on each side of the split
                    # >= split goes on right, < goes on left
                    gain, left_X, right_X, left_y, right_y, left_d, right_d = \
                        self.check_split_ab(X, y, feature, split, d)
                    # if we have a better gain, use this split and keep track of data
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split = split
                        best_left_X = left_X
                        best_right_X = right_X
                        best_left_y = left_y
                        best_right_y = right_y
                        best_left_d = left_d
                        best_right_d = right_d

        # if we haven't hit a leaf node
        # add subtrees recursively
        if best_gain > 0.0:
            left_tree = self.build_tree_ab(best_left_X, best_left_y, depth=depth + 1, d=best_left_d)
            right_tree = self.build_tree_ab(best_right_X, best_right_y, depth=depth + 1, d=best_right_d)
            return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree,
                        right_tree=right_tree)

        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)

    # take in features X and labels y
    # build a tree
    def fit(self, X, y):
        self.num_classes = len(set(y))
        self.root = self.build_tree_ab(X, y, depth=1, d=self.d)

