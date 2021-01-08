"""
This module implements a class for the decision tree "bumper" algorithm.
The code is based on a class that was taken from this site: https://betatim.github.io/posts/bumping/
Theory behind it is described in the link above, and also in chapter 8 of "Elements of statistical learning".

Author: Vedran Skrinjar
Date: December 10, 2020
"""

import numpy as np
from sklearn.base import ClassifierMixin, MetaEstimatorMixin
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, auc


class Bumper(ClassifierMixin, MetaEstimatorMixin):
    def __init__(self, 
                max_depth=2, 
                min_samples_split=10, 
                min_samples_leaf=5,
                min_impurity_decrease=.01,
                n_bumps=20,
                scoring_metric='prec',
                random_state=None):
        """
        This class implements bumper algorithm for decision tree classifier.
        It selects the best decision tree between "n_bumps" trees trained on resampled data.
        The best tree is the one with highest score on "scoring_metric". Scoring metric takes 
        values: 'acc', 'prec', 'rec', 'f1', or 'auc' corresponding to accuracy, precision, recall,
        f1-score, and area-under-curve respectively.
        """
        self.estimator = DecisionTreeClassifier(max_depth=max_depth, 
                                                min_samples_split=min_samples_split, 
                                                min_samples_leaf=min_samples_leaf, 
                                                min_impurity_decrease=min_impurity_decrease
                                                )
        self.n_bumps = n_bumps
        self.scoring_metric = scoring_metric
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        self.best_estimator_ = None
        best_score = 0
        best_estimator = None
        
        for n in range(self.n_bumps):
            indices = random_state.randint(0, n_samples, n_samples)
            
            estimator = clone(self.estimator)
            estimator.fit(X[indices], y[indices])
          
            # performance is measured on all samples
            score, _, _ = score_model(model=estimator, features=X, target=y)
            score = score[self.scoring_metric]
            if score > best_score:
                best_score = score
                best_estimator = estimator

        self.best_estimator_ = best_estimator
        return self
    
    def predict(self, X):
        return self.best_estimator_.predict(self.best_estimator_)


def score_model(*,model:"DecisionTreeClassifier", features:"numpy.array", target:"numpy.array")->dict:
    """
    Args:
        model: scikit-learn decision tree model
        features: an array of features
        target: 1d array of target labels.
    Returns:
        Scoring dictionary of model quality metrics and corresponding values.
        It also returns false positive rate and true positive rate which may be used for plotting ROC curves.
    """
    predicted_labels = model.predict(features)
    predicted_proba = model.predict_proba(features)[:,1]
    nrows = len(predicted_labels)
    is_target = target==1
    is_predicted_target = (predicted_proba>=.5)
    true_positive = np.sum(np.logical_and(is_target, is_predicted_target))/nrows
    true_negative = np.sum(np.logical_and(~is_target, ~is_predicted_target))/nrows
    false_positive = np.sum(np.logical_and(~is_target, is_predicted_target))/nrows
    false_negative = np.sum(np.logical_and(is_target, ~is_predicted_target))/nrows
    fpr, tpr, _ = roc_curve(target, predicted_proba)
    scoring_dict = ({
        "prec": np.round(100*precision_score(target, predicted_labels),2),
        "acc": np.round(100*accuracy_score(target, predicted_labels),2),
        "rec": np.round(100*recall_score(target, predicted_labels),2),
        "f1": np.round(100*f1_score(target, predicted_labels),2),
        "auc": np.round(100*auc(fpr,tpr),2),
        "True positive share": np.round(100*true_positive,2),
        "True negative share": np.round(100*true_negative,2),
        "False positive share": np.round(100*false_positive,2),
        "False negative share": np.round(100*false_negative,2),
    })
    return scoring_dict, fpr, tpr