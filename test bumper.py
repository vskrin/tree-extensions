import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

#import custom class
import bumper

def make_chessboard(N=1000,
                    xbins=(0.,0.5,1.),
                    ybins=(0.,0.5,1.)):
    """Chessboard pattern data"""
    X = np.random.uniform(size=(N,2))
    xcategory = np.digitize(X[:,0], xbins)%2
    ycategory = np.digitize(X[:,1], ybins)%2
    y = np.logical_xor(xcategory, ycategory)
    y = np.where(y, -1., 1.)
    return X,y

def plot_data(X,y, is_bumper=False):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=y)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    if is_bumper:
        ax.set_title("Bumper results")
    else:
        ax.set_title("Simple tree results")
    return (fig, ax)

def draw_decision_regions(X, y, estimator, resolution=0.01, is_bumper=False):
    """Draw samples and decision regions
    
    The true label is indicated by the colour of each
    marker. The decision tree's predicted label is
    shown by the colour of each region.
    """
    plot_step = resolution
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, axis = plot_data(X,y, is_bumper=is_bumper)
    axis.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdBu)
    plt.show()

# generate problem (data)
X_, y_ = make_chessboard(N=1000)
p = plot_data(X_, y_) #uncomment to see scatterplot without decision boundary


#set model parameters
max_depth = 2
min_samples_split=50
min_samples_leaf=25
min_impurity_decrease=0

#
# EVALUATE SIMPLE TREE
#
#fit simple tree
simple_tree = DecisionTreeClassifier(max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    min_impurity_decrease=min_impurity_decrease)
simple_tree.fit(X_, y_)
#plot decision region
with sns.axes_style('white'):
    draw_decision_regions(X_, y_, simple_tree, is_bumper=False)
#score simple tree
score, _, _ = bumper.score_model(model=simple_tree, 
                                features=X_, 
                                target=y_)
print("Simple tree score:\n", score)


#
# EVALUATE BUMPER
#
#fit bumper
bumped_tree = bumper.Bumper(n_bumps=100,
                            scoring_metric='rec',
                            max_depth=max_depth, 
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            min_impurity_decrease=min_impurity_decrease)
bumped_tree.fit(X_, y_)
#plot decision region
with sns.axes_style('white'):
    draw_decision_regions(X_, y_, bumped_tree.best_estimator_, is_bumper=True)
# score bumper
score, _, _ = bumper.score_model(model=bumped_tree.best_estimator_, 
                                features=X_, 
                                target=y_)
print("Bumper tree score:\n", score)
#plot decision path
plot_tree(bumped_tree.best_estimator_ ,  
            max_depth=3, 
            feature_names=["feat 1", "feat 2"], 
            class_names=["non-tgt", "tgt"], 
            rounded=True
            )
plt.show()
