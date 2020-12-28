# Some useful extensions of scikit-learn DecisionTreeClassifier

## Abstract

This set of modules contains some useful extension of the standard scikit-learn decision tree classifier.

Each extension is found in a separate Python module, and each extension has one accompanying testing module for demonstration.

The extensions are:
* "bumper" algorithm
* leaf report
* visualization of decision boundaries for two top-most splits
* visualization of decision boundaries projected on the space of two principal components (two leading PCA components)

## Bumper
Bumper algorithm is briefly explained in "The elements of statistical learning" by Hastie et al.
In particular, in 2nd edition of the book you can find it in section 8.9.

The book describes the method as follows:

```
Bumping uses bootstrap sampling to move randomly through model space. 

For problems where fitting method finds many local minima, 
bumping can help the method to avoid getting stuck in poor solutions.
```

In a few more words, what bumper algorithm does, is it uses bootstrap resampling to find better decision tree models. It achieves better performance if the original model was unable to reach optimal state due to symmetries in the data (see the reference below) or if randomness in the data led it to converge to a sub-optimal configuration (something like a local minimum).
Bumper aims to address the problems with randomness in the data by building many models on different bootstrap resamples of the data - not on the original data. To eliminate fears of building on something that's not "observed data", it helps to think of it this way: the original data may lead our model to behave worse than optimal, but we may take this data only as the empirical distribution. From this empirical distribution we build bootstrap resamples of the data distribution, and build models on top of that. 

Some examples of the situation described in the above discussion can be generated using ```test bumper.py``` module. 
Consider the following figure which displays a tree that failed to identify boundaries which are easy to spot just by looking at the distribution.
![simple two-level tree](two_level_tree.png)

Further branching out the tree doesn't help. This is a four-level tree:
![simple four-level tree](four_level_tree.png)

The reason for the failure of the two-level tree, and the reason four-level tree can't be any better are both the same - CART algorithm is a greedy algorithm and mistakes once made cannot be fixed downstream.

Two-level bumper tree is shown in the next figure, and four-level bumper looks almost the same so I omit it.

![two-level bumper tree](two_level_bumper.png)

An implementation of the bumper class can be found in the module ```bumper.py```. I wrote it by borrowing (heavily) from [this](https://betatim.github.io/posts/bumping/) resource. 
The algorithm is described very well in the above reference, so I will limit my comments here to modifications compared to the referenced code.

* Bumper class takes a few additional parameters for the DecisionTreeClassifier
* It takes an additional ```scoring_metric``` parameter which specifies model quality metric which is used to determine which bootstrap resampled (bumped?) tree is the best
* There is an additional ```score_model``` function which can be used to generate a dictionary of model scores

Note that the class cannot be written by inheriting directly from DecisionTreeClassifier, but having found the optimal model, for almost any purpose we wish to use ```best_estimator_``` only, since this is what contains our DecisionTreeClassifier model. I tend to do a deep copy of this object and discard the rest of the bumper.


## Leaf report
Shows leaf probabilities for target (non-target) class. Can be used for ....

## Decision boundaries at the tree-top
Implementation in Plotly...
Can be embedded in ...

## Decision boundaries in PCA space
Useful for qualitative estimation of tree performance ...