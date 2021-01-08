from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from leaf_report import get_leaf_report

def make_chessboard(N=1000,
                    xbins=(0.,0.5,1.),
                    ybins=(0.,0.5,1.)):
    """Chessboard pattern data"""
    X = np.random.uniform(size=(N,2))
    xcategory = np.digitize(X[:,0], xbins)%2
    ycategory = np.digitize(X[:,1], ybins)%2
    y = np.logical_xor(xcategory, ycategory)
    y = np.where(y, 0, 1)
    return X,y

# prepare data
X, y = make_chessboard(N=500)
X = pd.DataFrame(X)
X.columns = ["var1", "var2"]
y = pd.Series(y)

# prepare jinja
env = Environment(  loader=FileSystemLoader("."),
                    autoescape=select_autoescape(["html"])
                )
template = env.get_template("./leaf_report_template.html")

# set model parameters
max_depth = 4
min_samples_split=10
min_samples_leaf=5
min_impurity_decrease=0

# fit simple tree
simple_tree = DecisionTreeClassifier(max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    min_impurity_decrease=min_impurity_decrease)
simple_tree.fit(X, y)

# prepare leaf report
target_rate, leaf_report = get_leaf_report(simple_tree, X, y)
data = {
        "target_rate": target_rate,
        "leaf_report": leaf_report
        }
with open("./leaf_report.html", "w") as report:
    report.write(template.render(data))