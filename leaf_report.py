"""
This module implements a function to extract information for leaf report on a decision tree model.

Author: Vedran Skrinjar
Date: January 8, 2021
"""

from typing import Tuple
import numpy as np
import pandas as pd

def get_leaf_report(tree:"sklearn.DecisionTreeClassifier", features:"pandas.DataFrame", target:"pandas.Series")->Tuple[float, list]:
    """
    Runs through the tree structure in order to calculate leaf probabilities and decision paths leading to each leaf node.

    Args:
        tree: sklearn decision tree model
        features: pandas dataframe of predictive variables
        target: pandas series of target labels
    
    Returns:
        Overall target rate and a list with leaf report. Leaf report is a list with one dict per leaf node.
        Each dict contains:
            * "target_prob" - share of target rows in the leaf node
            * "total_rows" - total number of rows in the leaf node
            * "share_rows" - share of all rows in the dataset that ended up in the leaf node
            * "decision_path" - a nested list of splitting conditions to reach the leaf from the root
    """

    # save number of nodes, list of left/right children of each node, and split feature names and thresholds
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    # we want to define var feature as a list of column names. we have to accomodate "None" for leaf nodes 
    #(the fact that leaf nodes have no corresponding splitting features is represented by index -2 in tree)
    #(tree.tree_.feature is a list of length n_nodes whose entries are indices of features on which corresponding nodes split data)
    feature_dict = {i: features.columns[i] for i in range(len(features.columns))}
    feature_dict.update({-2:"None"})
    feature = [tree.tree_.feature]
    feature = [feature_dict[x] for x in feature[0]]
    threshold = tree.tree_.threshold

    # run through the tree. for each node we will track:
    # 1) its depth, 2) whether it is a leaf or a test node, and 3) its parent
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaf = np.zeros(shape=n_nodes, dtype=bool)
    parent = np.full(n_nodes, -1)
    # seed is the root node id and its parent depth
    stack = [(0, -1)]  
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        if children_left[node_id] == children_right[node_id]:
            is_leaf[node_id] = True
        else:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
            parent[children_left[node_id]] = node_id
            parent[children_right[node_id]] = node_id
            
    # get leaf probabilities - the share of target rows as percentage of all rows that ended up in given node
    # target shape is one-leaf-per-row, with columns: 1) # target rows, 2) # non-target, 3) share of target
    leaf_pointers = tree.apply(features) # a list of leaf node assignments for each row in the dataset
    leaf_df = pd.DataFrame(leaf_pointers)
    leaf_df["target"] = target.values
    leaf_df["count"] = 1
    leaf_df.columns=["leaf_id", "target", "count"]
    leaf_df = leaf_df.groupby(["leaf_id", "target"]).agg("count").reset_index(level=[0,1])
    #pivot the table so that leaf_id is index, target values 0 and 1 are new columns, and each entry has value equal to count 
    leaf_df = leaf_df.pivot("leaf_id", "target", "count").fillna(0)
    leaf_df["prob"] = leaf_df.iloc[:,1]/(leaf_df.iloc[:,0]+leaf_df.iloc[:,1])
    leaf_df["non-tgt"] = 1-leaf_df["prob"]

    # decision_paths is a nested list. given x, decision_paths[x][n] will contain: n=0: the id of the leaf node,
    # n=1: conditions that have to be satisfied in leaf's parent node in order to reach the leaf,
    # n>1: conditions that have to be met in order to move to the lower level in the tree in the direction of the leaf
    decision_paths = []
    leaf_number = -1
    for i in range(n_nodes):
        if is_leaf[i]:
            leaf_number+=1
            decision_paths.append([])
            n=i
            p=parent[i]
            decision_paths[leaf_number].append(n)
            while p>=0:
                if n==children_left[p]:
                    decision_paths[leaf_number].append([feature[p],"<=", round(threshold[p],2)])
                elif n==children_right[p]:
                    decision_paths[leaf_number].append([feature[p],">", round(threshold[p],2)])
                n=p
                p=parent[n]
                
    #prepare variables for leaf report
    target_rate = round((target==1).sum()/features.shape[0]*100,2)
    leaf_report = []
    # write information about leaf nodes
    for i in range(len(decision_paths)):
        leaf_path = decision_paths[i][-1:0:-1]
        leaf_prob = 100*leaf_df.loc[decision_paths[i][0], "prob"]
        leaf_row_num = int(leaf_df.loc[decision_paths[i][0], 1]+leaf_df.loc[decision_paths[i][0], 0])
        leaf_row_share = leaf_row_num/features.shape[0]*100        
        leaf_report.append({"target_prob": round(leaf_prob,2), 
                            "total_rows": leaf_row_num, 
                            "share_rows": round(leaf_row_share,2),
                            "decision_path": leaf_path})
    return target_rate, leaf_report
    
    
