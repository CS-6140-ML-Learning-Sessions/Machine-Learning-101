================================================================================
  TREE-BASED ALGORITHMS — STUDY NOTEBOOK
================================================================================

  A structured walkthrough of tree-based machine learning algorithms,
  from single decision trees to powerful ensemble methods.

________________________________________________________________________________

  WHAT THIS NOTEBOOK COVERS

  This notebook explores the Tree-Based Algorithm(CART and Adaboost), one of the most
  widely used and interpretable approaches in machine learning. We cover
  both theory and implementation, organized from the ground up.

________________________________________________________________________________

  TABLE OF CONTENTS

  1. Introduction to Decision Trees
  2. CART — Classification & Regression Trees
  3. Ensemble Methods Overview
  4. Bagging
  5. Boosting
       - AdaBoost
  6. Comparison & When to Use What

________________________________________________________________________________

  ALGORITHM FAMILY MAP

  Tree-Based Algorithms
  |
  +-- Single Tree
  |     |
  |     +-- CART (Classification & Regression Trees)
  |
  +-- Ensemble Trees
        |
        +-- Bagging (sampling WITH replacement)
        |     |
        |     +-- Random Forest
        |     +-- Pasting
        |
        +-- Boosting (sequential, each tree corrects the previous)
              |
              +-- AdaBoost
              +-- Gradient Boosting
              +-- XGBoost
              +-- LightGBM

________________________________________________________________________________

  KEY CONCEPTS AT A GLANCE

  Node          A decision point that splits data based on a feature
  Leaf          Terminal node — holds the final prediction
  Depth         Number of levels in the tree
  Gini          Measures how mixed the classes are at a node
  Entropy       Alternative split criterion based on information gain
  Bagging       Trains trees in PARALLEL on random data samples
  Boosting      Trains trees SEQUENTIALLY, each fixing previous errors
  Overfitting   Deep trees memorize training data — use pruning or ensembles

________________________________________________________________________________

  PREREQUISITES

  Libraries used in this notebook:

    - numpy
    - pandas
    - matplotlib
    - sklearn  (DecisionTree, RandomForest, AdaBoost, GradientBoosting)
    - xgboost
    - lightgbm

________________________________________________________________________________

  QUICK REFERENCE

  Algorithm          Type          Parallel?   Key Strength
  ---------------------------------------------------------------
  CART               Single Tree   ---         Interpretable, fast
  Random Forest      Bagging       Yes         Reduces variance
  Pasting            Bagging       Yes         Works on large datasets
  AdaBoost           Boosting      No          Handles weak learners
  Gradient Boosting  Boosting      No          High accuracy
  XGBoost            Boosting      No          Speed + regularization
  LightGBM           Boosting      No          Very fast, low memory

________________________________________________________________________________

  TIP: Run each section independently. Every algorithm section includes
  theory, a diagram, and a hands-on sklearn example.

================================================================================
