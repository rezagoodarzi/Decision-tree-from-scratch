# Decision Tree Classifier

This Python code implements a decision tree classifier for a given dataset. The decision tree can be constructed using either Gini impurity or Entropy as the impurity measure for node splitting.

# Prerequisites
Before running this code, ensure you have the following dependencies installed:

pandas
numpy

You also need to have two CSV files, 'train.csv' and 'test.csv,' in the same directory as this code. These files should contain the training and test datasets, respectively.

# Decision Tree

The decision tree is printed in a tree-like format, including information about the Entropy and Gini impurity values at each node. You can also save the tree structure to a text file.

# Usage
1-open the decision tree_test folder
2-run main.py

# License
[GNU]

# Author
[REZA_Goodarzi]

# Decision Tree Entropy and gini Text File 

## Purpose
The text file serves the following purposes:

Records the hierarchy and structure of the decision tree.
Provides a visual representation of the tree's nodes and branches.
Includes additional information about Entropy and Gini impurity values at each node.
Serves as documentation for the decision tree model.

## File Format
The text file is created in a human-readable, tree-like format. It represents the hierarchy of nodes and branches in the decision tree. Each node is identified by its feature, and branches represent possible feature values.

Additional information is included for each node, such as Entropy and Gini impurity values. This information helps users understand the quality of the splits at each node.

## Usage
The text file is generated when you run the save_and_print_tree function in your code. It saves the decision tree structure in the same tree-like format that is printed to the console.

Users can access and read the text file using text editors or programming tools, allowing for visual inspection of the decision tree structure.

## Accessibility
The text file is accessible for viewing and sharing. It can be used for the following purposes:

Documenting the decision tree model.
Sharing the model's structure and criteria with colleagues or collaborators.
Providing a clear visual representation of how the model makes decisions.

## Example
Here's an example of how a node in the text file is represented:


Root Node (Entropy: 0.826, Gini Impurity: 0.497)
   |-- Feature A: Value 1
   |   |-- Leaf Node (Entropy: 0.000, Gini Impurity: 0.000)
   |   |-- Leaf Node (Entropy: 0.811, Gini Impurity: 0.480)
   |
   |-- Feature B: Value 2
   |   |-- Leaf Node (Entropy: 0.918, Gini Impurity: 0.512)
   |   |-- Leaf Node (Entropy: 0.562, Gini Impurity: 0.320)

## Conclusion
The text file is a valuable component of the code, helping to document, understand, and share the decision tree model's structure. It aids in explaining how the model makes decisions and is a useful resource for collaborators and users of the code.

# Function Explanations decision tree_test

## calculate_gini_Index(node_data):

This function calculates the Gini Index for a given set of data samples.

## GINI_imourity(data, target_column, target):

This function calculates the Gini impurity for a specific column in the dataset. It measures the impurity of the data in that column with respect to the target variable. It is used to evaluate the impurity of potential splits when building the decision tree.


## calculate_entropy(data, column):

This function calculates the entropy for a specific column in the dataset. Entropy is another criterion used in decision tree algorithms to measure the impurity of data. It's used to determine information gain when making decisions on how to split the data.


## information_gain(data, feature, target_column):

This function calculates the information gain when splitting the data based on a specific feature. Information gain measures the reduction in entropy when a dataset is split on a particular feature. It is used to select the best feature for splitting the data.


## choose_best_feature(data, target_name):

This function selects the feature that minimizes Gini impurity. It iterates through the columns of the dataset and chooses the one with the lowest Gini impurity as the best feature for making a split in the decision tree.


## build_tree_imourity(data, target=None):

This is a recursive function that builds a decision tree based on Gini impurity. It selects the best feature for the root node, then recursively creates branches for each value of that feature. It continues this process until the tree is fully built.


## build_tree_Entropy(data, target=None):

This is a recursive function that builds a decision tree based on Entropy. It is similar to the build_tree_imourity function but uses Entropy as the criterion for feature selection and node splitting.


## predict(tree, data_point):

This function takes a decision tree and a data point as input and predicts the target variable for that data point by traversing the decision tree based on the feature values of the data point.


## print_tree(tree, indent=""):

This function prints the decision tree in a tree-like format with additional information about Entropy and Gini impurity values at each node. It helps visualize the decision tree structure.


## save_and_print_tree(tree, filename, indent=""):

This function combines the functionality of printing the tree structure and saving it to a text file. It prints the tree with additional information about Entropy and Gini impurity and saves the same structure to a text file with the specified filename.

# Clean_decisiontree_dataset File 

This file cleans the data
## split the distance.py
- using mean-max scaling This technique scales your data linearly to a specific range.

## clean the file.py
This code reads a CSV file containing airplane data, splits it into two separate DataFrames based on passenger satisfaction, and then saves these split DataFrames as "satisfied.csv" and "dissatisfied.csv" without including the index column.

## main.py
This code reads two CSV files, "satisfied.csv" and "dissatisfied.csv," each containing airplane passenger data. It then creates two test sets (test1 and test2) by selecting the first 15,000 rows of data from each category. Subsequently, it generates two training sets (train1 and train2) by selecting rows 15,000 to 16,000 from each category. Finally, it combines the test and training sets for both categories and saves them as "test.csv" and "train.csv," respectively, without including the index column. This approach is often used for data splitting in machine learning applications.


# Comparison of Gini method and entropy
One of the advantages of the Gini index algorithm is that it is faster because we use a logarithm in the entropy algorithm and it increases the time complexity of the algorithm.

According to the comparisons made between the accuracy of two algorithms:
- Accuracy with impurity: 0.912
- Accuracy with Entropy: 0.919

We come to the conclusion that the Gini algorithm and entropy are not much different, but the Gini index algorithm performs better in terms of time complexity.
But in less datasets, the entropy algorithm outputs more accurate results, and entropy may even provide a much better result.

but The Gini index algorithm often performs better in terms of time complexity compared to other impurity measures, such as the Entropy-based approach, for several reasons:
## Simplicity of Calculation:
The Gini index is computationally simpler to calculate compared to the Entropy. It involves squaring probabilities, which can be faster to compute than logarithms used in the Entropy calculation. This simplicity leads to faster impurity measurements.

## No Logarithms:
The Gini index does not involve the use of logarithms, as in the Entropy calculation. Logarithms are computationally more expensive operations. The absence of logarithms in the Gini index calculation contributes to its better time performance.

## Binary Split:
In decision trees, the Gini index typically deals with binary splits (two child nodes). Binary operations are generally more efficient than multi-way splits, which are often needed with Entropy-based methods.

## Splitting Criteria:
When selecting the best feature for splitting, the Gini index may require less computational effort compared to information gain or Gain Ratio (used with Entropy). It's a simpler measure for deciding how to split data, contributing to faster tree construction.
