import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

data_test = data_test.sample(frac=1).reset_index(drop=True)
data_test = data_test.drop(columns=['Unnamed: 0'])
data_test = data_test.drop(columns=['id'])

data = data.sample(frac=1).reset_index(drop=True)
data = data.drop(columns=['Unnamed: 0'])
data = data.drop(columns=['id'])


# ------------------------------------------------------------------------------------------
def calculate_gini_Index(node_data):
    total_samples = len(node_data)
    class_counts = np.unique(node_data, return_counts=True)[1]

    if total_samples == 0:
        return 0

    gini = 1 - sum((class_count / total_samples) ** 2 for class_count in class_counts)
    return gini


def GINI_imourity(data, target_column, target):
    total_samples = len(data)
    class_counts = np.unique(data[target_column], return_counts=True)[1]

    if total_samples == 0:
        return 0

    gini_sum = 0
    unique_values = data[target_column].unique()

    for value in unique_values:
        count_satisfied = len(data[(data[target_column] == value) & (data[target] == 'satisfied')])
        count_unsatisfied = len(
            data[(data[target_column] == value) & (data[target] == 'neutral or dissatisfied')])
        weight_satisfied = len(data[data[target_column] == value]) / data[target_column].count()
        leni = len(data[data[target_column] == value])
        if leni != 0:
            gini = 1
            gini -= (count_satisfied / leni) ** 2 + (count_unsatisfied / leni) ** 2
            gini_sum += weight_satisfied * gini

    return gini_sum


# ------------------------------------------------------------------------------------------
def calculate_entropy(data, column):
    unique_values, value_counts = np.unique(data[column], return_counts=True)

    probabilities = value_counts / len(data[column])
    # print(f'{column} probability : {probabilities}')
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


# -------------------------------------------------------------------------------------------
def information_gain(data, feature, target_column):
    total_entropy = calculate_entropy(data, target_column)
    unique_values = data[feature].unique()

    weighted_entropy = 0
    for value in unique_values:
        subset = data[data[feature] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * calculate_entropy(subset, target_column)

    info_gain = total_entropy - weighted_entropy
    return info_gain


# --------------------------------------------------------------------------------------------
# Choose the feature that minimizes Gini impurity
def choose_best_feature(data, target_name):
    best_feature = None
    min_gini = 100.1

    for feature in data.columns:
        if feature == target_name:
            continue

        split_gini = GINI_imourity(data, feature, target_name)
        if split_gini < min_gini:
            min_gini = split_gini
            best_feature = feature

    return best_feature


# --------------------------------------------------------------------------------------------

# Recursive function to build the decision tree
def build_tree_imourity(data, target=None):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    if len(data.columns) == 1:
        return data[target].mode()[0]

    best_feature = choose_best_feature(data, target)

    # Create the tree structure
    tree = {best_feature: {}}
    unique_values = data[best_feature].unique()

    for value in unique_values:
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = build_tree_imourity(subset, target)

    return tree


# --------------------------------------------------------------------------------------------

def build_tree_Entropy(data, target=None):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    if len(data.columns) == 1:
        return data[target].mode()[0]

    # Choose the best feature to split the data
    max_info_gain = -1
    best_feature = None
    for column in data.columns:
        if column != 'id' and column != 'Unnamed: 0' and column != target:
            target_column = column
            entropy_gain = information_gain(data, target_column, target)
            if entropy_gain > max_info_gain:
                max_info_gain = entropy_gain
                best_feature = column

    # Create the tree structure
    tree = {best_feature: {}}
    unique_values = data[best_feature].unique()

    for value in unique_values:
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = build_tree_Entropy(subset, target)

    return tree


# --------------------------------------------------------------------------------------------


def predict(tree, data_point):
    for feature, branches in tree.items():
        value = data_point[feature]
        if value in branches:
            if isinstance(branches[value], dict):
                return predict(branches[value], data_point)
            else:
                return branches[value]


# --------------------------------------------------------------------------------------------

# Build the decision tree
target = 'satisfaction'
best_feature = None
max_info_gain = -1
for column in data.columns:
    if column != 'id' and column != 'Unnamed: 0' and column != target:
        target_column = column
        gini = calculate_gini_Index(data[target_column])
        entropies = calculate_entropy(data, target_column)
        entropy_gain = information_gain(data, target_column, target)
        gini_weight = GINI_imourity(data, target_column, target)
        best_feature2 = choose_best_feature(data, target)
        if entropy_gain > max_info_gain:
            max_info_gain = entropy_gain
            best_feature = column
        print(f'{column} - GINI_impurity : {gini_weight}')
        print(f'{column} - entropy_gain : {entropy_gain}\n')
        print(f'best_feature until now : {best_feature2}')

decision_tree_imourity = build_tree_imourity(data, target)
decision_tree_Entropy = build_tree_Entropy(data, target)
print(decision_tree_imourity)
print(decision_tree_Entropy)
# Make predictions on the test data
predictions_imourity = data_test.apply(lambda row: predict(decision_tree_imourity, row), axis=1)
predictions_Entropy = data_test.apply(lambda row: predict(decision_tree_Entropy, row), axis=1)

# Calculate accuracy
correct_predictions_imourity = (predictions_imourity == data_test[target]).sum()
correct_predictions_entropy = (predictions_Entropy == data_test[target]).sum()

total_predictions = len(data_test)
accuracy_imourity = correct_predictions_imourity / total_predictions
accuracy_Entropy = correct_predictions_entropy / total_predictions

print("Accuracy with impurity:", accuracy_imourity)
print("Accuracy with Entropy:", accuracy_Entropy)


def print_tree(tree, indent=""):
    if isinstance(tree, dict):
        for feature, branches in tree.items():
            print(indent + feature + '(Entropy : ' + str(
                calculate_entropy(data, feature)) + ')' + ' ( gini_impurity : ' + str(
                GINI_imourity(data, feature, target)) + ')')
            for value, subtree in branches.items():
                print(indent + "   |-- " + str(value))
                print_tree(subtree, indent + "  ")
    else:
        print(indent + str(tree))


# Print the decision tree built using Gini impurity
print("Decision Tree (Gini Impurity):")
print_tree(decision_tree_imourity)

# Print the decision tree built using Entropy
print("\nDecision Tree (Entropy):")
print_tree(decision_tree_Entropy)


#if you want to save the tree in text file un comment the following code
'''
def save_and_print_tree(tree, filename, indent=""):
    with open(filename, 'w') as text_file:
        def write_and_print_tree(node, current_indent=""):
            if isinstance(node, dict):
                for feature, branches in node.items():
                    text_file.write(current_indent + feature + '(Entropy : ' + str(
                        calculate_entropy(data, feature)) + ')' + ' ( gini_impurity : ' + str(
                        GINI_imourity(data, feature, target)) + ')' + '\n')
                    for value, subtree in branches.items():
                        text_file.write(current_indent + "   |-- " + str(value) + '\n')
                        write_and_print_tree(subtree, current_indent + "  ")
            else:
                text_file.write(current_indent + str(node) + '\n')

        write_and_print_tree(tree, indent)


save_and_print_tree(decision_tree_imourity, 'decision_tree_gini.txt')

save_and_print_tree(decision_tree_Entropy, 'decision_tree_entropy.txt')
'''
# --------------------------------------------------------------------------------------------
