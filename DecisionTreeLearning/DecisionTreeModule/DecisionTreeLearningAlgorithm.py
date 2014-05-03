from ImpurityFunctions import gain_ratio

data_input_array = list()   # list of dictionaries to store the input data set
attributes_array = list()   # list to store the attributes in consideration
data_set_input_path = '../input.txt'


#From the data set, get the majority class value 
# used for the stopping criterion where the leaf is the value 
# of the majority class of the data subset
def majority_value(data, target_attr):
    dic = {}
    max_class_value = ""
    for record in data:
        dic[record[target_attr]] = dic.get(record[target_attr], 0) + 1
    counts = [(j,i) for i,j in dic.items()]
    count_max_value, max_class_value = max(counts)
    del dic
    return max_class_value

# Choose the best attribute with the highest information gain
# for splitting the node at the current level in
# the decision tree
def choose_attribute(data_input, attributes, target_attr, information_gain_ratio_func):
    data = data_input[:]
    best_gain = 0.0
    best_attr = None
    
    # target is the final classification result attribute whose
    # value is going to be the leaf of the tree
    for attr in attributes:
        if attr != target_attr:
            gain = information_gain_ratio_func(data, attr, target_attr) 
            if gain > best_gain:
                best_gain = gain
                best_attr = attr

    return best_attr

#Get the subset of the data set with respect to
# the attribute best with value val
# for the information gain entropy calculation
def get_subset( data_input, best, val):
    data = data_input[:]
    subset_list = []

    if not data:
        return subset_list
    else:
        for record in data:
            if record[best] == val:
                subset_list.append(record)
        return subset_list
    

# Decision Tree Algorithm
def create_decision_tree(data_input, attributes, target_attr, information_gain_ratio):
    data    = data_input[:]
    vals    = [record[target_attr] for record in data]
    default = majority_value(data, target_attr) # majority classification value

    # data set or attributes is empty, return the default value. 
    if not data or (len(attributes) - 1) <= 0:
        return default
    
    # All records have the same classification value, return the
    # classification value
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    
    else:
        # best splitting attribute
        best_attr = choose_attribute(data, attributes, target_attr, information_gain_ratio)
        
        # Create a new tree node with the best attribute
        tree = {best_attr:{}}

        #  generate a list containing the different values
        # of the chosen best attribute without duplicates
        unique_data = []
        for record in data:
            if unique_data.count(record[best_attr]) <= 0:
                unique_data.append(record[best_attr])

        # Create a new decision tree for each of the values in the best attribute
        for val in unique_data:
            # recursively create a subtree for the current value of the best attribute
            subtree = create_decision_tree(
                    get_subset(data, best_attr, val),     # data subset with best attribute containing val  
                    [attr for attr in attributes if attr != best_attr], # remove chosen attribute from attribute_array
                    target_attr,    # target classification attribute
                    information_gain_ratio) # impurity function to use

            # Add new subtree to the empty dictionary
            tree[best_attr][val] = subtree

    return tree

# load the training data set
def populateDataSetIntoMemory():
    first = False
    for line in open(data_set_input_path,'r'):  # read mode
        # read the attributes which is the first line in data set
        if not first: 
            first = True
            attribute_split = line.split(" ")
            for attr in attribute_split:
                attributes_array.append(attr.replace("\n",""))  # populate the attributes
            continue
        
        # read the data
        splits = line.split(" ")
        tmp = dict()
        i = 0   # value index in the array of values
        for attr in attributes_array:
            tmp[attr] = splits[i].replace("\n","")
            i = i+1
        data_input_array.append(tmp)    # populate the record set

#print the decision tree 
def print_tree(tree, string):
    if type(tree) == dict:
        print "%s%s" % (string, tree.keys()[0])
        for item in tree.values()[0].keys():
            print "%s\t%s" % (string, item)
            print_tree(tree.values()[0][item], string + "\t")
    else:
        print "%s\t->\t%s" % (string, tree)

if __name__ == "__main__":
    populateDataSetIntoMemory()
    tree = create_decision_tree( data_input_array, attributes_array, attributes_array[4], gain_ratio )
    print_tree(tree, "")