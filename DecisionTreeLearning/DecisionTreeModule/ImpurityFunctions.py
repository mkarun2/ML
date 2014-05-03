import math

# Calculate and return the Entropy for the data set or data subset
def entropy(data_input, target_attr):
    val_freq = {}
    data_entropy = 0.0
    data = data_input[:]
    length = len(data)
    
    # for each record in the data set
    # For each value of the target attribute, find its count/frequency
    for record in data:
        if(val_freq.has_key(record[target_attr])):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]] = 1.0

    # Calculate the entropy using the Entropy formula Entropy(D)
    for freq in val_freq.values():
        data_entropy += (-freq/length) * math.log(freq/length, 2)
        
    return data_entropy

# Information gain for attribute
def information_gain(data_input, attr, target_attr):
    val_freq = {}
    subset_entropy = 0.0
    data = data_input[:]
    
    for record in data:
        if(val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted by
    # their probability of occurring in the training set
    for val in val_freq.keys():
        val_prob        = val_freq[val] / sum(val_freq.values())
        data_subset     = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    return( entropy(data, target_attr) - subset_entropy )


# a normalized way of the information gain - information gain ratio
def gain_ratio(data_input, attr, target_attr):
    val_freq = {}
    splitinfo = 0.0
    data = data_input[:]

    for record in data:
        if(val_freq.has_key(record[attr])):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted by
    # their probability of occurring in the training set
    for val in val_freq.keys():
        val_prob        = val_freq[val] / sum(val_freq.values())
        data_subset     = [record for record in data if record[attr] == val]
        splitinfo      += (-val_prob) * math.log(val_prob, 2)

    return(information_gain(data, attr, target_attr) / splitinfo)
