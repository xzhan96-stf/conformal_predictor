import csv
import sys

def calculate_pvalue(train_alpha, test_alpha_y):
    '''
    Compute the pvalue by making calibration with the distribution of the training non-conformity measurement
    '''
    cnt = 0
    for i in range(len(train_alpha)):
        if test_alpha_y <= train_alpha[i]: # training observation has a higher non-conformity measurement than the current (test sample, label)
            cnt += 1
    return (cnt + 1)/(len(train_alpha) + 1)
        
args = sys.argv[1:]
if len(args) == 3: # the training probablities, test probabilities, number of labels
    train_filename = str(args[0]) # training data CSV: first column: predicted label, the other columns: the predicted probabilities for each label
    test_filename = str(args[1]) # test data CSV: first column: predicted label, the other columns: the predicted probabilities for each label
    D = int(args[2])

n_labels = D # replace D with the number of possible labels

# Read in the train_csv file
train_data = []
with open(train_filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        train_data.append(row)

# Iterate over each row and calculate the non-conformity measurement, store it as a list
train_alpha = []
for i in range(len(train_data)):
    label = int(train_data[i][0]) # extract the ground-truth label
    probs = [float(x) for x in train_data[i][1:]] # extract the probabilities
    assert len(probs) == D
    gt_prob = probs[label] # extract the ground-truth label's probability
    max_prob = max([probs[j] for j in range(n_labels) if j != label]) # extract the largest probability excluding the ground-truth label's probability
    train_alpha.append(0.5 - 0.5 * (gt_prob - max_prob)) # calculate the non-conformity measurement to be calibrated

    
# Read in the test_csv file
test_data = []
with open(test_filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        test_data.append(row)

# Iterate over each row and calculate the non-conformity measurement, pvalue and the credibility and confidence for each test case
credibility = []
confidence = []
for i in range(len(test_data)):
    test_label = int(test_data[i][0]) # extract the predicted label
    probs = [float(x) for x in test_data[i][1:]] # extract the probabilities
    
    p_value = [] # record the p-value for each possible label
    for y in range(n_labels): # try each possible label
        y_prob = probs[y] # extract the current trial label's probability
        max_prob = max([probs[j] for j in range(n_labels) if j != y]) 
        y_alpha = 0.5 - 0.5 * (y_prob - max_prob) # compute the non-conformity of the trial label
        p_value.append(calculate_pvalue(train_alpha, y_alpha)) # calibrate to compute the p-value for the trial label
    assert len(p_value) == D
    credibility.append(p_value[test_label])
    del p_value[test_label]
    confidence.append(1 - max(p_value))

# Output the results in a csv file
fields = ['Credibility', 'Confidence']
results = [[str(round(credibility[i],4)), str(round(confidence[i],4))] for i in range(len(test_data))]
with open('Conformal Prediction Output.csv', 'w', newline='') as f:
    # creating a csv writer object 
    csvwriter = csv.writer(f) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(results)