#Data Reading 
from tqdm import tqdm
import random
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def extract_features(X):
    features = {}
    X_split = X.split(' ')
    
    # Count the number of "good words" and "bad words" in the text
    good_words = ['love', 'good']
    bad_words = ['hate', 'bad']
    for x in X_split:
        
        if x in good_words:
            features['good_word_count'] = features.get('good_word_count', 0) + 1
        if x in bad_words:
            features['bad_word_count'] = features.get('bad_word_count', 0) + 1
    
    # The "bias" value is always one, to allow us to assign a "default" score to the text
    features['bias'] = 1
    return features

def read_XY_data(filename):
    X_data = []
    Y_data = []
    with open(filename, 'r') as f:
        for line in f:
            label, text = line.strip().split(' ||| ')
            X_data.append(text)
            Y_data.append(int(label))
    return X_data, Y_data

def run_classifier(X,feature_weights):
    score = 0
    for feat_name, feat_value in extract_features(X).items():
        score = score + feat_value * feature_weights.get(feat_name, 0)
    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0
def calculate_accuracy(X_data, Y_data,feature_weights):
    total_number = 0
    correct_number = 0
    for X, Y in zip(X_data, Y_data):
        Y_pred = run_classifier(X,feature_weights)
        total_number += 1
        if Y == Y_pred:
            correct_number += 1
    return correct_number / float(total_number)
    
def find_errors(X_data,Y_data,feature_weights):
    error_ids=[]
    y_preds=[]
    for i ,(x,y) in enumerate(zip(X_data,Y_data)):
        y_preds.append(run_classifier(x,feature_weights))
        if y!=y_preds[-1]:
            error_ids.append(i)
    for _ in range(5):
        my_id=random.choice(error_ids)
        x,y,y_pred=X_data[my_id],Y_data[my_id],y_preds[my_id]
        print(f'{x}\ntrue label: {y}\npredicted label: {y_pred}\n')

def main():
    #w
    feature_weights={'good_words_count':1.0,
                    'bad_words_count':-1.0,
                    'bias':0.0}
    #h
    x_train,y_train=read_XY_data('data/sst-sentiment-text-threeclass/train.txt')
    x_test,y_test=read_XY_data('data/sst-sentiment-text-threeclass/dev.txt')
    print(f'\nLength of X_train:{len(x_train)},X_test {len(x_test)}\n')
    #s=w.h
    train_accuracy = calculate_accuracy(x_train, y_train,feature_weights)
    test_accuracy = calculate_accuracy(x_test, y_test,feature_weights)
    
    print(f'Train accuracy: {train_accuracy}')
    print(f'Dev/test accuracy: {test_accuracy}')
    
    find_errors(x_train, y_train,feature_weights)
    

    

    
if __name__ == "__main__":
    main()