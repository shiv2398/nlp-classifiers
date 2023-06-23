import numpy as np
from scipy.sparse import dok_matrix
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
def read_XY_data(filename):
    X_data=[]
    Y_data=[]
    with open(filename,'r') as f:
        for lines in f:
            label,text=lines.strip().split(' ||| ')
            X_data.append(text)
            Y_data.append(int(label))
    return X_data,Y_data


def tokenize(datum):
    #split the string into words
    return datum.split(' ')
def build_feature_map(X):
    #we need to assign an index to each word in order to make a count vector
    #we start by gathering a st of all word types in the data
    word_types=set()
    for datum in X:
        for word in tokenize(datum):
            word_types.add(word)
    #create a dictionary keyed by word mapping it ot an index
    return {word: idx  for idx,word in enumerate(word_types)}



def extract_features(word_to_idx, X):
    # We are using a sparse matrix from scipy to avoid creating an 8000 x 18000 matrix
    features = dok_matrix((len(X), len(word_to_idx)))
    for i in range(len(X)):
        for word in tokenize(X[i]):
            if word in word_to_idx:
                # Increment the word count if it is present in the map.
                # Unknown words are discarded because we would not have
                # a learned weight for them anyway.
                features[i, word_to_idx[word]] += 1
    return features
#statistical testing for the bootstraping 

import numpy as np

rng = np.random.default_rng()


def eval_with_paired_bootstrap(gold, sys1, sys2, num_samples=10000, sample_ratio=0.5):
    """Evaluate with paired boostrap
    This compares two systems, performing a significance tests with
    paired bootstrap resampling to compare the accuracy of the two systems.

    Parameters
    ----------
    gold
      The correct labels
    sys1
      The output of system 1
    sys2
      The output of system 2
    num_samples
      The number of bootstrap samples to take
    sample_ratio
      The ratio of samples to take every time

    """
    assert len(gold) == len(sys1)
    assert len(gold) == len(sys2)

    gold = np.array(gold)
    sys1 = np.array(sys1)
    sys2 = np.array(sys2)

    sys1_scores = []
    sys2_scores = []
    wins = [0, 0, 0]
    n = len(gold)

    for _ in tqdm(range(num_samples)):
        # Subsample the gold and system outputs
        subset_idxs = rng.choice(n, int(n * sample_ratio), replace=True)
        sys1_score = (sys1[subset_idxs] == gold[subset_idxs]).mean()
        sys2_score = (sys2[subset_idxs] == gold[subset_idxs]).mean()

        if sys1_score > sys2_score:
            wins[0] += 1
        elif sys1_score < sys2_score:
            wins[1] += 1
        else:
            wins[2] += 1

        sys1_scores.append(sys1_score)
        sys2_scores.append(sys2_score)

    # Print win stats
    wins = [x / float(num_samples) for x in wins]
    print("Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f" % (wins[0], wins[1], wins[2]))
    if wins[0] > wins[1]:
        print("(sys1 is superior with p value p=%.3f)\n" % (1 - wins[0]))
    elif wins[1] > wins[0]:
        print("(sys2 is superior with p value p=%.3f)\n" % (1 - wins[1]))

    # Print system stats
    sys1_scores.sort()
    sys2_scores.sort()
    print(
        "sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]"
        % (
            np.mean(sys1_scores),
            np.median(sys1_scores),
            sys1_scores[int(num_samples * 0.025)],
            sys1_scores[int(num_samples * 0.975)],
        )
    )
    print(
        "sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]"
        % (
            np.mean(sys2_scores),
            np.median(sys2_scores),
            sys2_scores[int(num_samples * 0.025)],
            sys2_scores[int(num_samples * 0.975)],
        )
    )


def main():
    #h
    x_train,y_train=read_XY_data('/home/sahitya/Desktop/Adavanced_nlp/NLP_classifier/data/sst-sentiment-text-threeclass/train.txt')
    x_test,y_test=read_XY_data('/home/sahitya/Desktop/Adavanced_nlp/NLP_classifier/data/sst-sentiment-text-threeclass/test.txt')
    print(f'\nLength of X_train:{len(x_train)},X_test {len(x_test)}\n')
    
    
    sample_data = [
    "When is the homework due ?",
    "When are the TAs' office hours ?",
    "How hard is the homework ?",
    ]
    word_to_idx=build_feature_map(sample_data)
    features=extract_features(word_to_idx,sample_data)

    #build the map based on the training data
    word_to_idx=build_feature_map(x_train)
    print(f'unique word types in X_train :{len(word_to_idx)}')
    print('Sample words :')
    print(list(word_to_idx.keys())[:20])

    #convert our string into count vectors
    x_train_vec=extract_features(word_to_idx,x_train)
    x_test_vec=extract_features(word_to_idx,x_test)
    
    classifier=LogisticRegression(tol=1e1)
    classifier.fit(x_train_vec,y_train)
    
    #create a truncated version of the training so we have a second model to compare to
    x_train_vec_truc=extract_features(word_to_idx,[x[:100] for x in x_train])
    classifier_truc=LogisticRegression(tol=1e1)
    classifier_truc.fit(x_train_vec_truc,y_train)

    cls_preds = classifier.predict(x_test_vec)
    cls_trunc_preds = classifier_truc.predict(x_test_vec)
    baseline_preds = np.ones_like(cls_preds)

    eval_with_paired_bootstrap(y_test, cls_preds, baseline_preds)
    print()
    eval_with_paired_bootstrap(y_test, cls_preds, cls_trunc_preds)
    

if __name__ == "__main__":
    main()