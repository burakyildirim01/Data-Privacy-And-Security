import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


###############################################################################
############################### Label Flipping ################################
###############################################################################

def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    total = 0
    for i in range(100):
        y_train_temp = copy.deepcopy(y_train)
        y_train_df = pd.DataFrame(y_train_temp)
        selected_rows = y_train_df.sample(frac = n)
        indexes = selected_rows.index
        for i in y_train_df.index:
            if i in indexes:
                y_train_df.loc[i] = 1 - y_train_df.loc[i]
        y_train_temp = y_train_df.values
        y_train_temp = y_train_temp.ravel()

        if model_type == "DT":
            myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
            myDEC.fit(X_train, y_train_temp)
            DEC_predict = myDEC.predict(X_test)
            total += accuracy_score(y_test, DEC_predict)
        elif model_type == "LR":
            myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
            myLR.fit(X_train, y_train_temp)
            LR_predict = myLR.predict(X_test)
            total += accuracy_score(y_test, LR_predict)
        elif model_type == "SVC":
            mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
            mySVC.fit(X_train, y_train_temp)
            SVC_predict = mySVC.predict(X_test)
            total += accuracy_score(y_test, SVC_predict)
    return total/100

###############################################################################
############################## Inference ########################################
###############################################################################

def inference_attack(trained_model, samples, t):
    # TODO: You need to implement this function!
    probs = trained_model.predict_proba(samples)
    tp = 0
    fn = 0
    for i in probs:
        if i[1] > t:
            tp += 1
        else:
            fn += 1
    return tp/(tp+fn)

###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    X_train_temp = copy.deepcopy(X_train)
    y_train_temp = copy.deepcopy(y_train)

    X_train_temp2 = copy.deepcopy(X_train)
    y_train_temp2 = copy.deepcopy(y_train)

    selected = set()
    for i in range(num_samples):
        ind = random.randint(0, len(X_train)-1)
        while ind in selected or y_train_temp[ind] == 1:
            ind = random.randint(0, len(X_train)-1)
        selected.add(ind)
        randomX = copy.deepcopy(X_train_temp[ind])
        randomX[0] = 70
        X_train_temp = np.concatenate((X_train_temp, np.array([randomX])))
        y_train_temp = np.concatenate((y_train_temp, [1]))


    if model_type == 'DT':
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == 'SVC':
        model = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
    elif model_type == 'LR':
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)

    model.fit(X_train_temp, y_train_temp)

    n = 50
    X_test = np.zeros((n, len(X_train[0])))

    selected = set()
    for i in range(n):
        ind = random.randint(0, len(X_train)-1)
        while ind in selected or y_train_temp[ind] == 1:
            ind = random.randint(0, len(X_train)-1)
        selected.add(ind)
        randomX = copy.deepcopy(X_train_temp2[ind])
        randomX[0] = 70
        X_test[i] = randomX

    success_rate = 0
    if num_samples != 0:
        success_rate = np.count_nonzero(model.predict(X_test)==1)/n
    return success_rate



###############################################################################
############################## Evasion ########################################
###############################################################################

def evade_model(trained_model, actual_example):
    # TODO: You need to implement this function!
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    maximum_list = [42,90,29,16.8,96,65.9,220.4,19,68,31.1]
    minimum_list = [22,60,8,0,28.6,0.7,6.9,0,1.1,0]

    pred_class = actual_class
    while pred_class == actual_class:
        for i in range(len(modified_example)):
            if actual_class == 0:
                if modified_example[i] < maximum_list[i] and (i==4 or i==7):
                    modified_example[i] += 0.1
                    pred_class = trained_model.predict([modified_example])[0]
            elif actual_class == 1:
                if modified_example[i] > minimum_list[i] and (i==4 or i==7):
                    modified_example[i] -= 0.1
                    pred_class = trained_model.predict([modified_example])[0]
    return modified_example

def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i]-adversarial_example[i])
        return tot/len(actual_example)

###############################################################################
############################## Transferability ################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    # TODO: You need to implement this function!
    transfer_dict = {}
    dt2lr = 0
    dt2svc = 0
    svc2lr = 0
    svc2dt = 0
    lr2dt = 0
    lr2svc = 0

    DT_adversarial_examples = [evade_model(DTmodel, i) for i in actual_examples]
    LR_adversarial_examples = [evade_model(LRmodel, i) for i in actual_examples]
    SVC_adversarial_examples = [evade_model(SVCmodel, i) for i in actual_examples]

    dt2lr += sum([DTmodel.predict([i])[0] == LRmodel.predict([i])[0] for i in DT_adversarial_examples])
    dt2svc += sum([DTmodel.predict([i])[0] == SVCmodel.predict([i])[0] for i in DT_adversarial_examples])
    svc2lr += sum([SVCmodel.predict([i])[0] == LRmodel.predict([i])[0] for i in SVC_adversarial_examples])
    svc2dt += sum([SVCmodel.predict([i])[0] == DTmodel.predict([i])[0] for i in SVC_adversarial_examples])
    lr2dt += sum([LRmodel.predict([i])[0] == DTmodel.predict([i])[0] for i in LR_adversarial_examples])
    lr2svc += sum([LRmodel.predict([i])[0] == SVCmodel.predict([i])[0] for i in LR_adversarial_examples])

    transfer_dict['DTmodel To LRmodel'] = dt2lr
    transfer_dict['DTmodel To SVCmodel'] = dt2svc
    transfer_dict['LRmodel To DTmodel'] = lr2dt
    transfer_dict['LRmodel To SVCmodel'] = lr2svc
    transfer_dict['SVCmodel To LRmodel'] = svc2lr
    transfer_dict['SVCmodel To DTmodel'] = svc2dt

    print ("Transferability Results:")
    for k in transfer_dict:
        print(k+"\t"+str(transfer_dict[k]/len(actual_examples)))


###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you have implemented your model stealing attack
    result_model = None
    if model_type == "DT":
        myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
        myDEC.fit(examples, remote_model.predict(examples))
        result_model = myDEC
    elif model_type == "LR":
        myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
        myLR.fit(examples, remote_model.predict(examples))
        result_model = myLR
    elif model_type == "SVC":
        mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
        mySVC.fit(examples, remote_model.predict(examples))
        result_model = mySVC
    return result_model


###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ##
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "forest_fires.csv"
    features = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    df["DC"] = df["DC"].astype('float64')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))



    # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]

    n_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for n in n_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
            print("Accuracy of poisoned", model_type, str(n), ":", acc)

    # Inference attacks:
    samples = X_train[0:100]
    t_values = [0.99,0.98,0.96,0.8,0.7,0.5]
    for t in t_values:
        print("Recall of inference attack", str(t), ":", inference_attack(mySVC,samples,t))

    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)

    #Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"]
    num_examples = 40
    for a,trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a] , ":" , total_perturb/num_examples)


    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])

    # Model stealing:
    budgets = [8, 12, 16, 20, 24]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))


if __name__ == "__main__":
    main()
