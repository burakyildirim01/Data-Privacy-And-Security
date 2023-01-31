import math, random
import matplotlib.pyplot as plt
from copy import deepcopy

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

""" Helpers """


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    d = len(DOMAIN)
    p = (math.exp(epsilon)/(math.exp(epsilon)+d-1))
    q = (1-p)/(d-1)

    perturbed_val = 0


    coin = random.random()
    if coin <= p:
        perturbed_val = val
    else:
        tempDomain = deepcopy(DOMAIN)
        tempDomain.remove(val)
        perturbed_val = random.choice(tempDomain)
    return perturbed_val


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    estimation_list = []
    for i,c in enumerate(DOMAIN):
        v = c
        nv = perturbed_values.count(c)
        n = len(perturbed_values)
        p = (math.exp(epsilon)/(math.exp(epsilon)+len(DOMAIN)-1))
        q = (1-p)/(len(DOMAIN)-1)
        #print(perturbed_values)
        

        Iv = nv * p + (n - nv) * q
        estimation_list.append((Iv - (n*q)) / (p-q))
        #print(i)
    return estimation_list


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    total = 0
    perturbed_values = []
    for i,val in enumerate(dataset):
        perturbed_values.append(perturb_grr(val, epsilon))
        #print(i)
    estimation = estimate_grr(perturbed_values, epsilon)
    for c in range(len(DOMAIN)):
        actual = dataset.count(DOMAIN[c])
        estimated = estimation[c]
        #print(actual,estimated)
        total += abs(actual-estimated)
    return total/len(DOMAIN)


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    encoded = [0] * len(DOMAIN)
    encoded[val-1] = 1
    return encoded


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    p = (math.exp(epsilon/2))/(math.exp(epsilon/2)+1)
    q = 1/(math.exp(epsilon/2)+1)
    perturbed_val = []
    for i in range(len(encoded_val)):
        coin = random.random()
        if coin <= p:
            perturbed_val.append(encoded_val[i])
        else:
            perturbed_val.append(1-encoded_val[i])
    return perturbed_val


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    length = len(perturbed_values)
    perturbed_values = [sum(x) for x in zip(*perturbed_values)]
    estimation_list = []
    for val in range(len(perturbed_values)):
        nv = perturbed_values[val]
        n = length
        p = (math.exp(epsilon/2))/(math.exp(epsilon/2)+1)
        q = 1/(math.exp(epsilon/2)+1)
        
        Iv = nv * p + (n - nv) * q
        estimation_list.append((Iv - (n*q)) / (p-q))
    return estimation_list


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    total = 0
    perturbed_values = []
    encoded_values = []
    for i in dataset:
        encoded_values.append(encode_rappor(i))
    correct_answer = [sum(x) for x in zip(*encoded_values)]
    for i in dataset:
        perturbed_values.append(perturb_rappor(encode_rappor(i), epsilon))
    
    estimation = estimate_rappor(perturbed_values, epsilon)

    for c in range(len(DOMAIN)):
        actual = correct_answer[c]
        estimated = estimation[c]
        #print(correct_answer[c], estimation[c])
        total += abs(actual-estimated)
    return total/len(DOMAIN)


# OUE

# TODO: Implement this function!
def encode_oue(val):
    encoded = [0] * len(DOMAIN)
    encoded[val-1] = 1
    return encoded


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    p = 1/2
    q = 1/(math.exp(epsilon)+1)
    
    perturbed_val = []
    for i in range(len(encoded_val)):
        coin = random.random()
        if encoded_val[i] == 0:
            if coin <= q:
                perturbed_val.append(1-encoded_val[i])
            else:
                perturbed_val.append(encoded_val[i])
        if encoded_val[i] == 1:
            if coin <= p:
                perturbed_val.append(encoded_val[i])
            else:
                perturbed_val.append(1-encoded_val[i])
    return perturbed_val


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    estimation_list = []
    
    c_hat = [sum(x) for x in zip(*perturbed_values)]



    for c in range(len(DOMAIN)):
        n = len(perturbed_values)
        p = 1/2
        q = 1/(math.exp(epsilon)+1)
        


        estimator =  2 * ( ( ((math.exp(epsilon) + 1) * c_hat[c] - n)) / (math.exp(epsilon) - 1))
        estimation_list.append(estimator)
    return estimation_list


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    total = 0
    perturbed_values = []
    encoded_values = []
    for i in dataset:
        encoded_values.append(encode_oue(i))
        
    correct_answer = [sum(x) for x in zip(*encoded_values)]
    for i in dataset:
        perturbed_values.append(perturb_oue(encode_oue(i), epsilon))

    estimation = estimate_oue(perturbed_values, epsilon)

    for c in range(len(DOMAIN)):
        actual = correct_answer[c]
        estimated = estimation[c]
        total += abs(actual-estimated)
    return total/len(DOMAIN)


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")
    
    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)
    

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    
    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))


if __name__ == "__main__":
    main()

