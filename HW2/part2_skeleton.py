import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random


""" 
    Helper functions
    (You can define your helper functions here.)
"""


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    df = pd.read_csv(filename, sep=',', header = 0)
    return df


### HELPERS END ###


''' Functions to implement '''

# TODO: Implement this function!
def get_histogram(dataset, chosen_anime_id="199"):
    df = dataset
    df = df[chosen_anime_id]
    valCounts = df.value_counts().sort_index()
    xValues = list(valCounts.index.astype(int))
    xValues = list(range(min(xValues), max(xValues)+2))
    plt.ylabel("Counts")
    plt.title("Rating Counts for Anime id=" + chosen_anime_id)
    plt.hist(df, xValues, align="left")
    plt.show()
    return valCounts


# TODO: Implement this function!
def get_dp_histogram(counts, epsilon: float):
    sensitivity = 2
    xValues = list(counts.index.astype(int))
    xValues = list(range(min(xValues), max(xValues)+1))
    missingNums = list(set(xValues) - set(counts.index))
    counts = list(counts)
    for i in missingNums:
        counts.insert(i+1, 0)
    noise = np.random.laplace(0, sensitivity/epsilon, len(counts))
    noisy_counts = counts + noise
    #print(counts, noise, noisy_counts)

    plt.bar(xValues, noisy_counts)
    #plt.show()
    return noisy_counts


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    xValues = list(actual_hist.index.astype(int))
    xValues = list(range(min(xValues), max(xValues)+1))
    missingNums = list(set(xValues) - set(actual_hist.index))
    actual_hist = list(actual_hist)
    for i in missingNums:
        actual_hist.insert(i+1, 0)
    total = 0
    for i in range(len(actual_hist)):
        total += abs(actual_hist[i] - noisy_hist[i])
    return total/len(actual_hist)


# TODO: Implement this function! 
def calculate_mean_squared_error(actual_hist, noisy_hist):
    xValues = list(actual_hist.index.astype(int))
    xValues = list(range(min(xValues), max(xValues)+1))
    missingNums = list(set(xValues) - set(actual_hist.index))
    actual_hist = list(actual_hist)
    for i in missingNums:
        actual_hist.insert(i+1, 0)
    total = 0
    for i in range(len(actual_hist)):
        total += (actual_hist[i] - noisy_hist[i])**2
    return total/len(actual_hist)


# TODO: Implement this function!
def epsilon_experiment(counts, eps_values: list):
    avg_list = []
    mse_list = []
    for e in eps_values:
        avg_total = 0
        mse_total = 0
        for i in range(40):
            noisy_counts = get_dp_histogram(counts, e)
            avg_total += calculate_average_error(counts, noisy_counts)
            mse_total += calculate_mean_squared_error(counts, noisy_counts)
        avg_list.append(avg_total/40)
        mse_list.append(mse_total/40)
    return avg_list, mse_list
    


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def most_10rated_exponential(dataset, epsilon):
    sensitivity = 1
    r_values = list(dataset.columns)[1:]
    counts = []
    probabilities = []
    for i in r_values:
        counts.append(len(dataset[dataset[i]==10]))
    total = 0
    for i in range(len(counts)):
        total += np.exp((epsilon*counts[i])/(2*sensitivity))
    for i in range(len(counts)):
        probabilities.append(np.exp((epsilon*counts[i])/(2*sensitivity))/total)
    r_star = random.choices(r_values, weights=probabilities, k=1)[0]
    return r_star


# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    accuracy_list = []
    counts = []
    for i in dataset:
        if i != 'user_id':
            counts.append((i,len(dataset[dataset[i]==10])))
    correct_answer = max(counts, key=lambda x: x[1])[0]
    for e in eps_values:
        correct_pred = 0
        false_pred = 0
        for i in range(1000):
            if most_10rated_exponential(dataset, e) == correct_answer:
                correct_pred += 1
            else:
                false_pred += 1
        accuracy_list.append(correct_pred/(correct_pred+false_pred))
    return accuracy_list


# FUNCTIONS TO IMPLEMENT END #

def main():
    filename = "anime-dp.csv"
    dataset = read_dataset(filename)

    counts = get_histogram(dataset)


    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg, error_mse = epsilon_experiment(counts, eps_values)
    print("**** AVERAGE ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])
    print("**** MEAN SQUARED ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_mse[i])

    print ("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    exponential_experiment_result = exponential_experiment(dataset, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])


if __name__ == "__main__":
    main()

