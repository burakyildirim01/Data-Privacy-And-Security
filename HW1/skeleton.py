##############################################################################
# This skeleton was created by Efehan Guner  (efehanguner21@ku.edu.tr)       #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
from enum import unique
import glob
import os
import sys
from copy import deepcopy
import numpy as np
import datetime

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def getEquals(current_list, fields):
    equal_list = []
    for person1 in range(len(current_list)):
        for person2 in range(person1+1, len(current_list)):
            new_p1 = {k: current_list[person1][k] for k in fields}
            new_p2 = {k: current_list[person2][k] for k in fields}
            if (new_p1 == new_p2):
                equal_list.append(True)
            else:
                equal_list.append(False)
    return equal_list

def moveUp(DGHs, field, person, fields):
    if field in fields:
        flag = 0
        index = DGHs[field][0].index(person[field])
        currentLocation = DGHs[field][1][index]
        for j in range(index,-1,-1):
            if (DGHs[field][1][j] == currentLocation-1) and (flag!=1):
                person[field] = DGHs[field][0][j]
                flag = 1

def generalize(DGHs, field, fields, current_list):
    equal_list = getEquals(current_list, fields)
    while(not all(equal_list)):
        for person1 in range(len(current_list)):
            for person2 in range(person1+1, len(current_list)):
                new_p1 = {m: current_list[person1][m] for m in fields}
                new_p2 = {m: current_list[person2][m] for m in fields}
                for field in fields:
                    while (new_p1[field] != new_p2[field]):
                        p1_loc = DGHs[field][1][DGHs[field][0].index(new_p1[field])]
                        p2_loc = DGHs[field][1][DGHs[field][0].index(new_p2[field])]
                        if (p1_loc < p2_loc):
                            moveUp(DGHs, field, current_list[person2], fields)
                            moveUp(DGHs, field, new_p2, fields)
                        if (p2_loc < p1_loc):
                            moveUp(DGHs, field, current_list[person1], fields)
                            moveUp(DGHs, field, new_p1, fields)
                        if (p1_loc == p2_loc):
                            moveUp(DGHs, field, current_list[person2], fields)
                            moveUp(DGHs, field, current_list[person1], fields)
                            moveUp(DGHs, field, new_p2, fields)
                            moveUp(DGHs, field, new_p1, fields)
        equal_list = getEquals(current_list, fields)
    return current_list

def calculate_LM_cost(DGHs, field, person, fields):
    weight = 1/len(fields)
    nominator = 0
    denominator = 0
    if field in fields:
        values = DGHs[field][1]
        index = DGHs[field][0].index(person[field])
        currentLocation = DGHs[field][1][index]
        if (index == len(DGHs[field][1])-1 or DGHs[field][1][index+1]==currentLocation):
            nominator = 1
        else:
            for j in range(index+1,len(DGHs[field][1])-1,1):
                if (currentLocation ==  DGHs[field][1][j]):
                    nominator += 1
                    break
                if(values[j] == values[j+1] or values[j] > values[j+1]):
                    nominator += 1
                if(j+1 == len(values)-1):
                    if (values[j+1]<=values[j]):
                        nominator += 1


    if field in fields:
        names = DGHs[field][0]
        values = DGHs[field][1]
        for i in range(len(names)-1):
            if (values[i] == values[i+1] or values[i] > values[i+1]):
                denominator += 1
            if (i+1 == len(names)-1):
                if (values[i+1]<=values[i]):
                    denominator += 1
    print("nominator: ",nominator, denominator, field, person[field], 
        DGHs[field][1])

    return weight * ((nominator-1)/(denominator-1))


def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    #read dgh
    with open(DGH_file) as f:
        lines = f.readlines()
    #create a dictionary
    dgh = []
    for line in lines: 
        if line:
            numTabs = line.count("\t")
            dgh.append((line.strip(), numTabs))
    names = [item[0] for item in dgh]
    values = [item[1] for item in dgh]
    return [names, values]
    


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    fields = list(DGHs.keys())
    total_cost = 0

    for i in range(len(anonymized_dataset)):
        cost_for_record = 0
        for field in fields:
            cost_of_field = 0
            while(raw_dataset[i][field]!=anonymized_dataset[i][field]):
                moveUp(DGHs, field, raw_dataset[i], fields)
                cost_of_field += 1
            cost_for_record += cost_of_field
        total_cost += cost_for_record

    
    return total_cost



def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    fields = list(DGHs.keys())
    total_cost = 0
    for i in range(len(anonymized_dataset)):
        cost_of_record = 0
        for field in fields:
            cost_of_record += calculate_LM_cost(DGHs, field, anonymized_dataset[i], fields)
        total_cost += cost_of_record
    return total_cost
            

def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)
    
    #TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    fields = list(DGHs.keys())

    current_list = []


    
    count = 0
    for person in raw_dataset:
        count += 1
        current_list.append(person)
        if(count==k):
            break
    
    mod = 0
    length = 0
    if(D%k==0):
        length = D
    else:
        mod = D%k
        length = D-mod


    equal_list = getEquals(current_list, fields)
    while (count <= length):
        while(not all(equal_list)):
            for person1 in range(len(current_list)):
                for person2 in range(person1+1, len(current_list)):
                    #print(person1, person2)
                    new_p1 = {m: current_list[person1][m] for m in fields}
                    new_p2 = {m: current_list[person2][m] for m in fields}
                    for field in fields:
                        while (new_p1[field] != new_p2[field]):
                            p1_loc = DGHs[field][1][DGHs[field][0].index(new_p1[field])]
                            p2_loc = DGHs[field][1][DGHs[field][0].index(new_p2[field])]
                            if (p1_loc < p2_loc):
                                moveUp(DGHs, field, current_list[person2], fields)
                                moveUp(DGHs, field, new_p2, fields)
                            if (p2_loc < p1_loc):
                                moveUp(DGHs, field, current_list[person1], fields)
                                moveUp(DGHs, field, new_p1, fields)
                            if (p1_loc == p2_loc):
                                moveUp(DGHs, field, current_list[person2], fields)
                                moveUp(DGHs, field, current_list[person1], fields)
                                moveUp(DGHs, field, new_p2, fields)
                                moveUp(DGHs, field, new_p1, fields)
                            #print(new_p1[field], new_p2[field])
            equal_list = getEquals(current_list, fields)
        clusters.append(current_list)
        current_list = raw_dataset[count:count+k]
        equal_list = getEquals(current_list, fields)
        if(count+2*k>length):
            current_list = raw_dataset[count:length+mod]
            equal_list = getEquals(current_list, fields)
            break
        count += k
        
    
    while(not all(equal_list)):
            for person1 in range(len(current_list)):
                for person2 in range(person1+1, len(current_list)):
                    #print(person1, person2)
                    new_p1 = {m: current_list[person1][m] for m in fields}
                    new_p2 = {m: current_list[person2][m] for m in fields}
                    for field in fields:
                        while (new_p1[field] != new_p2[field]):
                            p1_loc = DGHs[field][1][DGHs[field][0].index(new_p1[field])]
                            p2_loc = DGHs[field][1][DGHs[field][0].index(new_p2[field])]
                            if (p1_loc < p2_loc):
                                moveUp(DGHs, field, current_list[person2], fields)
                                moveUp(DGHs, field, new_p2, fields)
                            if (p2_loc < p1_loc):
                                moveUp(DGHs, field, current_list[person1], fields)
                                moveUp(DGHs, field, new_p1, fields)
                            if (p1_loc == p2_loc):
                                moveUp(DGHs, field, current_list[person2], fields)
                                moveUp(DGHs, field, current_list[person1], fields)
                                moveUp(DGHs, field, new_p2, fields)
                                moveUp(DGHs, field, new_p1, fields)
                            #print(new_p1[field], new_p2[field])
            equal_list = getEquals(current_list, fields)
    clusters.append(current_list)
    
    #print(len(clusters[-1]))


    '''while(not all(equal_list)):
        for person1 in range(len(current_list)):
            for person2 in range(person1+1, len(current_list)):
                print(person1, person2)
                new_p1 = {m: current_list[person1][m] for m in fields}
                new_p2 = {m: current_list[person2][m] for m in fields}
                for field in fields:
                    while (new_p1[field] != new_p2[field]):
                        p1_loc = DGHs[field][1][DGHs[field][0].index(new_p1[field])]
                        p2_loc = DGHs[field][1][DGHs[field][0].index(new_p2[field])]
                        if (p1_loc < p2_loc):
                            moveUp(DGHs, field, current_list[person2], fields)
                            moveUp(DGHs, field, new_p2, fields)
                        if (p2_loc < p1_loc):
                            moveUp(DGHs, field, current_list[person1], fields)
                            moveUp(DGHs, field, new_p1, fields)
                        if (p1_loc == p2_loc):
                            moveUp(DGHs, field, current_list[person2], fields)
                            moveUp(DGHs, field, current_list[person1], fields)
                            moveUp(DGHs, field, new_p2, fields)
                            moveUp(DGHs, field, new_p1, fields)
                        print(new_p1[field], new_p2[field])
        equal_list = getEquals(current_list, fields)

        print(current_list)
    print(equal_list)
    '''
            
            
   
                   




    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)



def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i+1

    #TODO: complete this function.
    fields = list(DGHs.keys())
    flag_records = [(i+1) for i in range(len(raw_dataset))]
    clusters = []
    
    while(len(flag_records)>=2*k):
        current_list = []
        dists = {}
        topmost_index = min(flag_records)
        topmost_record = raw_dataset[topmost_index-1].copy()
        topmost_copy = raw_dataset[topmost_index-1].copy()
        copyset = deepcopy(raw_dataset)
        flag_records.remove(topmost_index)
        for i in flag_records:
            topmost_record = raw_dataset[topmost_index-1].copy()
            for field in fields:
                current_list = [topmost_record, copyset[i-1]]
                generalized_list = generalize(DGHs, field, fields, current_list)
                dists[i] = (calculate_LM_cost(DGHs, field, generalized_list[0], fields) +
                calculate_LM_cost(DGHs, field, generalized_list[1], fields))
        current_list = [topmost_copy]
        
        
        for j in range(k-1):
            min_index = min(dists, key=dists.get)
            current_list.append(raw_dataset[min_index-1].copy())
            flag_records.remove(min_index)
            dists.pop(min_index)
        
        #print(current_list)
        for field in fields:
            generalize(DGHs, field, fields, current_list)
        #print(current_list[0])
        
        #print(len(flag_records))
        clusters.append(current_list)
    

    dists = {}
    topmost_index = min(flag_records)
    topmost_record = raw_dataset[topmost_index-1].copy()
    topmost_copy = raw_dataset[topmost_index-1].copy()
    copyset = deepcopy(raw_dataset)
    flag_records.remove(topmost_index)
    for i in flag_records:
        topmost_record = raw_dataset[topmost_index-1].copy()
        for field in fields:
            current_list = [topmost_record, copyset[i-1]]
            generalized_list = generalize(DGHs, field, fields, current_list)
            dists[i] = (calculate_LM_cost(DGHs, field, generalized_list[0], fields) +
            calculate_LM_cost(DGHs, field, generalized_list[1], fields))
    current_list = [topmost_copy]
    for j in range(len(flag_records)):
        min_index = min(dists, key=dists.get)
        current_list.append(raw_dataset[min_index-1])
        flag_records.remove(min_index)
        dists.pop(min_index)
    
    for field in fields:
        generalize(DGHs, field, fields, current_list)


    #print(len(flag_records))
    clusters.append(current_list)



    
    # Finally, write dataset to a file
    anonymized_dataset = [None] * (len(raw_dataset)+1)

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']
        
    for i in range(1,len(anonymized_dataset)):
        anonymized_dataset[i-1] = anonymized_dataset[i]
    del anonymized_dataset[-1]

    write_dataset(anonymized_dataset, output_file)



def bottomup_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    '''
    fields = list(DGHs.keys())
    DGH_max_levels = []
    current_levels = [(k, 0) for k in fields]
    

    for i in DGHs:
        DGH_max_levels.append((i,max(DGHs[i][1])))

    current_list = []
    equal_list = [False]
    while(not all(equal_list)):
        for person in raw_dataset:
            trueNode = True
            for field in fields:
                index = DGHs[field][0].index(person[field])
                currentLocation = DGHs[field][1][index]
                if(currentLocation != current_levels[fields.index(field)][1]):
                    trueNode = False
            if(trueNode):
                current_list.append(person)
        equal_list = getEquals(current_list, fields)
    '''
                
        



    # Finally, write dataset to a file
    #write_dataset(anonymized_dataset, output_file)


#elements = read_DGH("/Users/burak/Downloads/HW1/DGHs/marital-status.txt")
#random_anonymizer('/Users/burak/Downloads/HW1/adult-hw1.csv','/Users/burak/Downloads/HW1/DGHs',5,'/Users/burak/Downloads/HW1/adult-hw1-anon.csv',1)
#clustering_anonymizer('/Users/burak/Downloads/HW1/adult-hw1.csv','/Users/burak/Downloads/HW1/DGHs',5,'/Users/burak/Downloads/HW1/adult-hw1-anon.csv')
#bottomup_anonymizer('/Users/burak/Downloads/HW1/adult-hw1.csv','/Users/burak/Downloads/HW1/DGHs',5,'/Users/burak/Downloads/HW1/adult-hw1-anon.csv')
#print(cost_LM('/Users/burak/Downloads/HW1/adult-hw1.csv','/Users/burak/Downloads/HW1/adult-hw1-anon.csv','/Users/burak/Downloads/HW1/DGHs'))
#print(elements)
# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now() ##
print(start_time) ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now() ##
print(end_time) ##
print(end_time - start_time)  ##

# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300 5
# skeleton.py random DGHs/ adult-hw1.csv adult-hw1-anon.csv 5 1

