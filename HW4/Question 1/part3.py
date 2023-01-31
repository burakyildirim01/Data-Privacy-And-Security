import pandas as pd
import numpy as np
import hashlib
from copy import deepcopy

digitalcorp_df = pd.read_csv('keystreching-digitalcorp.txt', sep=",")
rockyou_df = np.genfromtxt('rockyou.txt', dtype='str')

hashed_passwords = {}
max_iterations = 2000

for i in digitalcorp_df['salt']:
    for j in rockyou_df:
        hashed_passwords[hashlib.sha512((j+i).encode('utf-8')).hexdigest()] = (j,i) #hash = h(password + salt)
        # other combinations are tried as well by changing the order of the concatenation

        
for m in digitalcorp_df['hash_outcome']:
    if m in hashed_passwords.keys():
        index = digitalcorp_df[digitalcorp_df['hash_outcome'] == m].index[0]
        print("User: "+ digitalcorp_df['username'].values[index] + "\t" + "Password: " + hashed_passwords[m][0])

first_passwords = deepcopy(hashed_passwords)
second_passwords = {}

for i in range(1,max_iterations):
    

    found = False

    key_list = list(first_passwords.keys())
    for k in key_list:
        hash = hashlib.sha512((first_passwords[k][0]+k+first_passwords[k][1]).encode('utf-8')).hexdigest() #new hash = h(password + old hash + salt)
        # other combinations are tried as well by changing the order of the concatenation
        second_passwords[hash] = first_passwords[k]

    for m in digitalcorp_df['hash_outcome']:
        if m in second_passwords.keys():
            index = digitalcorp_df[digitalcorp_df['hash_outcome'] == m].index[0]
            print("User: "+ digitalcorp_df['username'].values[index] + "\t" + "Password: " + second_passwords[m][0])
            found = True
    
    if found:
        print("Found in iteration: " + str(i))
        passwords = [x[0] for x in list(second_passwords.values())]
        data = {'password':passwords, 'hash_of_password':list(second_passwords.keys())}
        hashed_passwords_df = pd.DataFrame.from_dict(data)
        hashed_passwords_df.to_csv('part3-attack-table-keystreching.csv', sep=',', index=False)
        break
    
    

    first_passwords = deepcopy(second_passwords)
    second_passwords = {}

    if i % 100 == 0:
        print(i)
