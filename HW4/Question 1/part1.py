import pandas as pd
import numpy as np
import hashlib

digitalcorp_df = pd.read_csv('digitalcorp.txt', sep=",")
rockyou_df = np.genfromtxt('rockyou.txt', dtype='str')

hashed_passwords = {}

for i in rockyou_df:
    hashed_passwords[hashlib.sha512(i.encode('utf-8')).hexdigest()] = i


print("INFERRED PASSWORDS")
print("******************")

for i in digitalcorp_df['hash_of_password']:
    if i in hashed_passwords.keys():
        index = digitalcorp_df[digitalcorp_df['hash_of_password'] == i].index[0]
        print("User: "+ digitalcorp_df['username'].values[index] + "\t" + "Password: " + hashed_passwords[i])


data = {'password':list(hashed_passwords.values()), 'hash_of_password':list(hashed_passwords.keys())}

hashed_passwords_df = pd.DataFrame.from_dict(data)
hashed_passwords_df.to_csv('part1a-attack-table-pure.csv', sep=',', index=False)