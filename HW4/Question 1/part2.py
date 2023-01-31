import pandas as pd
import numpy as np
import hashlib

digitalcorp_df = pd.read_csv('salty-digitalcorp.txt', sep=",")
rockyou_df = np.genfromtxt('rockyou.txt', dtype='str')

hashed_passwords = {}

for i in digitalcorp_df['salt']:
    for j in rockyou_df:
        hashed_passwords[hashlib.sha512((j+i).encode('utf-8')).hexdigest()] = j #hash = h(password + salt)
        hashed_passwords[hashlib.sha512((i+j).encode('utf-8')).hexdigest()] = j #hash = h(salt + password)

print("INFERRED PASSWORDS")
print("******************")

for i in digitalcorp_df['hash_outcome']:
    if i in hashed_passwords.keys():
        index = digitalcorp_df[digitalcorp_df['hash_outcome'] == i].index[0]
        print("User: "+ digitalcorp_df['username'].values[index] + "\t" + "Password: " + hashed_passwords[i])

data = {'password':list(hashed_passwords.values()), 'hash_of_password':list(hashed_passwords.keys())}
hashed_passwords_df = pd.DataFrame.from_dict(data)
hashed_passwords_df.to_csv('part2-attack-table-salty.csv', sep=',', index=False)