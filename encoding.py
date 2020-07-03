import csv
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

total_num_sc = 0
max_id = 0
num_exp = 0

# 1. get the unique id list
print('Get the unique id list')
id_list = []
with open('data/LOG.csv', encoding='utf-8') as f:
    f_csv = csv.reader(f)
    
    for row in f_csv:
        num_exp += 1
        for element in row:
            if element not in id_list and element != '' and element != '\ufeff326':
                id_list.append(element)

id_list = sorted(id_list)
num_id = len(id_list)
print('The length of id list is {}'.format(len(id_list)))
print('The size of testing data is {}'.format(num_exp))

# print('The Min and Max of id list are {} and {}.'.format(min(id_list), max(id_list)))


# 2. Multi-Label Binarizer
# mlb = MultiLabelBinarizer()
# with open('data/LOG.csv', encoding='utf-8') as f:
#     f_csv = csv.reader(f)
#     mlb.fit_transform(f_csv)

# Create the Multi-Label Matrix
print('Create Multi-Label Binarizer')
# mlb = np.zeros((num_exp, num_id))
mlb = np.zeros((num_exp, 339))

with open('data/LOG.csv', encoding='utf-8') as f:
    f_csv = csv.reader(f)
    for (id, row) in enumerate(f_csv):
        for element in row:
            if element in id_list:
                # idx =  id_list.index(element)
                idx = int(element) - 1
                mlb[id, idx] = 1
                total_num_sc += 1
print('The # and percentage of activated scan chains are {:.2f} and {:.2f}%.'.format(total_num_sc / num_exp, \
        100. * total_num_sc / (num_exp * num_id)))



# Encoding
# Toy Examples
num_training = 10000
len_sc = 100
group_ctrl = 10
chain_ctrl = 10
state_sc = [0., 1.]
prob_sc = [0.8, 0.2]
constraint = 0.5
encoded_efficiency = 0
encoded_success = 0 

samples = np.random.choice(state_sc, size=(num_training, len_sc), p=prob_sc)
encoded_samples_group = np.zeros((num_training, group_ctrl))
encoded_samples_chain = np.zeros((num_training, chain_ctrl))

# Specify the control bits of each test cube
for (id, sample) in enumerate(samples):
    for (ele_id, ele) in enumerate(sample):
        if ele == 1.0:
            encoded_samples_group[id, ele_id % group_ctrl] = 1
            encoded_samples_chain[id, ele_id // chain_ctrl] = 1

# Caculate the encoding efficiency
for id in range(num_training): 
    activated = encoded_samples_group[id, :].sum() \
               * encoded_samples_chain[id, :].sum()
    if (activated / len_sc) <= constraint:
        encoded_success += 1
    encoded_efficiency += activated

efficiency = encoded_efficiency / (num_training * len_sc)
success = encoded_success / num_training 
print('Encoding Efficiency is {:.2f}%.'.format(100.*efficiency))
print('Encoding success rate is {:.2f}%.'.format(100.*success))




