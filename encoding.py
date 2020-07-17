import csv
import math
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

num_sc = 0
num_exp = 0

# 1. get the unique id list
print('#' * 15)
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
# print('Id List:')
# print(id_list)
print('The length of id list is {}'.format(len(id_list)))
print('The size of testing data is {}'.format(num_exp))
# print('The Min and Max of id list are {} and {}.'.format(min(id_list), max(id_list)))


# 2. Multi-Label Binarizer
# mlb = MultiLabelBinarizer()
# with open('data/LOG.csv', encoding='utf-8') as f:
#     f_csv = csv.reader(f)
#     mlb.fit_transform(f_csv)
num_id = 339
# Create the Multi-Label Matrix
print('Create Multi-Label Binarizer')
mlb = np.zeros((num_exp, num_id))

with open('data/LOG.csv', encoding='utf-8') as f:
    f_csv = csv.reader(f)
    for (id, row) in enumerate(f_csv):
        for element in row:
            if element in id_list or element == '\ufeff326':
                # idx =  id_list.index(element)
                if element == '\ufeff326':
                    element = '326'
                idx = int(element) - 1
                mlb[id, idx] = 1
                num_sc += 1
print('The # and percentage of activated scan chains are {:.2f} and {:.2f}%.'.format(num_sc / num_exp, \
        100. * num_sc / (num_exp * num_id)))


# 3. Merging test cubes
print('*' * 15)
print('Merging test cubes')
group_ctrl = 19
chain_ctrl = 18
print('Control bits settings:{} chain ctrl and {} group ctrl'.format(chain_ctrl, group_ctrl))
merge_upper = 0.2
print('The upper bound of activated scan chain percentage is {}'.format(merge_upper))


def conflict_check(row1, row2):
    return (row1 * row2).sum() == 0

def group_bit_determine(row):
    # Determine the group control bits for a specific row
    encoded_group_ctrl = np.zeros(group_ctrl)
    for (id, ele) in enumerate(row):
        if ele == 1:
            encoded_group_ctrl[id % group_ctrl] = 1
    return encoded_group_ctrl

def calculate_group_overlap(row1, row2):
    # Cacalate the overlap percentage of grouping control bits
    ctrl1 = group_bit_determine(row1)
    ctrl2 = group_bit_determine(row2)
    return (ctrl1 * ctrl2).sum() / (ctrl1 + ctrl2 - ctrl1 * ctrl2).sum() 
 
def merge_two_cube(row1, row2):
    # Merge two testing cube
    for i in range(row1.shape[0]):
        if row2[i] == 1:
            row1[i] = 1
    return row1

def calculate_activated_percentage(row):
    return row.sum()/row.shape[0]
    
# Start merging the test cube
print('Create merged array')
merged_array = []

print('Start merging')
# Simplify the dataset
mlb = mlb[:1000]
merged_cube = mlb[0]
mlb = np.delete(mlb, 0, 0)
while mlb.shape[0] >= 1:
    overlap = np.zeros(mlb.shape[0])
    for (id, row) in enumerate(mlb):
        if conflict_check(merged_cube, row):
            overlap[id] = calculate_group_overlap(row, merged_cube)
    if overlap.sum() == 0:
        # all conflicted
        merged_array.append(merged_cube)
        merged_cube = mlb[0]
        mlb = np.delete(mlb, 0, 0)
        continue
    id_max = np.argmax(overlap)
    merged_cube = merge_two_cube(merged_cube, mlb[id_max])
    mlb = np.delete(mlb, id_max, 0)
    if calculate_activated_percentage(merged_cube) > merge_upper:
        merged_array.append(merged_cube)
        merged_cube = mlb[0]
        mlb = np.delete(mlb, 0, 0)

merged_array = np.array(merged_array)
print(merged_array.shape)

# 4. Encoding        
# Specify the control bits of each test cube
num_activated = 0
encoded_efficiency = 0
encoded_success = 0
num_exp = merged_array.shape[0]
print(num_exp)
constraint = 0.5
encoded_samples_group = np.zeros((num_exp, group_ctrl))
encoded_samples_chain = np.zeros((num_exp, chain_ctrl))
for (id, sample) in enumerate(merged_array):
    for (ele_id, ele) in enumerate(sample):
        if ele == 1.0:
            num_activated += 1
            encoded_samples_group[id, ele_id % group_ctrl] = 1
            encoded_samples_chain[id, ele_id // group_ctrl] = 1

# Caculate the encoding efficiency
for id in range(num_exp): 
    activated = encoded_samples_group[id, :].sum() \
               * encoded_samples_chain[id, :].sum()
    if (activated / num_id) <= constraint:
        encoded_success += 1
    encoded_efficiency += activated

activated = num_activated / (num_exp * num_id)
efficiency = encoded_efficiency / (num_exp * num_id)
success = encoded_success / num_exp 
print('Activated percentage after merging is {:.2f}%.'.format(100.*activated))
print('Encoding Efficiency is {:.2f}%.'.format(100.*efficiency))
print('Encoding success rate is {:.2f}%.'.format(100.*success))

   



# 4. Encoding
# Toy Examples
# num_training = 10000
# len_sc = 100
# group_ctrl = 10
# chain_ctrl = 10
# state_sc = [0., 1.]
# prob_sc = [0.8, 0.2]
# constraint = 0.5
# encoded_efficiency = 0
# encoded_success = 0 
# 
# samples = np.random.choice(state_sc, size=(num_training, len_sc), p=prob_sc)
# encoded_samples_group = np.zeros((num_training, group_ctrl))
# encoded_samples_chain = np.zeros((num_training, chain_ctrl))
# 
# # Specify the control bits of each test cube
# for (id, sample) in enumerate(samples):
#     for (ele_id, ele) in enumerate(sample):
#         if ele == 1.0:
#             encoded_samples_group[id, ele_id // group_ctrl] = 1
#             encoded_samples_chain[id, ele_id % group_ctrl] = 1
# 
# # Caculate the encoding efficiency
# for id in range(num_training): 
#     activated = encoded_samples_group[id, :].sum() \
#                * encoded_samples_chain[id, :].sum()
#     if (activated / len_sc) <= constraint:
#         encoded_success += 1
#     encoded_efficiency += activated
# 
# efficiency = encoded_efficiency / (num_training * len_sc)
# success = encoded_success / num_training 
# print('Encoding Efficiency is {:.2f}%.'.format(100.*efficiency))
# print('Encoding success rate is {:.2f}%.'.format(100.*success))




