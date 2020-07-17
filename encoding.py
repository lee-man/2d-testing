import csv
import random
import math
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import copy

from sklearn import metrics
import matplotlib.pyplot as plt
plt.switch_backend('agg')


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



class TwoDimEncoding(object):
    '''
    The class for Two-Dimention Low-Power Encoding.

    '''
    def __init__(self, mlb, group_ctrl=19, chain_ctrl=18, mux_ctrl=3, upper_bound=0.5):
        self.mlb = mlb
        self.num_cube = mlb.shape[0]
        self.num_id = mlb.shape[1]
        self.group_ctrl = group_ctrl
        self.chain_ctrl = chain_ctrl
        self.mux_ctrl = int(math.pow(2, mux_ctrl))
        self.upper_bound = upper_bound
        self.group_mapping = []
        self.merged_array = None
        self.num_merged_cube = None
        self.encoded_group = None
        self.encoded_chain = None
        self.encoded_mux = None
        self._print_info()
        
    
    def _print_info(self):
        print('*' * 5, 'Statistic', '*' * 5)
        print('The size of testing dataset is {}'.format(self.num_cube))
        print('The size of each test cube is {}'.format(self.num_id))
        print('Control bits settings:{} chain ctrl, {} group ctrl and mux crtl'.format(self.chain_ctrl, self.group_ctrl, self.mux_ctrl))
        print('The upper bound of activated scan chian for low power encoding is {}.'.format(self.upper_bound))

    def generate_group_mapping(self):
        '''
        Group Mapping: map the scan chain id to the underlying line.
        '''
        id_list = np.arange(self.num_id)
        self.group_mapping[0] = {str(id): id for id in range(self.num_id)}
        for j in range(1, self.mux_ctrl):
            np.random.shuffle(id_list)
            self.group_mapping[j] = {str(id): i for i, id in enumerate(id_list)}
            
    def check_conflict(self, cube1, cube2):
        '''
        Check whether two cubes have a confliction
        '''
        return (cube1 * cube2).sum() == 0
    
    def determin_group_bit(self, cube, mux_id):
        '''
        Determine the group control bits for a specific row
        '''
        if not mux_id < self.mux_ctrl:
            raise VauleError('The MUX id is beyond the range')
        for (id, ele) in enumerate(row):
            if ele == 1:
                encoded_group_ctrl[group_mapping[mux_id][str(id)] % group_ctrl] = 1
        return encoded_group_ctrl

    def calculate_group_overlap(self, cube1, cube2, mux_id):
        '''
        Cacalate the overlap percentage of grouping control bits
        '''
        ctrl1 = self.determin_group_bit(cube1, mux_id)
        ctrl2 = self.determin_group_bit(cube1, mux_id)
        return (ctrl1 * ctrl2).sum() / (ctrl1 + ctrl2 - ctrl1 * ctrl2).sum() 

    def merge_two_cube(self, cube1, cube2):
        '''
        Merge two testing cube.
        '''
        for i in range(row1.shape[0]):
            if row2[i] == 1:
                row1[i] = 1
        return row1
    
    def calculate_activated_percentage(self, cube):
        return cube.sum()/row.shape[0]

    def merging(self):
        print('*' * 15)
        print('Start Merging.')
        mlb = copy.deepcopy(self.mlb)
        merged_array = []
        # picked_cube = []
        merged_cube = mlb[0]
        
        mlb = np.delete(mlb, 0, 0)
        while mlb.shape[0] >= 1:
            for (id, row) in enumerate(mlb):
                if self.check_conflict(merged_cube, row):
                    merged_cube = self.merge_two_cube(merged_cube, mlb[id])
                    # picked_cube.append(mlb[id])
                    mlb = np.delete(mlb, id, 0)
                    if self.calculate_activated_percentage(merged_cube) > self.upper_bound:
                        merged_array.append(merged_cube)
                        merged_cube = mlb[0]
                        # picked_cube = []
                        mlb = np.delete(mlb, 0, 0)
                        continue
            merged_array.append(merged_cube)
            merged_cube = mlb[0]
            # picked_cube = []
            mlb = np.delete(mlb, 0, 0)
        
        self.merged_array = np.array(merged_array)

    def encoding(self):
        print('*' * 15)
        print('Start Encoding.')
        self.num_merged_cube = self.merged_array.shape[0]
        self.encoded_group = np.zeros((self.num_merged_cube, self.group_ctrl))
        self.encoded_chain = np.zeros((self.num_merged_cube, self.chain_ctrl))
        self.encoded_mux = np.zeros(self.num_merged_cube)
        for (id, sample) in enumerate(merged_array):
            group_bit = np.zeros((self.mux_ctrl, self.group_ctrl))
            chain_bit = np.zeros((self.mux_ctrl, self.chain_ctrl))
            for mux_bit in range(self.mux_ctrl):
                for (ele_id, ele) in enumerate(sample):
                    if ele == 1.0:
                        group_bit[mux_bit, self.group_mapping[str(ele_id)] % group_ctrl] = 1
                        chain_bit[mux_bit, self.group_mapping[str(ele_id)] // group_ctrl] = 1
            self.encoded_mux = np.argmin(group_bit.sum(axis=1) * chain_bit(axis=1))

    def eval(self):
        print('*' * 15)
        print('Evalutation.')
        specified_num = []
        activated_num = []
        encoded_efficiency = 0
        encoded_success = 0
        constrain = 0.5
        self.num_merged_cube = self.merged_array.shape[0]
        print('Total number of merged test cube is {}'.format(num_merged_cube))
        for id in range(num_merged_cube):
            specified_num[id] = self.merged_array[id].sum()
            activated_num[id] = self.encoded_group[id].sum() \
                            * self.encoded_chain[id].sum()
            if (activated / self.num_id) <= constraint:
                encoded_success += 1
        specified_num = np.array(specified_num)
        activated_num = np.array(activated_num)
        
        plt.figure()
        sns.distplot(specified_num, hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Specified')
        sns.distplot(activated_nums, hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Activated')
        plt.xlabel(args.process + ' # Scan Chain')
        plt.ylabel('Density')
        plt.legend()

        if not os.path.isdir('figs/'):
                os.makedirs(os.path.dirname('figs'))
        plt.savefig('figs/hist.png')

        specified_percentage = specified_num.sum() / (self.num_merged_cube * self.num_id)
        activated_percentage = activated_num.sum() / (self.num_merged_cube * self.num_id)
        succeeded =  encoded_success / self.num_merged_cube
        print('Specified scan chain percentage after merging is {:.2f}%.'.format(100.*specified_percentage))
        print('Activaed scan chain percentages is {:.2f}%.'.format(100.*activated_percentage))
        print('Encoding success rate is {:.2f}%.'.format(100.*succeeded))


        
encoder = TwoDimEncoding(mlb)
encoder.merging()
encoder.encoding()
encoder.eval()
   



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




