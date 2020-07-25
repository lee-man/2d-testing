import csv
import os
import random
import math
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import copy

from sklearn import metrics
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import seaborn as sns



num_sc = 0
num_exp = 0

###########################
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



###########################
# 2. Multi-Label Binarizer
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




##########################
# 3. Merging and Encoding using 2D structure
class TwoDimEncoding(object):
    '''
    The class for Two-Dimention Low-Power Encoding.

    '''
    def __init__(self, mlb, group_ctrl=19, chain_ctrl=18, mux_ctrl=3, upper_bound=0.5, map_mode='Stochastic'):
        self.mlb = mlb
        self.num_cube = mlb.shape[0]
        self.num_id = mlb.shape[1]
        self.group_ctrl = group_ctrl
        self.chain_ctrl = chain_ctrl
        self.mux_ctrl = int(math.pow(2, mux_ctrl))
        self.upper_bound = upper_bound
        self.sc_counts = None
        self.group_mapping = {}
        self.merged_array = None
        self.num_merged_cube = None
        self.encoded_group = None
        self.encoded_chain = None
        self.encoded_mux = None
        self._print_info()
        self.scan_chain_hist()
        self.generate_group_mapping()
        
    
    def _print_info(self):
        print('*' * 5, 'Statistic', '*' * 5)
        print('The size of testing dataset is {}'.format(self.num_cube))
        print('The size of each test cube is {}'.format(self.num_id))
        print('Control bits settings:{} chain ctrl, {} group ctrl and {} mux crtl'.format(self.chain_ctrl, self.group_ctrl, self.mux_ctrl))
        print('The upper bound of activated scan chian for low power encoding is {}.'.format(self.upper_bound))

    def scan_chain_hist(self, draw=False):

        # Draw the histogram of dense of each scan chain
        self.sc_counts = np.zeros(num_id)
        for row in self.mlb:
            for (eid, element) in enumerate(row):
                if element == 1:
                    self.sc_counts[eid] += 1

        if draw:
            plt.figure()
            x = [i for i in range(self.num_id)]
            # plt.plot(sc_counts)
            plt.scatter(x, self.sc_counts)
            plt.xlabel('Scan Chain ID')
            plt.ylabel('Density')
            if not os.path.isdir('figs/'):
                os.makedirs(os.path.dirname('figs/'))
            plt.savefig('figs/sc_counts.png')

        self.sc_counts /= self.sc_counts.max()
        ind = np.argsort(self.sc_counts)
        print (ind)
        exit()

    def generate_group_mapping(self, mode='random'):
        '''
        Group Mapping: map the scan chain id to the underlying line.
        '''
        
        id_list = np.arange(self.num_id)
        self.group_mapping[0] = {str(id): id for id in range(self.num_id)}
        # Random
        if mode == 'random':
            # id_list = np.arange(self.num_id)
            # self.group_mapping[0] = {str(id): id for id in range(self.num_id)}
            for j in range(1, self.mux_ctrl):
                np.random.shuffle(id_list)
                self.group_mapping[j] = {str(id): i for i, id in enumerate(id_list)}

        # # Build the weight matrix
        # weight_mat = np.zeros((self.num_id, self.num_id))
        # # TO DO: change to matric manipulation
        # for i in range(self.num_id):
        #     weight_mat[i] = self.sc_counts[i] + self.sc_counts

        # Stochastic
        elif mode == 'stochastic':
            # Did not consider the disentangle yet
            sc_conf = np.zeros((self.mux_ctrl-1, self.num_id, self.group_ctrl))
            constrain = 0.5
            satisfied = False

            for j in range(1, self.mux_ctrl):
                id_list = np.arange(self.num_id)
                id_list = np.random.choice(id_list, size=num_id, replace=False, p=self.sc_counts)
                self.group_mapping[j] = {str(id): i for i, id in enumerate(id_list)}
                for (key, value) in self.group_mapping[j]:
                    group_id = value // self.chain_ctrl
                    sc_conf[j][int(key)][group_id] = 1

        
        else:
            raise NotImplementedError('The mode should be either random, stochastic or deterministic')
        

        # Deterministic

            
    def check_conflict(self, cube1, cube2):
        '''
        Check whether two cubes have a confliction
        '''
        return (cube1 * cube2).sum() == 0
    
    # def determin_group_bit(self, cube, mux_id):
    #     '''
    #     Determine the group control bits for a specific row
    #     '''
    #     if not mux_id < self.mux_ctrl:
    #         raise VauleError('The MUX id is beyond the range')
    #     for (id, ele) in enumerate(row):
    #         if ele == 1:
    #             encoded_group_ctrl[group_mapping[mux_id][str(id)] % group_ctrl] = 1
    #     return encoded_group_ctrl

    # def calculate_group_overlap(self, cube1, cube2, mux_id):
    #     '''
    #     Cacalate the overlap percentage of grouping control bits
    #     '''
    #     ctrl1 = self.determin_group_bit(cube1, mux_id)
    #     ctrl2 = self.determin_group_bit(cube1, mux_id)
    #     return (ctrl1 * ctrl2).sum() / (ctrl1 + ctrl2 - ctrl1 * ctrl2).sum() 

    def merge_two_cube(self, cube1, cube2):
        '''
        Merge two testing cube.
        '''
        cube = np.zeros(cube1.shape[0])
        for i in range(cube1.shape[0]):
            if cube1[i] == 1 or cube2[i] == 1:
                cube[i] = 1
        return cube
    
    def calculate_specified_percentage(self, cube):
        return cube.sum()/cube.shape[0]

    def calculate_activated_percentage(self, merged_cube, to_merged_cube):
        cube = self.merge_two_cube(merged_cube, to_merged_cube)
        group_bit = np.zeros((self.mux_ctrl, self.group_ctrl))
        chain_bit = np.zeros((self.mux_ctrl, self.chain_ctrl))
        for mux_bit in range(self.mux_ctrl):
                for (ele_id, ele) in enumerate(cube):
                    if ele == 1.0:
                        group_bit[mux_bit, self.group_mapping[mux_bit][str(ele_id)] // self.chain_ctrl] = 1
                        chain_bit[mux_bit, self.group_mapping[mux_bit][str(ele_id)] % self.chain_ctrl] = 1
        encoded_mux = np.argmin(group_bit.sum(axis=1) * chain_bit.sum(axis=1))
        activated_num = group_bit[encoded_mux].sum() \
                            * chain_bit[encoded_mux].sum()
        return activated_num / cube.shape[0], cube


    def merging(self):
        print('*' * 15)
        print('Start Merging.')
        mlb = copy.deepcopy(self.mlb)
        mask = np.zeros(mlb.shape[0])
        idx_now = 0
        merged_array = []
        # picked_cube = []
        merged_cube = copy.deepcopy(mlb[idx_now])
        while idx_now < (mlb.shape[0] - 1):
            for (id, row) in enumerate(mlb):
                if id == (mlb.shape[0] - 1):
                    merged_array.append(merged_cube)
                    while mask[idx_now] == 1 and idx_now < (mlb.shape[0] - 1):
                        idx_now += 1
                    mask[idx_now] = 1
                    merged_cube = copy.deepcopy(mlb[idx_now])
                if mask[id] == 1:
                    continue
                if self.check_conflict(merged_cube, row):
                    activated_percentage, merged_cube_candidate = self.calculate_activated_percentage(merged_cube, mlb[id])
                    if activated_percentage <= self.upper_bound:
                        merged_cube = merged_cube_candidate
                        mask[id] = 1
                    else:
                        merged_array.append(merged_cube)
                        while mask[idx_now] == 1 and idx_now < (mlb.shape[0] - 1):
                            idx_now += 1
                        mask[idx_now] = 1
                        merged_cube = copy.deepcopy(mlb[idx_now])
                        break
                    
        self.merged_array = np.array(merged_array)

    def encoding(self):
        print('*' * 15)
        print('Start Encoding.')
        self.num_merged_cube = self.merged_array.shape[0]
        self.encoded_group = np.zeros((self.num_merged_cube, self.group_ctrl))
        self.encoded_chain = np.zeros((self.num_merged_cube, self.chain_ctrl))
        self.encoded_mux = np.zeros(self.num_merged_cube)
        for (id, sample) in enumerate(self.merged_array):
            group_bit = np.zeros((self.mux_ctrl, self.group_ctrl))
            chain_bit = np.zeros((self.mux_ctrl, self.chain_ctrl))
            for mux_bit in range(self.mux_ctrl):
                for (ele_id, ele) in enumerate(sample):
                    if ele == 1.0:
                        group_bit[mux_bit, self.group_mapping[mux_bit][str(ele_id)] // self.chain_ctrl] = 1
                        chain_bit[mux_bit, self.group_mapping[mux_bit][str(ele_id)] % self.chain_ctrl] = 1
            self.encoded_mux[id] = np.argmin(group_bit.sum(axis=1) * chain_bit.sum(axis=1))
            self.encoded_group[id] = group_bit[int(self.encoded_mux[id])]
            self.encoded_chain[id] = chain_bit[int(self.encoded_mux[id])]

    def eval(self):
        print('*' * 15)
        print('Evalutation.')
        specified_num = np.zeros(self.num_merged_cube)
        activated_num = np.zeros(self.num_merged_cube)
        # encoded_success = 0
        # constraint = 0.5
        self.num_merged_cube = self.merged_array.shape[0]
        print('Total number of merged test cube is {}'.format(self.num_merged_cube))
        for id in range(self.num_merged_cube):
            specified_num[id] = self.merged_array[id].sum()
            activated_num[id] = self.encoded_group[id].sum() \
                            * self.encoded_chain[id].sum()
            # if (activated_num[id] / self.num_id) <= constraint:
            #     encoded_success += 1
        ranges = (np.min(specified_num), np.max(activated_num))
        
        plt.figure()
        sns.distplot(specified_num, hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Specified')
        sns.distplot(activated_num, hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Activated')
        plt.xlabel(' # Scan Chain')
        plt.ylabel('Density')
        plt.legend()

        if not os.path.isdir('figs/'):
                os.makedirs(os.path.dirname('figs/'))
        plt.savefig('figs/hist.png')

        specified_percentage = specified_num.sum() / (self.num_merged_cube * self.num_id)
        activated_percentage = activated_num.sum() / (self.num_merged_cube * self.num_id)
        # succeeded =  encoded_success / self.num_merged_cube
        print('Specified scan chain percentage after merging is {:.2f}%.'.format(100.*specified_percentage))
        print('Activaed scan chain percentages is {:.2f}%.'.format(100.*activated_percentage))
        # print('Encoding success   rate is {:.2f}%.'.format(100.*succeeded))




if __name__ == '__main__':
    # mlb = mlb[:10000] 
    encoder = TwoDimEncoding(mlb, map_mode='stochastic')
    encoder.merging()
    encoder.encoding()
    encoder.eval()

