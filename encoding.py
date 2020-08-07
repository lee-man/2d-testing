import csv
import os
import random
import math
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import copy
import argparse

from sklearn import metrics
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import seaborn as sns


def create_mlb():
    ###########################
    # 1. get the unique id list 
    num_sc = 0
    num_exp = 0
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
                if element == '\ufeff326' and ('326' not in id_list):
                    id_list.append('326')

    id_list = sorted(id_list)
    num_id = len(id_list)
    # print('Id List:')
    # print(id_list)
    print('The length of id list is {}'.format(num_id))
    print('The size of testing data is {}'.format(num_exp))

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

    np.save('data/mlb.npy', mlb)


##########################
# 3. Merging and Encoding using 2D structure
class TwoDimEncoding(object):
    '''
    The class for Two-Dimention Low-Power Encoding.

    '''
    def __init__(self, mlb_path, group_ctrl=19, chain_ctrl=18, mux_ctrl=3, upper_bound=0.5, sim_constraint=0.5, map_mode='stochastic', seed=0):
        self.mlb = np.load(mlb_path)
        self.num_cube = self.mlb.shape[0]
        self.num_id = self.mlb.shape[1]
        self.group_ctrl = group_ctrl
        self.chain_ctrl = chain_ctrl
        self.mux_ctrl = int(math.pow(2, mux_ctrl))
        self.upper_bound = upper_bound
        self.sim_constraint = sim_constraint
        self.mode = map_mode
        self.seed = seed
        self.sc_counts = None
        self.group_mapping = {}
        self.merged_array = None
        self.num_merged_cube = None
        self.encoded_group = None
        self.encoded_chain = None
        self.encoded_mux = None
        self._print_info()
        # self._set_seed()
        self.scan_chain_hist()
        self.generate_group_mapping()
        
    
    def _print_info(self):
        print('*' * 5, 'Statistic', '*' * 5)
        print('The size of testing dataset is {}'.format(self.num_cube))
        print('The size of each test cube is {}'.format(self.num_id))
        print('Control bits settings:{} chain ctrl, {} group ctrl and {} mux crtl'.format(self.chain_ctrl, self.group_ctrl, self.mux_ctrl))
        print('The upper bound of activated scan chian for low power encoding is {}.'.format(self.upper_bound))

    def _set_seed(self):
        # Set Seed
        np.random.seed(self.seed)
        random.seed(self.seed)

    def scan_chain_hist(self, draw=False):

        # Draw the histogram of dense of each scan chain
        self.sc_counts = np.zeros(self.num_id)
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

        if self.mode == 'stochastic':
            ind = np.argsort(self.sc_counts)
            ind = ind[::-1]
            # mutate the mlb according to the ranking
            self.mlb = self.mlb[:, ind]
            self.sc_counts = self.sc_counts[ind]
        # normalize
        self.sc_counts += 10 # add 10 to avoid zero case.
        self.sc_counts /= self.sc_counts.sum() 
        


    def generate_group_mapping(self):
        '''
        Group Mapping: map the scan chain id to the underlying line.
        '''
        
        id_list = np.arange(self.num_id)
        self.group_mapping[0] = {str(id): id for id in range(self.num_id)}
        # Random
        if self.mode == 'random':
            for j in range(1, self.mux_ctrl):
                np.random.shuffle(id_list)
                self.group_mapping[j] = {str(id): i for (i, id) in enumerate(id_list)}

        # Stochastic
        elif self.mode == 'stochastic':

            sc_conf = np.zeros((self.mux_ctrl-1, self.num_id))

            for j in range(1, self.mux_ctrl):
                satisfied = False
                while not satisfied:
                    id_list = np.arange(self.num_id)
                    id_list = np.random.choice(id_list, size=self.num_id, replace=False, p=self.sc_counts)
                    self.group_mapping[j] = {str(id): i for (i, id) in enumerate(id_list)}
                    for (key, value) in self.group_mapping[j].items():
                        group_id = value // self.chain_ctrl
                        sc_conf[j-1][int(key)] = group_id
                    for conf_ind in range(j-1):
                        similarity = (sc_conf[j-1] == sc_conf[conf_ind]).sum() / self.num_id
                        if similarity > self.sim_constraint:
                            sc_conf[j-1] = np.zeros(self.num_id)
                            satisfied = False
                            break
                    if sc_conf[j-1].sum() != 0:
                        satisfied = True
        elif self.mode == 'deterministic':
            raise NotImplementedError('The deterministic mode has not been done yet')
        
        else:
            raise NotImplementedError('The mode should be either random, stochastic or deterministic')
        

            
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
        merged_cube = copy.deepcopy(mlb[idx_now])
        while idx_now < (mlb.shape[0] - 1):
            for id in range(idx_now+1, mlb.shape(0))
                row = mlb[id]
                if id == (mlb.shape[0] - 1):
                    if mask[id] == 1:
                        merged_array.append(merged_cube)
                    else:
                        activated_percentage, merged_cube_candidate = self.calculate_activated_percentage(merged_cube, row)
                        if activated_percentage <= self.upper_bound:
                            merged_cube = merged_cube_candidate
                            mask[id] = 1
                        merged_array.append(merged_cube)
                    while mask[idx_now] == 1 and idx_now < (mlb.shape[0] - 1):
                        idx_now += 1
                    mask[idx_now] = 1
                    merged_cube = copy.deepcopy(mlb[idx_now])
                elif mask[id] == 1:
                    continue
                elif self.check_conflict(merged_cube, row):
                    activated_percentage, merged_cube_candidate = self.calculate_activated_percentage(merged_cube, row)
                    if activated_percentage <= self.upper_bound:
                        merged_cube = merged_cube_candidate
                        mask[id] = 1

                    
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
        self.num_merged_cube = self.merged_array.shape[0]
        print('Total number of merged test cube is {}'.format(self.num_merged_cube))
        for id in range(self.num_merged_cube):
            specified_num[id] = self.merged_array[id].sum()
            activated_num[id] = self.encoded_group[id].sum() \
                            * self.encoded_chain[id].sum()
          
        ranges = (np.min(specified_num), np.max(activated_num))
        
        plt.figure()
        sns.distplot(specified_num, hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Specified')
        sns.distplot(activated_num, hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Activated')
        plt.xlabel(' # Scan Chain')
        plt.ylabel('Density')
        plt.legend()

        if not os.path.isdir('figs/'):
                os.makedirs(os.path.dirname('figs/'))
        plt.savefig('figs/hist_{}_{}_{}_{}.png'.format(self.mode, self.upper_bound, self.sim_constraint, self.mux_ctrl))

        specified_percentage = specified_num.sum() / (self.num_merged_cube * self.num_id)
        activated_percentage = activated_num.sum() / (self.num_merged_cube * self.num_id)

        print('Specified scan chain percentage after merging is {:.2f}%.'.format(100.*specified_percentage))
        print('Activaed scan chain percentages is {:.2f}%.'.format(100.*activated_percentage))


def get_args():
    '''
    Arguments for 2D encoding structure.
    '''
    args = argparse.ArgumentParser(add_help=False,
                                    description='Arguments for 2D encoding structure')
    
    args.add_argument('--map_mode',
                        default='stochastic', type=str,
                        help='The grouping/mapping mode')


    args.add_argument('--upper_bound', 
                        default=0.5, type=float,
                        help='The upper bound of specified scan chain per test cube after merging')
    args.add_argument('--sim_constraint',
                        default=0.15, type=float,
                        help='The constraint of similirity between two grouping approaches')

    args.add_argument('--seed',
                        default=0, type=int,
                        help='seed value')
    
    args.add_argument('--mux_ctrl',
                        default=3, type=int,
                        help='The control bits for MUX')

    args.add_argument('--num_compare',
                        default=10, type=int,
                        help='The number of test cubes to check')
    return args.parse_args()




def main(args):
    args = get_args()
    # initilize and evaluate
    # create_mlb()
    encoder = TwoDimEncoding('data/mlb.npy', map_mode=args.map_mode, upper_bound=args.upper_bound, 
                                sim_constraint=args.sim_constraint, seed=args.seed, mux_ctrl=args.mux_ctrl)
    encoder.merging()
    encoder.encoding()
    encoder.eval()


if __name__ == '__main__':
    main(get_args())

