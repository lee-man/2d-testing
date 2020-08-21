import csv
import os
import random
import math
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import copy
import argparse
import logging
logging.basicConfig(level=logging.INFO)

from sklearn import metrics
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import seaborn as sns


def create_mlb():
    ###########################
    # 1. get the unique id list 
    num_sc = 0
    num_exp = 0
    logging.info('#' * 15)
    logging.info('Get the unique id list')
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
    logging.info('The length of id list is {}'.format(num_id))
    logging.info('The size of testing data is {}'.format(num_exp))

    ###########################
    # 2. Multi-Label Binarizer
    num_id = 339
    # Create the Multi-Label Matrix
    logging.info('Create Multi-Label Binarizer')
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
    logging.info('The # and percentage of activated scan chains are {:.2f} and {:.2f}%.'.format(num_sc / num_exp, \
            100. * num_sc / (num_exp * num_id)))

    np.save('data/mlb.npy', mlb)


##########################
# 3. Merging and Encoding using 2D structure or EDT structure
class TwoDimEncoding(object):
    '''
    The class for Two-Dimention Low-Power Encoding.

    '''
    def __init__(self, mlb_path, group_ctrl=19, chain_ctrl=18, mux_ctrl=3, upper_bound=0.5, sim_constraint=0.5, map_mode='stochastic', seed=0, num_compare=10, conflict='sc'):
        # self.mlb = np.load(mlb_path)[:10000]
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
        self.num_compare = num_compare
        self.conflict = conflict
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
        if self.conflict == 'cell':
            self.create_mlb_with_cell()
        
    
    def _print_info(self):
        logging.info('*****Statistic：')
        logging.info('The size of testing dataset is {}'.format(self.num_cube))
        logging.info('The size of each test cube is {}'.format(self.num_id))
        logging.info('Control bits settings:{} chain ctrl, {} group ctrl and {} mux crtl'.format(self.chain_ctrl, self.group_ctrl, self.mux_ctrl))
        logging.info('The upper bound of activated scan chian for low power encoding is {}.'.format(self.upper_bound))
        logging.info('The grouping method is {}.'.format(self.mode))
        logging.info('The conflict model is {}'.format(self.conflict))

    def _set_seed(self):
        # Set Seed
        np.random.seed(self.seed)
        random.seed(self.seed)

    def create_mlb_with_cell(self, mean=10, density=500, std=None):
        # cell = np.zeros(self.num_id, density)
        logging.info('Started to create mlb with cell attribute.')
        mlb_w_cell = np.random.choice([0, 1], (self.num_cube, self.num_id, density), [1-mean/density, mean/density]).astype(float)
        self.mlb = mlb_w_cell * np.expand_dims(self.mlb, axis=2)


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
        cube = np.zeros(cube1.shape)
        cube = ((cube1 + cube2) > 0).astype(float)
        # for i in range(cube1.shape[0]):
        #     if cube1[i] == 1 or cube2[i] == 1:
        #         cube[i] = 1
        return cube
    
    
    def calculate_specified_percentage(self, cube):
        return cube.sum()/cube.shape[0]

    def calculate_activated_percentage(self, merged_cube, to_merged_cube):
        cube = self.merge_two_cube(merged_cube, to_merged_cube)
        if len(cube.shape) == 2:
            cube_wo_cell = (cube.sum(axis=1) > 0).astype(float)
        else:
            cube_wo_cell = cube
        group_bit = np.zeros((self.mux_ctrl, self.group_ctrl))
        chain_bit = np.zeros((self.mux_ctrl, self.chain_ctrl))
        for mux_bit in range(self.mux_ctrl):
                for (ele_id, ele) in enumerate(cube_wo_cell):
                    if ele == 1.0:
                        group_bit[mux_bit, self.group_mapping[mux_bit][str(ele_id)] // self.chain_ctrl] = 1
                        chain_bit[mux_bit, self.group_mapping[mux_bit][str(ele_id)] % self.chain_ctrl] = 1
        encoded_mux = np.argmin(group_bit.sum(axis=1) * chain_bit.sum(axis=1))
        activated_num = group_bit[encoded_mux].sum() \
                            * chain_bit[encoded_mux].sum()
        return activated_num / cube.shape[0], cube




    def merging(self):
        logging.info('*' * 15)
        logging.info('Start Merging.')
        mlb = copy.deepcopy(self.mlb)
        mask = np.zeros(mlb.shape[0])
        idx_now = 0
        mask[0] = 1
        merged_array = []
        merged_cube = copy.deepcopy(mlb[idx_now])
        while idx_now < (mlb.shape[0] - 1):
            for id in range(idx_now+1, mlb.shape[0]):
                row = mlb[id]
                if id == (mlb.shape[0] - 1):
                    if mask[id] != 1:
                        activated_percentage, merged_cube_candidate = self.calculate_activated_percentage(merged_cube, row)
                        if activated_percentage <= self.upper_bound:
                            merged_cube = merged_cube_candidate
                            mask[id] = 1
                    merged_array.append(merged_cube)
                    while mask[idx_now] == 1 and idx_now < (mlb.shape[0] - 1):
                        idx_now += 1
                    mask[idx_now] = 1
                    merged_cube = copy.deepcopy(mlb[idx_now])
                    # break
                elif mask[id] == 1:
                    continue
                elif self.check_conflict(merged_cube, row):
                    activated_percentage, merged_cube_candidate = self.calculate_activated_percentage(merged_cube, row)
                    if activated_percentage <= self.upper_bound:
                        if self.mode == 'random':
                            merged_cube = merged_cube_candidate
                            mask[id] = 1
                        elif self.mode == 'stochastic':
                            # heuristic merge: look forward 10 steps
                            if self.num_compare:
                                count = 0
                                z = id + 1
                                id_compare = id
                                while (count < self.num_compare) and (z < mlb.shape[0]):
                                    if mask[z] == 1:
                                        z += 1
                                        continue
                                    elif self.check_conflict(merged_cube, mlb[z]):
                                        activated_percentage_z, merged_cube_candidate_z = self.calculate_activated_percentage(merged_cube, mlb[z])
                                        if (activated_percentage_z <= self.upper_bound) and (activated_percentage_z <= activated_percentage):
                                            id_compare = z
                                            merged_cube_candidate = merged_cube_candidate_z
                                            count += 1
                                    z += 1
                                mask[id_compare] = 1
                                merged_cube = merged_cube_candidate
                            else:
                                merged_cube = merged_cube_candidate
                                mask[id] = 1     

        if self.conflict == 'cell':
            merged_array = np.array(merged_array)
            self.merged_array = (np.array(merged_array).sum(axis=2) > 0).astype(float)
        else:
            self.merged_array = np.array(merged_array)

    def encoding(self):
        logging.info('*' * 15)
        logging.info('Start Encoding.')
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
        logging.info('*' * 15)
        logging.info('Evalutation.')
        specified_num = np.zeros(self.num_merged_cube)
        activated_num = np.zeros(self.num_merged_cube)
        self.num_merged_cube = self.merged_array.shape[0]
        logging.info('Total number of merged test cube is {}'.format(self.num_merged_cube))
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
        plt.savefig('figs/hist_{}_{}_{}_{}_{}.png'.format(self.mode, self.conflict, self.upper_bound, self.sim_constraint, self.mux_ctrl))

        specified_percentage = specified_num.sum() / (self.num_merged_cube * self.num_id)
        activated_percentage = activated_num.sum() / (self.num_merged_cube * self.num_id)

        logging.info('Specified scan chain percentage after merging is {:.2f}%.'.format(100.*specified_percentage))
        logging.info('Activaed scan chain percentages is {:.2f}%.'.format(100.*activated_percentage))


class EDTEncoder(object):
    '''
    The class for EDT encoder, under some prior probabilistic model
    '''
    def __init__(self, mlb_path, edt_ctrl=37, upper_bound=0.5):
        self.mlb = np.load(mlb_path)
        self.num_cube = self.mlb.shape[0]
        self.num_id = self.mlb.shape[1]
        self.edt_ctrl = edt_ctrl
        self.upper_bound = upper_bound
        self.assign_prob()


    
    def assign_prob(self, draw=True):
        self.prob_success = np.zeros(self.num_id)
        self.prob_success[:self.edt_ctrl] = 1
        self.prob_success[self.edt_ctrl:] = np.power(0.5, range(self.num_id - self.edt_ctrl))

        if draw:
            plt.figure()
            plt.bar(range(self.num_id), self.prob_success)
            plt.xlabel('Scan Chian ID')
            plt.ylabel('Encoding Success Rate')
            if not os.path.isdir('figs/'):
                os.makedirs(os.path.dirname('figs/'))
            plt.savefig('figs/encoding_prob.png')


        


def get_args():
    '''
    Arguments for 2D encoding structure or EDT structure.
    '''
    args = argparse.ArgumentParser(add_help=False,
                                    description='Arguments for 2D encoding structure')
    
    args.add_argument('--encoder_model',
                        default='2D', type=str, choices=['2D', 'EDT'],
                        help='The encoder model: 2D strcuture or EDT structure')

    args.add_argument('--map_mode',
                        default='stochastic', type=str, choices=['stochastic', 'random'],
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
                        default=0, type=int,
                        help='The number of test cubes to check')
    
    args.add_argument('--conflict_model',
                        default='sc', type=str, choices=['sc', 'cell'],
                        help='The conflict model to use')

    return args.parse_args()




def main(args):
    args = get_args()
    # initilize and evaluate
    # create_mlb()
    if args.encoder_model == '2D':
        encoder = TwoDimEncoding('data/mlb.npy', map_mode=args.map_mode, upper_bound=args.upper_bound, 
                                    sim_constraint=args.sim_constraint, seed=args.seed, mux_ctrl=args.mux_ctrl, num_compare=args.num_compare,
                                    conflict=args.conflict_model)
        encoder.merging()
        encoder.encoding()
        encoder.eval()
    else:
        encoder = EDTEncoder('data/mlb.npy')
    


if __name__ == '__main__':
    main(get_args())

