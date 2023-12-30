##############################################################################################
# This program is used to assign OD demand on network with link capacity constraints and flow priority
# Inputs: Passenger graph and set of passenger groups
# Output: Flow function
# Note: 1. the time is recorded as the minutes difference compared to a reference time (e.g. 7:00 am).
#       2. OD demand is time, demand in pandas.
#       3. Dictionaries in Python are implemented as hash tables, there is no ordering. To track the order of joined
#          queue with arrival times, it needs a list of order of the arrival times.
#       4. tap_in demand is in time period, for example 7:15 = 7:00-7:15

import _NPM_egine_path_choice
import _blackbox

import sys

import _DefaultValues
import pandas as pd
import os
import numpy as np
import sys
import copy
import time
import random
import scipy
import math
import multiprocessing as mp

#
import random

from deap import base
from deap import creator
from deap import tools

pd.options.mode.chained_assignment = None  # default='warn'


class OptimizationModel(object):
    def __init__(self, model_file_name, REFERENCE_TIME, time_interval_demand):
        self.test_name = model_file_name.split('_')[1]
        self.SIM_TIME_PERIOD = model_file_name.split('_')[2].split('-')
        self.input_file_path = model_file_name + '/'
        self.ITINERARY = pd.DataFrame()
        self.EVENTS = pd.DataFrame()
        self.CARRIERS = pd.DataFrame()
        self.QUEUES = pd.DataFrame()
        self.TB_TXN_RAW = pd.DataFrame()
        self.PATH_ATTRIBUTE = pd.DataFrame()
        self.TRANSFER_WT = pd.DataFrame()
        self.NETWORK = pd.DataFrame()
        self.OPERATION_ARRANGEMENT = pd.DataFrame()
        self.EMPTY_TRAIN_TIME_LIST = []
        self.TB_TXN = pd.DataFrame()
        self.TB_EXIT_DEMAND = pd.DataFrame()
        self.QUEUES_POOL = {}
        self.LIST_CARRIERS = []
        self.opt_time_period = []
        self.REFERENCE_TIME = REFERENCE_TIME
        # self.intial_beta = beta
        self.time_interval = time_interval_demand
        self.support_input = {}

    def Load_input_files(self, demand_name, GENERATE):

        # Variable settings
        path_external_data = 'data/'  # input data are in External_data folder

        self.ITINERARY = pd.read_csv(self.input_file_path + 'tb_itinerary.csv')  # Itinerary table
        self.EVENTS = pd.read_csv(self.input_file_path + 'tb_event.csv')  # Event list
        self.CARRIERS = pd.read_csv(self.input_file_path + 'tb_carrier.csv', index_col=0)  # Carrier table
        self.QUEUES = pd.read_csv(self.input_file_path + 'tb_queue.csv', index_col=0)
        self.TB_TXN_RAW = pd.read_csv(self.input_file_path + 'tb_txn.csv')
        self.PATH_ATTRIBUTE = pd.read_csv(self.input_file_path + 'path_attributes_for_opt.csv')
        self.TRANSFER_WT = pd.read_csv(path_external_data + 'Transfer_Walking_Time.csv')
        self.ACC_EGR_TIME = pd.read_csv(path_external_data + 'Access_egress_time.csv')
        self.NETWORK = pd.read_csv(self.input_file_path + 'tb_network.csv')  # Service network information
        self.OPERATION_ARRANGEMENT = pd.read_csv(path_external_data + 'Empty_Train_Arrangement.csv')

        # ****************process empty train*****************
        EMPTY_TRAIN_TIME_LIST = []
        operation_control = self.OPERATION_ARRANGEMENT.loc[self.OPERATION_ARRANGEMENT.test_name == self.test_name]
        if len(operation_control) > 0:
            operation_control.loc[:, 'dispatch_time'] = pd.to_timedelta(operation_control.time).dt.total_seconds() - \
                                                        pd.to_timedelta(self.REFERENCE_TIME).total_seconds()
            for index, operation_info in operation_control.iterrows():
                event_temp = self.EVENTS.copy()
                event_temp['carrier'] = event_temp['carrier_id'].apply(lambda x: x.split('_'))
                event_temp['line'] = event_temp['carrier'].apply(lambda x: int(x[0]))
                event_temp['dir'] = event_temp['carrier'].apply(lambda x: int(x[1]))
                event_temp = event_temp.loc[(event_temp['event_type'] == 1) &
                                            (event_temp['line'] == operation_info.line) &
                                            (event_temp['dir'] == operation_info.direction) &
                                            (event_temp['event_station'] == operation_info.station)]  # departure
                event_temp['time_diff'] = event_temp['event_time'] - operation_info.dispatch_time
                event_temp['time_diff'] = event_temp['time_diff'].abs()
                if len(event_temp) > 0:
                    min_index = event_temp['time_diff'].idxmin()
                    EMPTY_TRAIN_TIME_LIST.append(
                        (self.EVENTS.loc[min_index, 'carrier_id'], self.EVENTS.loc[min_index, 'event_time']))
        # ****************************************************

        self.EMPTY_TRAIN_TIME_LIST = EMPTY_TRAIN_TIME_LIST


        self.opt_time_period = [int(self.SIM_TIME_PERIOD[0]),
                                int(self.SIM_TIME_PERIOD[1])]

        self.support_input = {'ITINERARY': self.ITINERARY, 'EVENTS': self.EVENTS, 'CARRIERS': self.CARRIERS,
                              'QUEUES': self.QUEUES, 'TB_TXN_RAW': self.TB_TXN_RAW,
                              'PATH_ATTRIBUTE': self.PATH_ATTRIBUTE, 'TRANSFER_WT': self.TRANSFER_WT,
                              'ACC_EGR_TIME': self.ACC_EGR_TIME,
                              'NETWORK': self.NETWORK, 'OPERATION_ARRANGEMENT': self.OPERATION_ARRANGEMENT,
                              'EMPTY_TRAIN_TIME_LIST': self.EMPTY_TRAIN_TIME_LIST}
        # ----
        if GENERATE:
            self.TB_EXIT_DEMAND_raw = pd.DataFrame()
        else:
            self.TB_EXIT_DEMAND_raw = pd.read_csv(
                self.input_file_path + 'exit_demand_synthesized_' + demand_name + '.csv')
            # ----------------
            self.TB_EXIT_DEMAND_raw = self.TB_EXIT_DEMAND_raw[
                (self.TB_EXIT_DEMAND_raw.exit_time >= self.opt_time_period[0]) & (
                            self.TB_EXIT_DEMAND_raw.exit_time <= self.opt_time_period[1])]
            self.TB_EXIT_DEMAND_raw['time'] = self.TB_EXIT_DEMAND_raw[
                                                  'exit_time'] // self.time_interval * self.time_interval
            # print (OD_state)
            self.TB_EXIT_DEMAND = \
            self.TB_EXIT_DEMAND_raw.groupby([self.TB_EXIT_DEMAND_raw.origin, self.TB_EXIT_DEMAND_raw.destination,
                                             self.TB_EXIT_DEMAND_raw.time])['flow'].sum().reset_index(drop=False)
            self.TB_EXIT_DEMAND = self.TB_EXIT_DEMAND.rename(columns={'flow': 'demand'})

    def BlackboxFunc(self, beta):
        beta_path = beta[0:4]
        OD_state, passenger_state, station_state, carrier_state = _NPM_egine_path_choice. \
            NPMModel(self.support_input, beta_path, path_share=[], static=False, info_print_num=100).run_assignment(
            out_put_all=False)
        # OD_state.to_csv('NPM_egine_true_OD_state_ForFLOW.csv',index=False)
        print('current beta is:', beta)
        OD_state['time'] = OD_state['exit_time'] // self.time_interval * self.time_interval
        OD_state = OD_state.loc[(OD_state.time >= self.opt_time_period[0]) & (OD_state.time <= self.opt_time_period[1])]
        w1 = 1
        w2 = 600
        exit_flow_diff = self.BlackboxFunc_15min_OD(OD_state)
        entropy = self.BlackboxFunc_Entropy(OD_state)
        return w1 * exit_flow_diff + w2 * entropy

    def get_OD_state(self, beta_all):
        beta_path = beta_all[0:4]
        OD_state, passenger_state, station_state, carrier_state = _NPM_egine_path_choice. \
            NPMModel(self.support_input, beta_path, path_share=[], static=False, info_print_num=100).run_assignment(
            out_put_all=False)
        return OD_state

    def BlackboxFunc_Entropy(self, OD_state):
        TB_EXIT_DEMAND_Entropy = self.TB_EXIT_DEMAND_raw.groupby(
            [self.TB_EXIT_DEMAND_raw.origin, self.TB_EXIT_DEMAND_raw.destination,
             self.TB_EXIT_DEMAND_raw.time]).filter(lambda x: len(x) > 50)  # MORE THAN 50 samples
        TB_EXIT_DEMAND_Entropy['travel_time'] = TB_EXIT_DEMAND_Entropy['exit_time'] - TB_EXIT_DEMAND_Entropy[
            'entry_time']
        TB_EXIT_DEMAND_Entropy = TB_EXIT_DEMAND_Entropy.groupby(['origin', 'destination', 'time']). \
            apply(lambda x: np.histogram(x['travel_time'], bins=7, density=False)[0] / len(x)).reset_index().rename(
            columns={0: 'tt_bin'})  # 7 different time invertals

        OD_state = OD_state.groupby([OD_state.origin, OD_state.destination,
                                     OD_state.time]).filter(lambda x: len(x) > 50)  # MORE THAN 50 samples
        OD_state['travel_time'] = OD_state['exit_time'] - OD_state['entry_time']
        OD_state = OD_state.groupby(['origin', 'destination', 'time']). \
            apply(lambda x: np.histogram(x['travel_time'], bins=7, density=False)[0] / len(x)).reset_index().rename(
            columns={0: 'tt_bin'})  # 7 different time invertals
        if len(OD_state) > 0:
            # Merge the est flow with exit demand data
            flow_merge = OD_state.merge(TB_EXIT_DEMAND_Entropy, on=['origin', 'destination', 'time'], how='inner')
            flow_merge['KL_div'] = -99
            for index, row in flow_merge.iterrows():
                if len(row['tt_bin_x']) == len(row['tt_bin_y']):
                    flow_merge.loc[index, 'KL_div'] = scipy.stats.entropy(row['tt_bin_x'], row['tt_bin_y'])
            # flow_merge.to_csv('Entropy_record.csv', index=False)
            flow_merge = flow_merge.loc[(flow_merge['KL_div'] != -99) &
                                        (flow_merge['KL_div'] != np.inf)]
            travel_time_distribution_entropy = flow_merge['KL_div'].sum()
            print('KL_div is', travel_time_distribution_entropy)
            return travel_time_distribution_entropy
        else:
            print('Error occurs is KL div calculation ...')
            return np.inf

    def BlackboxFunc_15min_OD(self, OD_state):
        """ Objective using exit demand flow seems to be not very sensitive
        Other options: K-L divergence of journey time distribution for Origin_Destination_Time
        """
        # Run the NPM model and get the OD_state
        # Make the inputs consistent with different optimization solver

        # print (OD_state)
        est_flow = OD_state.groupby([OD_state.origin, OD_state.destination,
                                     OD_state.time])['flow'].sum().reset_index(drop=False)
        # print (est_flow)
        # Only keep the record within specified optimization time period

        if len(est_flow) > 0:
            # Merge the est flow with exit demand data
            est_act_exit_flow = est_flow.merge(self.TB_EXIT_DEMAND, on=['origin', 'destination', 'time'], how='inner')

            # Calculate the difference between estimated flow (from npm model) and the actual (from afc and sjc data)
            est_act_exit_flow['diff'] = est_act_exit_flow['demand'] - est_act_exit_flow['flow']
            est_act_exit_flow['squared_diff'] = est_act_exit_flow['diff'] ** 2

            # the objective value is the squared root of the estimated and actual exit flow difference
            # obj = (est_act_exit_flow['squared_diff'].sum() / len(est_act_exit_flow)) ** 0.5
            print('num of od pair:', len(est_act_exit_flow))
            obj = est_act_exit_flow['squared_diff'].sum()

            # print('objective function value is %s' % obj)
            # The Bayesian optimizer MAXIMIZE the objective function
            print('OD diff is', obj)
            return obj
        else:
            print('Error occurs in OD diff calculation ...')
            return np.inf



def get_random_beta(bound):
    bound_sort = bound
    return [random.uniform(bd[0], bd[1]) for bd in bound_sort]


# def test_for_beta(opt_model, beta_list):
#     record = pd.DataFrame()
#     for beta in beta_list:
#         obj = opt_model.BlackboxFunc(beta)
#         new_output = pd.DataFrame(
#             {'x1': [beta[0]], 'x2': [beta[1]],
#              'x3': [beta[2]],
#              'x4': [beta[3]], 'x5': [beta[4]], 'target': [obj]})
#         record = pd.concat([record,new_output], ignore_index=True)
#         print ('beta is:',beta)
#         print ('objective function is:', obj)
#     record.to_csv('Beta_test_results.csv',index=False)


def Generate_synthetic_demand(opt_model, beta_all, output_file_name):
    OD_state = opt_model.get_OD_state(beta_all)
    OD_state.to_csv(opt_model.input_file_path + 'exit_demand_' + output_file_name + '.csv', index=False)


# -------------------------------------------- Main ---------------------------------------------------------
if __name__ == "__main__":
    '''
        x1 * self.tb_path_attribute['in_vehicle_time'] +
        x2 * self.tb_path_attribute['transfer_over_dist'] +
        x3 * self.tb_path_attribute['no_of_transfer'] +
        x4 * self.tb_path_attribute['commonality_factor'] 
    '''

    tic = time.time()
    model_file_name = 'Case_HK-18-19_64800-68400'
    time_interval_demand = 15 * 60  # resolution of exit OD


    # make sure to change the bound when change name
    output_file_name = 'uniform'


    # ===============
    my_opt = OptimizationModel(model_file_name, _DefaultValues.REFERENCE_TIME, time_interval_demand)
    my_opt.Load_input_files(output_file_name, GENERATE = True)

    True_beta_list = [0, 0, 0, 0]

    beta_all = True_beta_list

    Generate_synthetic_demand(my_opt, beta_all, output_file_name)


