# ****************************** Generate input files from smart card data and choice model ***************************
# Inputs: AFC transactions, Timetable (or Train Movement Data), Network file ()
#
# Code: direction {Down = 2, Up = 1, toward central or not}
#       event_type {departure = 1, arrival = 0}
#       time period {reference time 7:00 = 0, 7: 15 = 15*60 = [0-15*60]}
#       line code
#       carrier_id {line_direction_departureTimeFromFirstStation}
#       queue_id {station_line_direction}
#       passenger_id {origin_destination_path_departureTimeFromFirstStation}
# tb_demand: origin, destination, path, itinerary = 1, time, demand
#            [1, 4, 1, 1, 0, 60] = 60 pax travel from 1->4 using path 1 with initial itinerary 1
# tb_itinerary: origin, destination, path, itinerary, boarding_station, alighting_station, line, direction
#            [1,4,1,1,1,3,1,1] = Itinerary 1 of path 1 for OD 1->4 with board 1 and alight 4 using line 1, direction 1
# tb_queue: queue_id,queue_capacity,queue_station,queue_line,queue_direction,initial_time_sim
#            [1_1_1,100,1,1,1,-5] = queue_id (station, line, direction), capacity 100, station 1, line 1, direction 1,
#                                   initial time of simulation -5 seconds
# tb_network: carrier_line,carrier_direction,carrier_first_station,carrier_last_station,link_sequence,
#             link_start,link_end
#             [1,1,1,3,1,1,3] = line 1 with direction 1, first station 1 and end station 3, the first link is from
#                               station 1 to 3
# tb_carrier: carrier_id,carrier_capacity,carrier_line,carrier_direction,carrier_arrive_first_station,
#             carrier_first_station,carrier_last_station
#             [1_1_0,100,1,1,1,3] = carrier 1_1_0 operating on line 1, direction 1, departure the first station
# list_event: event_id,event_time,event_station,event_type,carrier_id
#             [1,0,1,0,1_1_0] = carrier 1_1_0 arrives (0) at station 1 at time 0
# ****************************** Generate input files from smart card data and choice model ***************************

from _DefaultValues import *
import pandas as pd
import time
import datetime as dt
import copy
import os
import random
import numpy as np
from _postprocess_mtr_network_operation import post_process

pd.options.mode.chained_assignment = None  # default='warn'


# Define generic functions
def generate_carrier_id(carrier_line, carrier_direction, carrier_trip):
    return str(int(carrier_line)) + '_' + str(int(carrier_direction)) + '_' + str(carrier_trip)


def generate_queue_id(queue_station, queue_line, queue_direction):
    return str(int(queue_station)) + '_' + str(int(queue_line)) + '_' + str(int(queue_direction))


def process_afc_sjsc(df, flag):
    # 1. Data formatting
    # Formatting the columns names, change the timestamp format to YYYY-MM-DD,
    # and extract useful columns for further use
    # Output formats: user_id, txn_dt, txn_type_co,txn_subtype_co,entry_stn,txn_loc,txn_audit_no
    # Input formats:
    #     octopus: csc_phy_id,business_dt,txn_dt,txn_type_co,txn_subtype_co,train_entry_stn,txn_loc,txn_audit_no,
    #              hw_type_co,mach_no,train_direct_ind,txn_value,modal_disc_value,csc_rv_value
    #     sjtc: sjsc_id,business_dt,txn_dt,txn_type_co,txn_subtype_co,entry_stn,txn_loc,txn_seq_no,
    #           mach_no,recycle_count,prom_code,txn_value
    df.columns = map(str.lower, df.columns)
    # mtr_format = DATE.split('-')[2] + '/' + DATE.split('-')[1] + '/' + DATE.split('-')[0]
    # df['txn_dt'] = df['txn_dt'].str.replace(mtr_format, DATE)
    df['txn_dt'] = pd.to_datetime(df['txn_dt'], infer_datetime_format=True).dt.time.astype(str)

    if flag == 'octopus':
        df.rename(columns={'csc_phy_id': 'user_id', 'train_entry_stn': 'entry_stn'}, inplace=True)
    elif flag == 'sjtc':
        df.rename(columns={'sjsc_id': 'user_id', 'txn_seq_no': 'txn_audit_no'}, inplace=True)

    # 2. construct
    return df[['user_id', 'txn_dt', 'txn_type_co', 'txn_subtype_co', 'entry_stn', 'txn_loc', 'txn_audit_no']]


def process_network_operation(network_operation):

    int_list = ['origin', 'destination', 'line', 'direction', 'station', 'link_start', 'link_end']
    network_operation[int_list] = network_operation[int_list].astype(int)

    return network_operation


# Define classes
class FilePreparation(object):
    # reference time, e.g. '6:00'
    # time_period, demand time period, e.g. ['7:00', '9:00']
    # demand_time_interval, e.g. 15
    # demand_exit_or_entry, e.g. 'entry'

    def __init__(self, sim_time_period, file_path, demand_file_path, file_network_operation,
                 file_time_table, output_file_path, route_choice_interval=CHOICE_INTERVAL):
        # initialize the variables

        self.route_choice_interval = route_choice_interval
        self.reference_time = pd.to_timedelta(REFERENCE_TIME).total_seconds()
        self.exit_entry = 'exit'  # generate the actual exit demand file for optimization use

        self.file_od_matrix = demand_file_path
        self.output_file_path = output_file_path

        self.sim_time_period = sim_time_period
        self.queue_capacity = 100000
        self.cool_down_time = DEFAULT_COOL_DOWN_TIME * 60  # 15 * 60 seconds
        self.warm_up_time = DEFAULT_WARM_UP_TIME * 60  # 15 * 60 seconds

        self.demand_file_path = demand_file_path

        self.carrier_time_period = [
            pd.to_timedelta(sim_time_period[0]).total_seconds() - self.warm_up_time - self.reference_time,
            pd.to_timedelta(sim_time_period[1]).total_seconds() + self.cool_down_time - self.reference_time]
        self.txn_time_period = [self.carrier_time_period[0], self.carrier_time_period[1] + 60 * 60]

        # reading and processing the external files
        self.network_operation = file_network_operation
        self.network_operation = process_network_operation(self.network_operation)
        self.time_table = file_time_table
        print('Start time table preprocess')
        tic = time.time()
        self.initialize_time_table()
        print('Finish time table preprocess', time.time() - tic, 'sec')
        # initialize the final generated tables
        self.tb_itinerary = pd.DataFrame(index=range(0, len(self.network_operation)),
                                         columns=['origin', 'destination', 'path', 'itinerary', 'boarding_station',
                                                  'alighting_station', 'line', 'direction'])
        self.tb_carrier = pd.DataFrame(index=range(0, len(self.time_table)),
                                       columns=['carrier_id', 'carrier_car_no', 'carrier_line', 'carrier_direction',
                                                'carrier_trip',
                                                'carrier_arrive_first_station', 'carrier_first_station',
                                                'carrier_last_station',
                                                'carrier_serve_stations'])
        self.tb_network = pd.DataFrame(index=range(0, len(self.time_table)),
                                       columns=['carrier_line', 'carrier_direction', 'carrier_first_station',
                                                'carrier_last_station', 'link_sequence', 'link_start', 'link_end'])
        self.tb_queue = pd.DataFrame(index=range(0, len(self.time_table)),
                                     columns=['queue_id', 'queue_capacity', 'queue_station', 'queue_line',
                                              'queue_direction', 'initial_time_sim'])
        self.list_event = pd.DataFrame()
        self.path_share = pd.DataFrame()
        self.tb_txn = pd.DataFrame()
        self.TAPIN_ini = {}
        self.initial_sim_time_list = pd.DataFrame(index=range(0, len(self.time_table)),
                                                  columns=['line', 'direction', 'trip_id', 'station',
                                                           'initial_sim_time'])

    def generate_input_files(self):
        tic = time.time()
        print('generating tb_txn...')
        self.create_tb_txn()
        print('tb_txn generated successfully with elapsed time %s seconds' % (time.time() - tic))

        tic = time.time()
        print('generating tb_path_share...')
        self.generate_path_share()
        print('tb_path_share generated successfully with elapsed time %s seconds' % (time.time() - tic))

        tic = time.time()
        print('generating path_attributes_for_opt...')
        self.generate_path_attributes()
        print('path_attributes_for_opt generated successfully with elapsed time %s seconds' % (time.time() - tic))

        tic = time.time()
        print('generating tb_itinerary...')
        self.create_tb_itinerary()
        print('tb_itinerary generated successfully with elapsed time %s seconds' % (time.time() - tic))

        tic = time.time()
        print('generating tb_carrier and tb_network ...')
        self.create_tb_carrier_network()
        print('tb_carrier_network generated successfully with elapsed time %s seconds' % (time.time() - tic))

        tic = time.time()
        print('generating tb_queue ...')
        self.create_tb_queue()
        print('tb_queue generated successfully with elapsed time %s seconds' % (time.time() - tic))

        tic = time.time()
        print('generating tb_event ...')
        self.create_event_list()
        print('tb_event generated successfully with elapsed time %s seconds' % (time.time() - tic))



    def create_tb_txn(self):
        self.tb_txn = pd.read_csv(self.demand_file_path)
        self.tb_txn.to_csv(self.output_file_path + 'tb_txn.csv', index=False)

    def mtr_choice_period(self, pd_series_time, period):
        # MP: 5:15-10:00
        # OP: 10:00-16:00
        # EP: 16:00-19:30
        # Other: 19:30 - 23:00

        if period == 'MP':
            return (pd_series_time > 5 * 60 * 60 + 15 * 60 - self.reference_time) & (
                        pd_series_time <= 10 * 60 * 60 - self.reference_time)
        if period == 'OP':
            return (pd_series_time > 10 * 60 * 60 - self.reference_time) & (
                        pd_series_time <= 16 * 60 * 60 - self.reference_time)
        if period == 'EP':
            return (pd_series_time > 16 * 60 * 60 - self.reference_time) & (
                        pd_series_time <= 19 * 60 * 60 + 30 * 60 - self.reference_time)
        if period == 'OTHER':
            return (pd_series_time > 19 * 60 * 60 + 30 * 60 - self.reference_time) & (
                        pd_series_time <= 23 * 60 * 60 - self.reference_time)

    def generate_path_share(self):

        # Mimic OD transactions
        txn_time_list = np.arange(self.txn_time_period[0] // self.route_choice_interval * self.route_choice_interval,
                                  self.txn_time_period[1] // self.route_choice_interval * self.route_choice_interval,
                                  self.route_choice_interval)

        # Generate the template od_t table
        od_path = \
            self.network_operation[['origin', 'destination', 'path_id', 'morning_peak_path_share', 'off_peak_path_share', 'evening_peak_path_share',
                                    'other_time_path_share']].drop_duplicates()
        od_path_time = copy.deepcopy(od_path)
        od_path_time['TIME_INTERVAL'] = txn_time_list[0]

        for k in np.arange(1, len(txn_time_list)):
            od_path['TIME_INTERVAL'] = txn_time_list[k]
            od_path_time = pd.concat([od_path_time, od_path])

        # generate path share
        od_path_time['If_MP'] = self.mtr_choice_period(od_path_time['TIME_INTERVAL'], 'MP')
        od_path_time['If_OP'] = self.mtr_choice_period(od_path_time['TIME_INTERVAL'], 'OP')
        od_path_time['If_EP'] = self.mtr_choice_period(od_path_time['TIME_INTERVAL'], 'EP')
        od_path_time['If_OTHER'] = self.mtr_choice_period(od_path_time['TIME_INTERVAL'], 'OTHER')

        # Check un-matched
        od_path_time['Sum_Judge'] = od_path_time['If_MP'] | od_path_time['If_OP'] | od_path_time['If_EP'] | \
                                    od_path_time['If_OTHER']
        od_path_time.loc[~od_path_time['Sum_Judge'], 'If_OTHER'] = True  # Assign it to other path share
        path_share = np.sum(np.array(od_path_time[['If_MP', 'If_OP', 'If_EP', 'If_OTHER']]) * np.array(
            od_path_time[['morning_peak_path_share', 'off_peak_path_share', 'evening_peak_path_share', 'other_time_path_share']]), axis=1)
        od_path_time['path_share'] = path_share

        self.path_share = od_path_time[['origin', 'destination', 'path_id', 'TIME_INTERVAL', 'path_share']]
        self.path_share.rename(columns={'origin': 'origin', 'destination': 'destination', 'path_id': 'path_id',
                                        'TIME_INTERVAL': 'time_interval'}, inplace=True)

        int_list = ['origin', 'destination', 'time_interval', 'path_id']
        self.path_share[int_list] = self.path_share[int_list].astype(int)
        self.path_share = self.path_share.sort_values(by=['origin', 'destination', 'path_id'])
        self.path_share.to_csv(self.output_file_path + 'tb_path_share.csv', index=False)

    def generate_path_attributes(self):
        # Generate the template od_t table
        # Mimic OD transactions
        txn_time_list = np.arange(self.txn_time_period[0] // self.route_choice_interval * self.route_choice_interval,
                                  self.txn_time_period[1] // self.route_choice_interval * self.route_choice_interval,
                                  self.route_choice_interval)

        # Generate the template od_t table

        od_path = \
            self.network_operation.loc[:,
            ['origin', 'destination', 'path_id', 'line', 'direction', 'station', 'link_start',
             'link_end', 'transfer_or_not', 'link_travel_time']].drop_duplicates().reset_index(drop=True)
        od_path = od_path.sort_values(by=['origin', 'destination', 'path_id', 'link_travel_time'],
                                      ascending=[True, True, True, False])
        path_att = od_path.loc[:, ['origin', 'destination', 'path_id']].drop_duplicates().reset_index(drop=True)

        # in-veh time
        temp = od_path.groupby(['origin', 'destination', 'path_id'], sort=False).first()[
            ['link_travel_time']].reset_index(drop=False)
        path_att = path_att.merge(temp, left_on=['origin', 'destination', 'path_id'],
                                  right_on=['origin', 'destination', 'path_id'])
        path_att = path_att.rename(columns={'link_travel_time': 'InVeh_time'})
        # num of transfer
        temp = od_path.loc[:,
               ['origin', 'destination', 'path_id', 'line', 'direction']].drop_duplicates()
        temp = temp.groupby(['origin', 'destination', 'path_id'], sort=False).size().reset_index(drop=True)
        path_att['Num_transfer'] = temp
        path_att['Num_transfer'] = path_att['Num_transfer'] - 1  # num of transfer = num of lines - 1

        # map distance
        map_distance = pd.read_csv('data/Station_Map_Distances.csv')
        distance_data = od_path.merge(map_distance[['link_start_id', 'link_end_id', 'Distance']],
                                      left_on=['link_start', 'link_end'], right_on=['link_start_id', 'link_end_id'],
                                      how='left')
        distance_data.loc[distance_data['link_start'] == distance_data['link_end'], 'Distance'] = 0
        distance_data = distance_data.drop(columns=['link_start_id', 'link_end_id'])
        distance_data_na = distance_data.loc[distance_data['Distance'].isna()]
        # print (len(distance_data_na))
        distance_data = \
            distance_data.groupby(['origin', 'destination', 'path_id'], sort=False).sum().reset_index()[
                ['origin', 'destination', 'path_id', 'Distance']]
        # print(len(distance_data))
        path_att = path_att.merge(distance_data, left_on=['origin', 'destination', 'path_id'],
                                  right_on=['origin', 'destination', 'path_id'])

        # transfer time MTR measure
        od_path['index'] = od_path.index
        transfer_station = od_path.groupby(
            ['origin', 'destination', 'path_id', 'line']).last().reset_index(drop=False)
        transfer_station = transfer_station.loc[transfer_station['link_travel_time'] != 0].reset_index(drop=True)
        transfer_station_after_index = transfer_station['index'] + 1
        transfer_station_after = od_path.loc[transfer_station_after_index, :].reset_index(drop=True)

        transfer_station['Line_after'] = transfer_station_after['line']
        transfer_station['Dir_after'] = transfer_station_after['direction']
        transfer_station['Station_after'] = transfer_station_after['station']
        transfer_station['platform_x'] = transfer_station['station'].apply(str) + '_' + transfer_station[
            'line'].apply(str) + '_' + transfer_station['direction'].apply(str)
        transfer_station['platform_y'] = transfer_station['Station_after'].apply(str) + '_' + transfer_station[
            'Line_after'].apply(str) + '_' + \
                                         transfer_station['Dir_after'].apply(str)

        Transfer_Walking_Time = pd.read_csv('data/Transfer_Walking_Time.csv')
        transfer_station = transfer_station.merge(Transfer_Walking_Time, left_on=['platform_x', 'platform_y'],
                                                  right_on=['platform_x', 'platform_y'], how='left')

        transfer_station['walking_time'] = transfer_station['walking_time'] / 60
        transfer_station = transfer_station.rename(columns={'walking_time': 'Transfer_time'})
        # temp = transfer_station.loc[transfer_station['walking_time'].isna(),['platform_x','platform_y']].drop_duplicates()
        transfer_station = \
            transfer_station.groupby(['origin', 'destination', 'path_id']).sum().reset_index(drop=False)[
                ['origin', 'destination', 'path_id', 'Transfer_time']]
        path_att = path_att.merge(transfer_station[['origin', 'destination', 'path_id', 'Transfer_time']],
                                  left_on=['origin', 'destination', 'path_id'],
                                  right_on=['origin', 'destination', 'path_id'], how='left').fillna(0)
        path_att['InVeh_time'] = path_att['InVeh_time'] - path_att[
            'Transfer_time']  # subtract transfer time, update the inveh time

        # transfer time over distance
        path_att['transfer_over_dist'] = path_att['Transfer_time'] / path_att['Distance']

        # commonality factor
        gama = 5
        temp = od_path.loc[:, ['origin', 'destination', 'path_id', 'station']]
        temp2 = temp.groupby(['origin', 'destination', 'path_id'], sort=False).size().reset_index(
            drop=False).rename(columns={0: 'Num_station'})
        temp = temp.merge(temp, left_on=['origin', 'destination', 'station'],
                          right_on=['origin', 'destination', 'station'])
        # print (temp)
        temp = temp.drop_duplicates()
        temp = temp.groupby(['origin', 'destination', 'path_id_x', 'path_id_y'],
                            sort=False).size().reset_index(drop=False).rename(columns={0: 'Num_overlap_station'})
        # num_station
        # print (temp)
        temp = temp.merge(temp2, left_on=['origin', 'destination', 'path_id_x'],
                          right_on=['origin', 'destination', 'path_id']). \
            rename(columns={'Num_station': 'Num_station_x'}).drop(columns=['path_id'])
        temp = temp.merge(temp2, left_on=['origin', 'destination', 'path_id_y'],
                          right_on=['origin', 'destination', 'path_id']). \
            rename(columns={'Num_station': 'Num_station_y'}).drop(columns=['path_id'])
        temp['Lij_Li_Lj_gama'] = (temp['Num_overlap_station'] / np.sqrt(
            temp['Num_station_x'] * temp['Num_station_y'])) ** gama
        temp2 = temp.groupby(['origin', 'destination', 'path_id_x'], sort=False).sum()[
            ['Lij_Li_Lj_gama']].reset_index(drop=False)
        temp2['CF'] = np.log(temp2['Lij_Li_Lj_gama'])  # final CFi
        temp2 = temp2.rename(columns={'path_id_x': 'path_id'})
        path_att = path_att.merge(temp2[['origin', 'destination', 'path_id', 'CF']],
                                  left_on=['origin', 'destination', 'path_id'],
                                  right_on=['origin', 'destination', 'path_id'], how='left').fillna(0)

        # ----******----Differentiate for different time interval----******----
        # only for time-dependent attributes
        path_att_time = path_att.copy()
        path_att_time['TIME_INTERVAL'] = txn_time_list[0]
        for k in np.arange(1, len(txn_time_list)):
            path_att['TIME_INTERVAL'] = txn_time_list[k]
            path_att_time = pd.concat([path_att_time, path_att])

        # total waiting time
        time_table = self.time_table.loc[:, ['Line_ID', 'Direction_ID', 'From_ID', 'Dep_From_ID']]
        time_table['TIME_INTERVAL'] = time_table[
                                          'Dep_From_ID'] // self.route_choice_interval * self.route_choice_interval
        time_table = time_table.sort_values(by=['From_ID', 'Line_ID', 'Direction_ID', 'Dep_From_ID']).reset_index(
            drop=True)
        time_table2 = time_table.loc[1:, :].reset_index(drop=True)
        time_table2['headway'] = time_table2['Dep_From_ID'] - time_table.loc[0:len(time_table2), 'Dep_From_ID']
        # filter un reasonable values
        time_table2 = time_table2.loc[(time_table2['headway'] > 0) & (time_table2['headway'] < 1000)]
        time_table_new = \
            time_table2.groupby(['From_ID', 'Line_ID', 'Direction_ID', 'TIME_INTERVAL'], sort=False).mean()[
                ['headway']].reset_index(drop=False)
        od_path_boarding = od_path.groupby(['origin', 'destination', 'path_id', 'line', 'direction'],
                                           sort=False).first().reset_index(drop=False) \
            [['origin', 'destination', 'path_id', 'line', 'direction', 'station']]
        od_path_boarding = od_path_boarding.merge(time_table_new,
                                                  left_on=['line', 'direction', 'station'],
                                                  right_on=['Line_ID', 'Direction_ID', 'From_ID'])
        # od_path_boarding = od_path_boarding.sort_values(by=['origin', 'destination', 'path_id','TIME_INTERVAL'])
        od_path_boarding = od_path_boarding.groupby(['origin', 'destination', 'path_id', 'TIME_INTERVAL']).sum()[
            ['headway']].reset_index(drop=False)
        od_path_boarding['waiting_time'] = od_path_boarding['headway'] / 2 / 60  # miniutes
        path_att_time = path_att_time.merge(od_path_boarding,
                                            left_on=['origin', 'destination', 'path_id', 'TIME_INTERVAL'],
                                            right_on \
                                                =['origin', 'destination', 'path_id', 'TIME_INTERVAL']).drop(
            columns=['headway'])


        # ------******---
        """ Desired output for path attribute table"""
        # Rename path_att to: origin, destination, path_id, time_interval, in_vehicle_time, no_of_transfer, waiting_time
        #  transfer_time, commonality_factor
        path_att_time = path_att_time.sort_values(by=['origin', 'destination', 'path_id', 'TIME_INTERVAL'])
        path_att_time = path_att_time.rename(
            columns={'origin': 'origin', 'destination': 'destination', 'path_id': 'path_id',
                     'TIME_INTERVAL': 'time_interval',
                     'InVeh_time': 'in_vehicle_time', 'Num_transfer': 'no_of_transfer',
                     'Transfer_time': 'transfer_time',
                     'CF': 'commonality_factor'})
        path_att_time.to_csv(self.output_file_path + 'path_attributes_for_opt.csv', index=False)

    def create_tb_itinerary(self):
        # outputs: ['origin', 'destination', 'path', 'itinerary', 'boarding_station',
        #                                                   'alighting_station', 'line', 'direction']

        od_path = self.network_operation.loc[:,
                  ['origin', 'destination', 'path_id', 'line', 'direction', 'station', 'link_start',
                   'link_end', 'link_travel_time']]
        od_path = od_path.sort_values(by=['origin', 'destination', 'path_id', 'link_travel_time'],
                                      ascending=[True, True, True, False])
        temp = od_path.loc[:, ['origin', 'destination', 'path_id', 'line',
                               'direction']].drop_duplicates().reset_index(drop=True)
        temp['itinerary'] = temp.groupby(['origin', 'destination', 'path_id']).cumcount()
        temp['itinerary'] += 1
        boarding_station = od_path.groupby(['origin', 'destination', 'path_id', 'line', 'direction'],
                                           sort=False).first()[['station']].reset_index(drop=True)
        alighting_station = \
        od_path.groupby(['origin', 'destination', 'path_id', 'line', 'direction'],
                        sort=False).last()[['station']].reset_index(drop=True)
        temp['boarding_station'] = boarding_station['station']
        temp['alighting_station'] = alighting_station['station']
        self.tb_itinerary = temp.rename(
            columns={'origin': 'origin', 'destination': 'destination', 'path_id': 'path', 'line': 'line',
                     'direction': 'direction'})

        # save results into csv file
        int_list = ['origin', 'destination', 'path', 'itinerary', 'boarding_station', 'alighting_station', 'line',
                    'direction']
        self.tb_itinerary[int_list] = self.tb_itinerary[int_list].astype(int)

        self.tb_itinerary.to_csv(self.output_file_path + 'tb_itinerary.csv', index=False)

        return self.tb_itinerary

    def create_tb_path_disrupt(self):
        disrupt_id = 0
        disrupt_path = pd.DataFrame()
        for link in link_disrupt:
            disrupt_id += 1
            link_start = int(link[0].split('-')[0])
            link_end = int(link[0].split('-')[1])
            disrupt_time = [pd.to_timedelta(link[1].split('-')[0]).total_seconds(),
                            pd.to_timedelta(link[1].split('-')[1]).total_seconds()]

            od_path = self.network_operation.loc[:,
                      ['origin', 'destination', 'path_id', 'line', 'direction', 'station',
                       'link_start', 'link_end', 'link_travel_time']]
            od_path = od_path.sort_values(by=['origin', 'destination', 'path_id', 'link_travel_time'],
                                          ascending=[True, True, True, False])
            disrupt_path_temp = od_path.loc[(od_path['link_start'] == link_start) &
                                            (od_path['link_end'] == link_end),
                                            ['origin', 'destination', 'path_id']]
            disrupt_path_temp['disrupt'] = disrupt_id
            disrupt_path_temp['d_time_start'] = disrupt_time[0]
            disrupt_path_temp['d_time_end'] = disrupt_time[1]
            disrupt_path = pd.concat([disrupt_path, disrupt_path_temp])
        disrupt_path = disrupt_path.rename(
            columns={'origin': 'pax_origin', 'destination': 'pax_destination', 'path_id': 'pax_path'})
        disrupt_path = disrupt_path.drop_duplicates()
        disrupt_path.to_csv(self.output_file_path + 'tb_path_disrupt.csv', index=False)

    def create_tb_carrier_network(self):
        # create carrier and network tables from timetable file
        # group timetable information by origin, destination, path, no sort

        time_table_temp = self.time_table.copy()

        self.tb_carrier = self.time_table.loc[(self.time_table['Arr_From_ID'] >= self.carrier_time_period[0]) &
                                              (self.time_table['Dep_To_ID'] <= self.carrier_time_period[1])]

        self.tb_carrier = self.tb_carrier.reset_index(drop=True)
        self.tb_carrier['carrier_car_no'] = self.tb_carrier['Car_Num'].astype(int)
        self.tb_carrier['carrier_line'] = self.tb_carrier['Line_ID'].astype(int)
        self.tb_carrier['carrier_direction'] = self.tb_carrier['Direction_ID'].astype(int)
        self.tb_carrier['carrier_trip'] = self.tb_carrier['Trip_ID']
        self.tb_carrier['carrier_id'] = self.tb_carrier['carrier_line'].apply(str) + '_' + self.tb_carrier[
            'carrier_direction'].apply(str) + '_' + self.tb_carrier['carrier_trip'].apply(str)

        index_columns = ['carrier_line', 'carrier_direction', 'carrier_trip', 'carrier_id', 'carrier_car_no']

        carrier_first_station = self.tb_carrier.groupby(index_columns).first()[['From_ID', 'Arr_From_ID']].reset_index() \
            .rename(columns={'From_ID': 'carrier_first_station', 'Arr_From_ID': 'carrier_arrive_first_station'})
        carrier_last_station = self.tb_carrier.groupby(index_columns).last()[['To_ID']].reset_index().rename(
            columns={'To_ID': 'carrier_last_station'})

        # generate station list
        time_table_temp = time_table_temp.merge(carrier_first_station, left_on=['Line_ID', 'Direction_ID', 'Trip_ID'],
                                                right_on=['carrier_line', 'carrier_direction',
                                                          'carrier_trip'])  # filter out useless data
        station_list = time_table_temp.groupby(['Line_ID', 'Direction_ID', 'Trip_ID']).apply(
            lambda x: '_'.join(list(x['From_ID'].astype(int).astype('str')))).reset_index(). \
            rename(columns={0: 'carrier_serve_stations'})
        time_table_last = time_table_temp.groupby(['Line_ID', 'Direction_ID', 'Trip_ID']).last()[
            ['To_ID']].reset_index().rename(columns={'To_ID': 'last_station'})
        station_list = station_list.merge(time_table_last, left_on=['Line_ID', 'Direction_ID', 'Trip_ID'],
                                          right_on=['Line_ID', 'Direction_ID', 'Trip_ID'])
        station_list['carrier_serve_stations'] = station_list['carrier_serve_stations'] + '_' + station_list[
            'last_station'].astype(int).astype(str)

        self.tb_carrier = carrier_first_station.merge(carrier_last_station, left_on=index_columns,
                                                      right_on=index_columns)
        self.tb_carrier = self.tb_carrier.merge(station_list,
                                                left_on=['carrier_line', 'carrier_direction', 'carrier_trip'],
                                                right_on=['Line_ID', 'Direction_ID', 'Trip_ID'])

        self.tb_network = time_table_temp.loc[:, ['Line_ID', 'Direction_ID', 'From_ID', 'To_ID']].drop_duplicates()
        self.tb_network = self.tb_network.rename(
            columns={'Line_ID': 'carrier_line', 'Direction_ID': 'carrier_direction',
                     'From_ID': 'link_start', 'To_ID': 'link_end'})
        int_carrier_list = ['carrier_car_no', 'carrier_line', 'carrier_direction', 'carrier_arrive_first_station',
                            'carrier_first_station', 'carrier_last_station']
        int_network_list = ['carrier_line', 'carrier_direction', 'link_start', 'link_end']
        self.tb_carrier[int_carrier_list] = self.tb_carrier[int_carrier_list].astype(int)
        self.tb_network[int_network_list] = self.tb_network[int_network_list].astype(int)
        useful_columns = ['carrier_id', 'carrier_car_no', 'carrier_line', 'carrier_direction',
                          'carrier_trip', 'carrier_arrive_first_station',
                          'carrier_first_station', 'carrier_last_station',
                          'carrier_serve_stations']

        self.tb_carrier = self.tb_carrier.loc[:, useful_columns]

        self.tb_carrier.to_csv(self.output_file_path + 'tb_carrier.csv', columns=useful_columns, index=False)
        self.tb_network.to_csv(self.output_file_path + 'tb_network.csv', index=False)

    def create_tb_queue(self):
        # create queue pool from timetable file

        self.tb_queue = self.time_table.loc[:, ['From_ID', 'Line_ID', 'Direction_ID']].drop_duplicates()
        self.tb_queue = self.tb_queue.rename(columns={'From_ID': 'queue_station', 'Line_ID': 'queue_line',
                                                      'Direction_ID': 'queue_direction'})
        self.tb_queue['initial_time_sim'] = self.carrier_time_period[0]
        int_list = ['queue_station', 'queue_line', 'queue_direction', 'initial_time_sim']
        self.tb_queue[int_list] = self.tb_queue[int_list].astype(int)
        self.tb_queue['queue_id'] = self.tb_queue['queue_station'].astype('str') + '_' + \
                                    self.tb_queue['queue_line'].astype('str') + '_' + \
                                    self.tb_queue['queue_direction'].astype('str')

        self.tb_queue['queue_capacity'] = self.queue_capacity
        self.tb_queue.to_csv(self.output_file_path + 'tb_queue.csv',
                             columns=['queue_id', 'queue_capacity', 'queue_station',
                                      'queue_line', 'queue_direction', 'initial_time_sim'], index=False)

    def create_event_list(self):
        # create event list from timetable file, an event is either an arrival or departure time ordered by time
        # should create with respect to tb_carrier and simulation time period
        if len(self.tb_carrier.dropna()) < 1:
            self.create_tb_carrier_network()

        self.tb_carrier['carrier_line'] = self.tb_carrier['carrier_line'].apply(int)
        self.tb_carrier['carrier_direction'] = self.tb_carrier['carrier_direction'].apply(int)
        self.tb_carrier['carrier_trip'] = self.tb_carrier['carrier_trip'].apply(str)
        self.time_table['Line_ID'] = self.time_table['Line_ID'].apply(int)
        self.time_table['Direction_ID'] = self.time_table['Direction_ID'].apply(int)
        self.time_table['Trip_ID'] = self.time_table['Trip_ID'].apply(str)
        time_table_sim = \
            self.tb_carrier.merge(self.time_table,
                                  left_on=['carrier_line', 'carrier_direction', 'carrier_trip'],
                                  right_on=['Line_ID', 'Direction_ID', 'Trip_ID'],
                                  how='left')
        time_table_sim = time_table_sim.loc[time_table_sim['Arr_From_ID'] >= self.carrier_time_period[0]]
        time_table_sim = time_table_sim.loc[time_table_sim['Dep_To_ID'] <= self.carrier_time_period[1]]
        # time_table_sim = carrier_sim.merge(self.time_table, on=['Line_ID', 'Direction_ID', 'Trip_ID'], how='left')

        # be careful when copy a data frame or object (copy value but not address)
        df_arr = copy.deepcopy(time_table_sim)
        df_dep = copy.deepcopy(time_table_sim)
        df_arr['event_time'] = df_arr['Arr_From_ID']
        df_arr['event_type'] = 0  # arrive
        df_dep['event_time'] = df_dep['Dep_From_ID']
        df_dep['event_type'] = 1  # departure
        time_table_sim = time_table_sim.sort_values(['Arr_From_ID'])
        df_add_arr_last = time_table_sim.groupby(['carrier_id']).last().reset_index()
        df_add_arr_last['From_ID'] = df_add_arr_last['To_ID']
        df_add_arr_last['event_time'] = df_add_arr_last['Arr_To_ID']
        df_add_arr_last['event_type'] = 0

        # combine the arrive and departure events together and ordered by event time
        df_event = pd.concat([df_arr, df_dep, df_add_arr_last], ignore_index=True).sort_values('event_time') \
            [['event_time', 'From_ID', 'event_type', 'carrier_id', 'carrier_line',
              'carrier_direction']].drop_duplicates().reset_index(drop=True)
        df_event['event_id'] = df_event.index + 1

        # save results into csv file
        df_event = df_event.rename(columns={'From_ID': 'event_station'})
        self.list_event = df_event.loc[:, ['event_id', 'event_time', 'event_station', 'event_type', 'carrier_id']]

        int_list = ['event_id', 'event_time', 'event_station', 'event_type']
        self.list_event[int_list] = self.list_event[int_list].astype(int)

        self.list_event.to_csv(self.output_file_path + 'tb_event.csv', index=False)

    def initialize_time_table(self):
        # re-represent the arrival and departure time point as the time refereed to the reference time
        global TIME_TABLE_VAR
        if TIME_TABLE_VAR[1] > TIME_TABLE_VAR[0]:
            rand_time_from = random.sample(np.arange(TIME_TABLE_VAR[0], TIME_TABLE_VAR[1]), len(self.time_table))
        else:
            rand_time_from = [0] * len(self.time_table)

        self.time_table['Arr_From_ID'] = pd.to_timedelta(
            self.time_table['Arr_From']).dt.total_seconds() - self.reference_time + rand_time_from
        self.time_table['Dep_From_ID'] = pd.to_timedelta(
            self.time_table['Dep_From']).dt.total_seconds() - self.reference_time + rand_time_from
        self.time_table['Arr_To_ID'] = pd.to_timedelta(
            self.time_table['Arr_To']).dt.total_seconds() - self.reference_time + rand_time_from
        self.time_table['Dep_To_ID'] = pd.to_timedelta(
            self.time_table['Dep_To']).dt.total_seconds() - self.reference_time + rand_time_from
        time_columns = ['Arr_From_ID', 'Dep_From_ID', 'Arr_To_ID', 'Dep_To_ID']
        # <4:00:00 is the second day, add 1 day in the index
        for col in time_columns:
            self.time_table.loc[self.time_table[col] < 4 * 3600, col] += 24 * 3600
        # self.time_table = self.time_table[self.time_table['Arr_From_ID'] > 0]


# -----------------------------------------  Main Function ---------------------------------------------------
def main(para, network_file, time_table_file, demand_file_path, output_file_path):

    global TIME_TABLE_VAR
    # Variable settings

    sim_time_period = para[1].split('-')

    path_external_data = ''
    TIME_TABLE_VAR = [0, 0]

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # generate inputs files for passenger assignment
    assignment_files = FilePreparation(sim_time_period, path_external_data, demand_file_path, network_file, time_table_file,
                                       output_file_path)

    assignment_files.generate_input_files()

    return output_file_path


def prepare_input(para, network_file, time_table_file, demand_file_path, output_file_path):
    out_put_path = main(para, network_file, time_table_file, demand_file_path, output_file_path)
    print('Assignment data preparation finished!\n')
    return out_put_path


if __name__ == "__main__":
    demand_file_path = 'data/demand.csv'
    Test_name = 'HK-18-19'
    time_period = '18:00:00-19:00:00'
    para = [Test_name, time_period]
    sim_time_period = para[1].split('-')
    time_start = str(int(pd.to_timedelta(sim_time_period[0]).total_seconds()))
    time_end = str(int(pd.to_timedelta(sim_time_period[1]).total_seconds()))
    TEST_NAME = para[0]
    output_file_path = 'Case_' + TEST_NAME + '_' + time_start + '-' + \
                       time_end + '/'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    mtr_raw_assignment_file_path = 'data/mtr_network_operation_assignment.csv'
    network_file = post_process(mtr_raw_assignment_file_path)
    time_table_file = pd.read_csv('data/Timetable_MTR.csv')
    out_put_path = prepare_input(para, network_file, time_table_file, demand_file_path, output_file_path)

