# Generate mtr_network_operation information by combining the mtr provided network and operation csv spreadsheet
#  clean the trip leg with walking and blank ones
#  Basically the lines should be in the 9 lines
# Author: Zhenliang Ma, Northeastern University Date: 12/03/2016
import pandas as pd
import csv
import os
import time
from _DefaultValues import *
from itertools import chain

pd.options.mode.chained_assignment = None  # default='warn'


def fast_concate(frames):
    def fast_flatten(input_list):
        return list(chain.from_iterable(input_list))

    COLUMN_NAMES = frames[0].columns
    df_dict = dict.fromkeys(COLUMN_NAMES, [])
    for col in COLUMN_NAMES:
        # Use a generator to save memory
        extracted = (frame[col] for frame in frames)

        # Flatten and save to df_dict
        df_dict[col] = fast_flatten(extracted)
    df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]
    return df


def add_new_transfer(assignment_file):
    # select the lines in the timetable
    lines_list = LINE_CODE_LIST
    mtr_file = pd.read_csv(assignment_file)
    # mtr_file = mtr_file[mtr_file.line.isin(lines_list)]
    mtr_file = mtr_file.sort_values(by=['origin', 'destination', 'path_id', 'link_travel_time'],
                                    ascending=[True, True, True, False])
    # add path choice at ADM
    # passengers transfer from ISL line, Down direction to TWL line, Up direction
    # would have path choice to transfer at ADM or Central
    # [line	direction	station]
    X_line = [13, 2, 2]
    Y_line = [11, 1, 2]
    new_transfer_station = 1
    fraction_Y = int(INCLUDE_CEN.split('CEN')[1]) / 100
    # print (fraction_Y) # transfer in CEN
    fraction_X = 1 - fraction_Y  # transfer in ADM
    mtr_file['index'] = mtr_file.index
    all_path_X = mtr_file.loc[(mtr_file.line == X_line[0]) &
                              (mtr_file.direction == X_line[1]) &
                              (mtr_file.station == X_line[2])]
    all_path_Y = mtr_file.loc[(mtr_file.line == Y_line[0]) &
                              (mtr_file.direction == Y_line[1]) &
                              (mtr_file.station == Y_line[2])]
    all_path = all_path_X.merge(all_path_Y, left_on=['origin', 'destination', 'path_id'],
                                right_on=['origin', 'destination', 'path_id'], how='inner')
    all_path['index_test'] = all_path['index_y'] - all_path['index_x']
    all_path = all_path.loc[all_path['index_test'] == 1]  # two stations not adjacent are filtered
    # ----part 1: X_line->new_transfer_station
    all_path_part1 = all_path.loc[:, ['origin', 'destination', 'path_id', 'link_travel_time_x', 'morning_peak_path_share_x', \
                                      'off_peak_path_share_x', 'evening_peak_path_share_x', 'other_time_path_share_x']]
    all_path_part1['line'] = X_line[0]  #
    all_path_part1['direction'] = X_line[1]
    all_path_part1['station'] = X_line[2]
    all_path_part1['link_start'] = X_line[2]  #
    all_path_part1['link_end'] = new_transfer_station
    all_path_part1['link_travel_time'] = all_path_part1['link_travel_time_x']
    all_path_part1 = all_path_part1.drop(columns=['link_travel_time_x'])
    all_path_part1 = all_path_part1.rename(columns={'morning_peak_path_share_x': 'morning_peak_path_share', 'off_peak_path_share_x': 'off_peak_path_share',
                                                    'evening_peak_path_share_x': 'evening_peak_path_share', 'other_time_path_share_x': 'other_time_path_share'})
    all_path_part1.loc[:, ['morning_peak_path_share', 'off_peak_path_share', 'evening_peak_path_share', 'other_time_path_share']] *= fraction_Y
    # ----part 2: new_transfer_station->new_transfer_station
    all_path_part2 = all_path.loc[:, ['origin', 'destination', 'path_id', 'link_travel_time_x', 'morning_peak_path_share_x', \
                                      'off_peak_path_share_x', 'evening_peak_path_share_x', 'other_time_path_share_x']]
    all_path_part2['line'] = X_line[0]  #
    all_path_part2['direction'] = X_line[1]
    all_path_part2['station'] = new_transfer_station
    all_path_part2['link_start'] = new_transfer_station  #
    all_path_part2['link_end'] = new_transfer_station
    all_path_part2['link_travel_time'] = all_path_part2['link_travel_time_x'] - 0.01
    all_path_part2 = all_path_part2.drop(columns=['link_travel_time_x'])
    all_path_part2 = all_path_part2.rename(columns={'morning_peak_path_share_x': 'morning_peak_path_share', 'off_peak_path_share_x': 'off_peak_path_share',
                                                    'evening_peak_path_share_x': 'evening_peak_path_share', 'other_time_path_share_x': 'other_time_path_share'})
    all_path_part2.loc[:, ['morning_peak_path_share', 'off_peak_path_share', 'evening_peak_path_share', 'other_time_path_share']] *= fraction_Y
    # ----part 3: new_transfer_station->Y_line
    all_path_part3 = all_path.loc[:, ['origin', 'destination', 'path_id', 'link_travel_time_y', 'morning_peak_path_share_x', \
                                      'off_peak_path_share_x', 'evening_peak_path_share_x', 'other_time_path_share_x']]
    all_path_part3['line'] = Y_line[0]  #
    all_path_part3['direction'] = Y_line[1]
    all_path_part3['station'] = new_transfer_station
    all_path_part3['link_start'] = new_transfer_station  #
    all_path_part3['link_end'] = Y_line[2]
    all_path_part3['link_travel_time'] = all_path_part3['link_travel_time_y'] + 0.01
    all_path_part3 = all_path_part3.drop(columns=['link_travel_time_y'])
    all_path_part3 = all_path_part3.rename(columns={'morning_peak_path_share_x': 'morning_peak_path_share', 'off_peak_path_share_x': 'off_peak_path_share',
                                                    'evening_peak_path_share_x': 'evening_peak_path_share', 'other_time_path_share_x': 'other_time_path_share'})
    all_path_part3.loc[:, ['morning_peak_path_share', 'off_peak_path_share', 'evening_peak_path_share', 'other_time_path_share']] *= fraction_Y
    # *********************************************
    all_path_ID = all_path.loc[:, ['origin', 'destination', 'path_id']].drop_duplicates()
    mtr_file_index = mtr_file.merge(all_path_ID, left_on=['origin', 'destination', 'path_id'], \
                                    right_on=['origin', 'destination', 'path_id'], how='inner')[['index']]

    new_path = mtr_file.loc[mtr_file_index['index'], :]
    new_path = new_path.loc[~((new_path.line == X_line[0]) &
                              (new_path.direction == X_line[1]) &
                              (new_path.station == X_line[2]))]
    new_path.loc[:, ['morning_peak_path_share', 'off_peak_path_share', 'evening_peak_path_share', 'other_time_path_share']] *= fraction_Y

    new_path = pd.concat([new_path, all_path_part1, all_path_part2, all_path_part3], sort=False)
    new_path = new_path.sort_values(by=['origin', 'destination', 'path_id', 'link_travel_time'],
                                    ascending=[True, True, True, False])
    # new_path.to_csv('New_path'+para['incl_cen_file']+'.csv', index=False,columns=['origin', 'destination', 'path_id',\
    # 'line','direction','station','link_start',\
    # 'link_end','link_travel_time','morning_peak_path_share', 'off_peak_path_share', 'evening_peak_path_share',\
    # 'other_time_path_share'])

    new_path['path_id'] = new_path['path_id'].apply(lambda x: x + '_NEW')

    # add additional delay time
    def add_daley(x):
        link_travel_time = x.loc[x.transfer_or_not.isna()].iloc[0]['link_travel_time']
        x.loc[x['link_travel_time'] >= link_travel_time, 'link_travel_time'] += 6.7  # addtional time
        return x

    new_path = new_path.groupby(['origin', 'destination', 'path_id'])
    new_path_list = []
    for ind, group in new_path:
        group = add_daley(group)
        new_path_list.append(group)
    new_path = fast_concate(new_path_list)
    mtr_file.loc[mtr_file_index['index'], ['morning_peak_path_share', 'off_peak_path_share', 'evening_peak_path_share', 'other_time_path_share']] *= fraction_X
    mtr_file = pd.concat([mtr_file, new_path], sort=False).reset_index(drop=True)
    mtr_file = mtr_file.sort_values(by=['origin', 'destination', 'path_id', 'link_travel_time'],
                                    ascending=[True, True, True, False])
    temp = mtr_file.loc[:, ['origin', 'destination', 'path_id']].drop_duplicates()
    temp['path_id_ID'] = temp.groupby(['origin', 'destination']).cumcount()
    temp['path_id_ID'] += 1
    mtr_file = mtr_file.merge(temp, left_on=['origin', 'destination', 'path_id'], \
                              right_on=['origin', 'destination', 'path_id'], \
                              how='inner')
    mtr_file = mtr_file.drop(columns=['path_id', 'index'])
    mtr_file = mtr_file.rename(columns={'path_id_ID': 'path_id'})
    mtr_file = mtr_file.sort_values(by=['origin', 'destination', 'path_id', 'link_travel_time'],
                                    ascending=[True, True, True, False])
    # mtr_file.to_csv('External_data/MTR_Network_Operation_Assignment_'+para['incl_cen_file']+'.csv', index=False,
    #                 columns=['origin', 'destination', 'path_id',
    #                          'line','direction','station','link_start',
    #                          'link_end','link_travel_time','morning_peak_path_share', 'off_peak_path_share', 'evening_peak_path_share',
    #                          'other_time_path_share'])
    return mtr_file


# def add_dummy_train_to_multiindex_station(mtr_file):
#     boarding = mtr_file.groupby(['origin', 'destination', 'path_id'], sort = False).first().reset_index()
#     nocons = boarding.loc[boarding['origin']!=boarding['link_start']]
#
#     return mtr_file

def post_process(assignment_file):
    # print (user_file_path)
    mtr_file = add_new_transfer(assignment_file)

    return mtr_file


if __name__ == "__main__":
    print('Directly run _prepare_input_files_for_assignment')
#     # para_list = pd.read_excel('parameter.xls')
#     user_file_path = '0_user_configuration_parameter.csv'
#     assignment_file = 'External_data/mtr_network_operation_assignment.csv'
#     mtr_file = post_process(user_file_path, assignment_file)
