import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("Paired")



def calculate_RMSE(top_N_stations, od_file_top_N):
    RMSE = {}
    for model in od_file_top_N:
        if model != 'True':
            RMSE[model] = np.sqrt(np.mean(np.square(od_file_top_N[model].loc[top_N_stations, 'flow'] - od_file_top_N['True'].loc[top_N_stations, 'flow'])))
    return RMSE

def plot_exit_demand(od_file, N, save_fig, fig_name_tail):
    actual_od = od_file['True'].copy()
    actual_od = actual_od.sort_values(by=['flow'], ascending=False)
    top_N_stations = list(actual_od.iloc[:N,0])
    for model in od_file:
        od_file[model] = od_file[model].loc[od_file[model]['destination'].isin(top_N_stations)]
        od_file[model] = od_file[model].set_index(['destination'])

    RMSE = calculate_RMSE(top_N_stations, od_file)

    for model in od_file:
        od_file[model]['flow'] /= 1000

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111)
    fontsize = 14
    width = 0.2

    p0_0 = ax.bar(np.array(range(len(top_N_stations))) - 2 * width, od_file['True'].loc[top_N_stations, 'flow'], width=width,
                  color=colors[0], align='center')
    p0_1 = ax.bar(np.array(range(len(top_N_stations))) - width, od_file['Estimated'].loc[top_N_stations, 'flow'], width=width,
                  color=colors[1], align='center')
    p0_2 = ax.bar(np.array(range(len(top_N_stations))), od_file['Uniform'].loc[top_N_stations, 'flow'], width=width,
                  color=colors[2],
                  align='center')
    p1 = ax.bar(np.array(range(len(top_N_stations))) + width, od_file['Shortest path'].loc[top_N_stations, 'flow'], width=width, color=colors[3],
                align='center')
    plt.yticks(fontsize=fontsize)
    plt.xticks(np.array(range(len(top_N_stations))), top_N_stations,
               fontsize=fontsize, rotation=0)

    plt.legend([p0_0, p0_1, p0_2, p1], ['True',
                                            'Estimated, RMSE=' + str(int(round(RMSE['Estimated'], 0))),
                                            'Uniform, RMSE=' + str(int(round(RMSE['Uniform'], 0))),
                                            "Shortest path, RMSE=" + str(int(round(RMSE['Shortest path'], 0)))], fontsize=fontsize)
    plt.xlim(-1, N)
    # plt.ylim(5, 40)
    plt.ylabel('Exit flow by destination (' + r'$\times$' + '1000)', fontsize=fontsize)
    plt.xlabel('Station ID', fontsize=fontsize)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig(f'img/od_exit_flow_compare_{fig_name_tail}.jpg', dpi=200)



def get_OD_exit_flow(file_name, study_period):
    od_state = pd.read_csv(file_name)
    od_state = od_state.loc[(od_state['exit_time'] >= study_period[0]) & (od_state['exit_time'] <= study_period[1])]
    od_state_agg = od_state.groupby(['destination'])['flow'].sum().reset_index()
    branch_station = [25, 42, 22, 8, 74, 75]  # delete branch station, not reasonable at this time
    # branch_station = []
    od_state_agg = od_state_agg.loc[~od_state_agg['destination'].isin(branch_station)]
    return od_state_agg


if __name__ == '__main__':
    model_file_path = {'True': 'exit_demand_optimal_beta.csv',
                       'Estimated': 'exit_demand_synthesized_reference.csv',
                       'Shortest path': 'exit_demand_shortest_path.csv',
                       'Uniform': 'exit_demand_uniform.csv'}
    ##################
    study_period = [64800, 66600] #[66600, 68400]
    # study_period = [66600, 68400]
    od_file = {}
    for model,file_path in model_file_path.items():
        od_file[model] = get_OD_exit_flow('Case_HK-18-19_64800-68400/'+file_path, study_period)

    N = 25
    save_fig = 1
    fig_name_tail = f'{study_period[0]}_{study_period[1]}'
    plot_exit_demand(od_file,  N, save_fig, fig_name_tail)
