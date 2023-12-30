import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

time_interval = (15*60)

demand = pd.read_csv('Case_HK-18-19_64800-68400/exit_demand_synthesized_reference.csv')

print('total', len(demand))

demand['time_id'] = demand['entry_time'] // time_interval * time_interval
demand['hour'] = demand['time_id'] // 3600
demand['minute'] = demand['time_id'] % 3600 // 60

demand['time_id_str'] = demand['hour'].astype('str') + ':' + demand['minute'].astype('str')

demand_agg = demand.groupby('time_id_str')['flow'].sum().reset_index()

save_fig = 1

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
fontsize = 20
width = 0.7

x_label = list(demand_agg['time_id_str'])
x_label_new = []
x_tick_new = []
for idx, x in enumerate(x_label):
    if idx % 2 == 0:
        if len(x) == 4:
            x += '0'
        x_label_new.append(x)
        x_tick_new.append(idx)

p1 = ax.bar(np.array(demand_agg.index), demand_agg['flow'] / 1000, width=width, color='royalblue',
            align='center')
# p2 = ax.bar(np.array(demand_agg.index) + 0.5*width, exit_od_agg1['flow'], width=width, color='lightcoral',
#             align='center', label = 'Simulation')
plt.yticks(fontsize=fontsize)
plt.xticks(x_tick_new, x_label_new,
           fontsize=fontsize, rotation=0)
# plt.legend([p1, p2], ["Ground truth", "Simulation"], fontsize=fontsize)
plt.ylabel('Tap-in flow (' + r'$\times$1000)', fontsize=fontsize)
plt.xlabel('Time', fontsize=fontsize)
# plt.legend(fontsize=fontsize)
plt.tight_layout()
if save_fig == 0:
    plt.show()
else:
    plt.savefig('img/tap_in_flow.jpg', dpi=200)