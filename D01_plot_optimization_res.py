import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


opt_res = pd.read_csv('output/CORS_RBF_results_1_reference.csv')
opt_res = opt_res.iloc[:101]

best_y = np.min(opt_res['target'])
best_beta = opt_res.loc[opt_res['target'] == best_y]
print('best_beta', best_beta)
opt_res['num_iter'] = range(len(opt_res))
# opt_res['num_iter']


save_fig = 1

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
fontsize = 20
width = 0.7

x_label = list(opt_res['num_iter'])
x_label_new = []
x_tick_new = []
for idx, x in enumerate(x_label):
    if idx % 10 == 0:
        # if len(x) == 4:
        #     x += '0'
        x_label_new.append(x)
        x_tick_new.append(idx)

y_res = list(np.round(opt_res['target'] / 1000, 1))
min_res = np.inf
y_res_opt = []
for obj in y_res:
    if obj <= min_res:
        min_res = obj
    y_res_opt.append(min_res)




plt.plot(opt_res['num_iter'], y_res, alpha = 0.2, label='Raw objective')
plt.plot(opt_res['num_iter'], y_res_opt,linewidth = 2, marker = 'o', color = 'g',markersize = 5 , label='Best objective')
# p2 = ax.bar(np.array(demand_agg.index) + 0.5*width, exit_od_agg1['flow'], width=width, color='lightcoral',
#             align='center', label = 'Simulation')
plt.yticks(fontsize=fontsize)
plt.xticks(x_tick_new, x_label_new,
           fontsize=fontsize, rotation=0)
# plt.legend([p1, p2], ["Ground truth", "Simulation"], fontsize=fontsize)
plt.ylabel('Objective function (' + r'$\times$1000)', fontsize=fontsize)
plt.xlabel('Number of iterations', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
if save_fig == 0:
    plt.show()
else:
    plt.savefig('img/objective_function.jpg', dpi=200)

a=1