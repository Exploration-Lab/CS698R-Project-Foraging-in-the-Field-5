import matplotlib.pyplot as plt

import pandas as pd
import numpy as np



loc = ('my_env/envs/berries_loc.xlsx')
data = pd.read_excel(loc)
berx_cor = data['x'].to_numpy().tolist()
bery_cor = data['y'].to_numpy().tolist()
sz = data['size'].to_numpy().tolist()
for i in range(len(bery_cor)):
    berx_cor[i] -= 10000
    bery_cor[i] -= 11000

# x = np.loadtxt('paths/agent_path239')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
# x = np.loadtxt('paths/agent_path'+str(i))
x = np.loadtxt('paths/agent_path_test_a10')
x_cor = x[:,0]
y_cor = x[:,1]
score = x[:,2]
juice = x[:,3]
tim = x[:,4]
juice = juice/max(juice)
y = []
for i in range(len(score)):
    y.append(i)
plt.plot(x_cor, y_cor, 'r')

plt.scatter(berx_cor, bery_cor, sz)
plt.show()
plt.ylabel("Cumulative reward")
plt.xlabel("Steps")
plt.plot(y, score)
plt.show()
plt.xlabel('Time')
plt.ylabel('Health')
plt.plot(tim, juice)
plt.show()

