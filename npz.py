# links:
# https://github.com/geoelements/gns
# https://dataverse-prod.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/HUBMDM

import numpy as np

data = np.load('dataset_npz/train.npz', mmap_mode='r', allow_pickle=True)

#print(data.files)
print(data["simulation_trajectory_0"][0].shape)
#print(data["simulation_trajectory_0"][0].shape) -> (1000, 678, 2)
#print(data["simulation_trajectory_0"][0][0].shape) -> (678, 2)
for i in range(5):
    print(data["simulation_trajectory_0"][0][i,0:4])
print(data["simulation_trajectory_0"][1])

for k in data.files:
    print(k)
    #print(data[k])
    #print(k.keys())

#print(data)