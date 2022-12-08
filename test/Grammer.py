#%% import library & other

import numpy as np

#%% 1

# [0, 1, 2, 3, 4]
np_1 = np.arange(5)
# 4
np_2 = np.array([[1,2], [3,4]]).size
# [[1], [3]]
np_3 = np.array([[1,2], [3,4]])[:, :1]
# [2, 4]
np_4 = np.array([[1,2], [3,4]])[:, 1]
# [[2, 4], [6, 12]]
np_5 = np_3 * np_4
# [[2]]
np_6 = np.where(np_1 == 2)

#%% 2

# get()
d = {'a': 1, 'b': 2}
# 1
print(d.get('a', 0))
# 0
print(d.get('c', 0))
# None (no KeyError raised)
print(d.get('c'))
# {'a': 1, 'b': 2}
print(d)

d = {'a': 1, 'b': 2}
# 1
print(d.pop('a', 0))
# 0
print(d.pop('c', 0))
# KeyError raised
print(d.pop('c'))
# {'b': 2}
print(d)


