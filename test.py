# elif params['dset'] == 'Caltech101-20':
import scipy.io as sio
from core.Config import load_config

from core import util
import numpy as np
params = load_config('./config/nus.yaml')

from scipy.sparse import csc_matrix
# params = load_config('./config/Caltech101-20.yaml')

# row = np.array([0, 2, 2, 0, 1, 2])
# col = np.array([0, 0, 1, 2, 2, 2])
# data = np.array([1, 2, 3, 4, 5, 6])


# print(csc_matrix((data, (row, col)), shape=(3, 3)))

# print(csc_matrix((data, (row, col)), shape=(3, 3)).toarray())
view=1


mat = sio.loadmat('./data/'+params['dset']+'.mat')

y = mat['Y']
print(y[:1000].shape)
print(np.unique(y[:1000]))
X = mat['X'][0]



print(X[1][:1000].shape) # X[0][0].shape
# X = mat['X'][0]


# x = X[view-1]

# x # is the view ( a scipy.sparse.csc_matrix )

# # print(x[0].shape)
# # print(x[0])
# # print(x.shape)
# # print(x[0].toarray().shape)
# print()

# # for i in x:
#     # print()
# import time
# start_time = time.time()

# total= np.array([x[i].toarray()[0] for i in range(x.shape[0])])
# print(total.shape)
# print("--- %s seconds ---" % (time.time() - start_time))
# # print(x[0].toarray()[0])]

# x = util.normalize(x)
# y = np.squeeze(mat['Y'])

# # split it into two partitions
# data_size = x.shape[0]
# train_index, test_index = util.random_index(data_size, int(data_size*0.5), 1)
# test_set_x = x[test_index]
# test_set_y = y[test_index]
# train_set_x = x[train_index]
# train_set_y = y[train_index]
# # else:
# # raise ValueError('Dataset provided ({}) is invalid!'.format(params['dset']))
