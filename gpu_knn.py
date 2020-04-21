import torch
from knn_cuda import KNN
import numpy as np


# args = parse_args()
mask = np.load('/shared/xudongliu/data/argoverse-tracking/argo_track/' + 'sample'+ '/npy_mask_dense/0000/000000.npy')
positive_list = np.argwhere(mask==1)
positive_label = np.ones((len(positive_list)))
# print(positive_list)

negative_list = np.argwhere(mask==0)
negative_label = np.zeros((len(negative_list)))
# print(negative_list)

query_list = np.concatenate([positive_list, negative_list], axis=0)
query_list = np.expand_dims(query_list, axis=0)

query_label = np.concatenate([positive_label, negative_label], axis=0)

ignore_list = np.argwhere(mask==-1)
ignore_list = np.expand_dims(ignore_list, axis=0)
# print("time: %.2f"%(time.time()-end))

# neigh = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=args.jobs)
# print("time: %.2f"%(time.time()-end))
query_list = torch.from_numpy(query_list).cuda()
# query_list = query_list.view(())
ignore_list = torch.from_numpy(ignore_list).cuda()

knn = KNN(k=5, transpose_mode=True)
dist, indx = knn(query_list, ignore_list)
print(indx.shape)
# neigh.fit(query_list, query_label)
# print("time: %.2f"%(time.time()-end))

# ignore_label = neigh.predict(ignore_list)
# print("time: %.2f"%(time.time()-end))

# new_mask = mask
# new_mask[np.where(mask==-1)] = ignore_label



    