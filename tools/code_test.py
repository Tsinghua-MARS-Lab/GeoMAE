# import torch
# from spconv.pytorch.hash import HashTable
#
# is_cpus = [False]
# max_size = 1000
# k_dtype = torch.int32
# v_dtype = torch.int64
# for is_cpu in is_cpus:
#         if is_cpu:
#                 dev = torch.device("cpu")
#                 table = HashTable(dev, k_dtype, v_dtype)
#         else:
#                 dev = torch.device("cuda:0")
#
#                 table = HashTable(dev, k_dtype, v_dtype, max_size=max_size)
#
#         keys = torch.tensor([[5,6], [3,6], [7,6], [4,6], [6,6], [2,6], [10,6] ,[8,6]], dtype=k_dtype, device=dev)
#         len=keys.shape[0]
#         values = torch.arange(len, dtype=v_dtype, device=dev)
#         keys_query = torch.tensor([[8,6], [10,6], [2,6], [6,6], [4,6], [7,6], [3,6], [5,6],[11,6]], dtype=k_dtype, device=dev)
#
#         table.insert(keys, values)
#
#         vq, is_empty = table.query(keys_query)
#         print(vq,is_empty)
#         ks, vs, cnt = table.items()
#         cnt_item = cnt.item()
#         print(cnt, ks[:cnt_item], vs[:cnt_item])
#
#         table.assign_arange_()
#         ks, vs, cnt = table.items()
#         cnt_item = cnt.item()
#         print(cnt, ks[:cnt_item], vs[:cnt_item])
mystr=""
for i in range(23):
    mystr+=(str(i)+',')
print(mystr)