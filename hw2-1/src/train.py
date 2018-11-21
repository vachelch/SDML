# coding: utf-8
import sys
from collections import Counter

from scipy import sparse
import pandas as pd 
import numpy as np
from lightfm import LightFM, evaluation

# from skopt import forest_minimize

# from user_item_features import get_features
# from utils import train_test_split

# train_dir = 'rating_train.csv'
# pred_path = "pred_0.92.csv"

train_dir = sys.argv[1]
pred_path = sys.argv[2]

def write_csv(ids, preds, pred_path):
    data = list(zip(ids, preds))
    data.sort(key=lambda tup: tup[0])
    
    ids, preds = zip(*data)

    with open(pred_path, 'w') as f:
        f.write("userid,foodid\n")
        for uid, items in zip(ids, preds):
            items_res = " ".join([str(item) for item in items])
            f.write("%d,%s\n"%(uid, items_res))



######## read data, and indexing  ########
data_df = pd.read_csv(train_dir)

usr_ids_raw = data_df['userid']
food_ids_raw = data_df['foodid']

usr_ids_set = set(usr_ids_raw)
food_ids_set = set(food_ids_raw)

usr_raw2new = dict(zip(usr_ids_set, range(len(usr_ids_set))))
food_raw2new = dict(zip(food_ids_set, range(len(food_ids_set))))

usr_new2raw = dict(zip(range(len(usr_ids_set)), usr_ids_set))
food_new2raw = dict(zip(range(len(food_ids_set)), food_ids_set))

usr_ids = [usr_raw2new[uid] for uid in usr_ids_raw]
food_ids = [food_raw2new[fid] for fid in food_ids_raw]

######## All data  ########
pair_cnts = Counter()
for uid, fid in zip(usr_ids, food_ids):
    hash_str = str(uid) + ',' + str(fid)
    pair_cnts[hash_str] += 1

I = []
J = []
V = []

for pair, cnt in pair_cnts.items():
    x, y = pair.split(',')
    x = int(x)
    y = int(y)
    I.append(x)
    J.append(y)
    V.append(cnt)


# print("sparsity: %f"%(len(I) / len(usr_ids_set) / len(food_ids_set)) )

# ######## split data  ########
# I_train, J_train, V_train, I_val, J_val, V_val = train_test_split(I, J, V, 0.15)


####### input data  ########
all_data = sparse.coo_matrix((V,(I,J)),shape=(len(usr_ids_set),len(food_ids_set)))
# train_data = sparse.coo_matrix((V_train,(I_train,J_train)),shape=(len(usr_ids_set),len(food_ids_set)))
# val_data = sparse.coo_matrix((V_val,(I_val,J_val)),shape=(len(usr_ids_set),len(food_ids_set)))

# ####### side features  ######## 
# f_row, f_col, f_data, u_row, u_col, u_data, f_shape, u_shape = get_features()
# f_row = [food_raw2new[i] for i in f_row]
# u_row = [usr_raw2new[i] for i in u_row]
# user_features = sparse.csr_matrix((u_data,(u_row, u_col)),shape= u_shape)
# food_features = sparse.csr_matrix((f_data,(f_row, f_col)),shape= f_shape)


# ####### scikit-optimize, find optimal parameters  ######## 
# # option 1. random pick parameter
# def objective(params):
#     # unpack
#     epochs, learning_rate, no_components,\
#     alpha, max_sampled = params
    
#     user_alpha = alpha
#     item_alpha = alpha
#     model = LightFM(loss='warp',
#                     random_state=2018,
#                     learning_rate=learning_rate,
#                     no_components=no_components,
#                     user_alpha=user_alpha,
#                     item_alpha=item_alpha,
#                     max_sampled=max_sampled)
#     model.fit(train_data, epochs=epochs,
#               num_threads=30, verbose=True)
    
#     patks = evaluation.precision_at_k(model, val_data,
#                                       train_interactions=None,
#                                       k=20, num_threads=20)
#     mapatk = np.mean(patks)
#     # Make negative because we want to _minimize_ objective
#     out = -mapatk
#     # Handle some weird numerical shit going on
#     if np.abs(out + 1) < 0.01 or out < -1.0:
#         return 0.0
#     else:
#         return out


# space = [(30, 250), # epochs
#          (10**-4, 1.0, 'log-uniform'), # learning_rate
#          (20, 200), # no_components
#          (10**-6, 10**-1, 'log-uniform'), # alpha
#          (5, 60), # max_sampled
#         ]

# res_fm = forest_minimize(objective, space, n_calls=300,
#                      random_state=0,
#                      verbose=True, 
#                      n_jobs = 20)


# # # print result
# print('Maximimum p@k found: {:6.5f}'.format(-res_fm.fun))
# print('Optimal parameters:')
# params = ['epochs', 'learning_rate', 'no_components', 'alpha', 'max_sampled']
# for (p, x_) in zip(params, res_fm.x):
#     print('{}: {}'.format(p, x_))




######## train the model  ######## 
model = LightFM(loss='warp', 
                learning_rate = 0.036281404040243825,
                no_components = 29,
                user_alpha = 0.00048625731451155697,
                item_alpha = 0.00048625731451155697,
                max_sampled = 37,
                )
# model.fit(train_data, user_features, food_features, epochs=10, num_threads=20)
model.fit(all_data, epochs= 197, num_threads=10)

# patks = evaluation.precision_at_k(model, val_data,
#                                   train_interactions=None,
#                                   # user_features = user_features,
#                                   # item_features = food_features,
#                                   k=20, num_threads=20)

# mapatk = np.mean(patks)
# print(mapatk)



######## predict  ######## 
preds = []
food_ids_vocab = np.array(list(food_ids_set))
usr_ids_vocab = np.array(list(usr_ids_set))

# get top foods
for i in range(len(usr_ids_set)):
    uid = i

    # scores = model.predict(uid, np.arange(len(food_ids_set)), item_features = food_features, user_features = user_features)
    scores = model.predict(uid, np.arange(len(food_ids_set)))
    top_items = food_ids_vocab[np.argsort(-scores)]

    # filter
    pred = []
    for fid in top_items:
        hash_str = str(uid) + ',' + str(fid)
        if hash_str not in pair_cnts:
            pred.append(food_new2raw[fid])

            if len(pred) > 20:
                break

    preds.append(pred)

# raw uid
ids = [usr_new2raw[i] for i in range(len(usr_ids_set))]

# write to csv
write_csv(ids, preds, pred_path)













