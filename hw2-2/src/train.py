#coding: utf-8
import numpy as np
import pandas as pd 
from collections import Counter
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
import sys

format = "%Y-%m-%d"

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


time_cnt = 2
# fm_score = pickle.load(open('fm_score.pk', 'rb'))

######## read data, and indexing  ########
data_df = pd.read_csv(train_dir)


usr_ids_raw = data_df['userid'].tolist()
food_ids_raw = data_df['foodid'].tolist()
dates = data_df['date'].tolist()

def transformation(X):
    poly = PolynomialFeatures(2)
    return poly.fit_transform(X) 

def get_train_pair(usr_ids_raw, food_ids_raw, dates, k):
    user_food_label = set() # set(["uid,foodid", ])
    user_food_train = {} # {"uid,fid": [cnt, [time1, ]]}}

    last_uid = -1
    date_f = "2015-05-01"
    t2 = datetime.strptime(date_f, format)

    for i in range(len(usr_ids_raw)-1, -1, -1):
        uid = usr_ids_raw[i]
        fid = food_ids_raw[i]
        hash_str = str(uid) + "," + str(fid)

        if uid != last_uid:
            food_num = 0

        if food_num < k:
            if hash_str not in user_food_label:
                user_food_label.add(hash_str)
                food_num += 1

        # train pair
        else:
            date = dates[i]
            t1 = datetime.strptime(date, format)
            t_diff = (t2-t1).days

            if hash_str not in user_food_train:
                user_food_train[hash_str] = [1, [t_diff]]
            else:
                user_food_train[hash_str][0] += 1
                user_food_train[hash_str][1].append(t_diff)

        last_uid = uid

    return user_food_train, user_food_label

def get_true(user_food_train, user_food_label):
    # train data
    true_x = []

    # true label
    for pair in user_food_label:
        if pair in user_food_train:
            cnt, t_diffs = user_food_train[pair]
            
            if len(t_diffs) >= time_cnt:
                true_x.append([cnt, *(t_diffs[:time_cnt])])

    return np.array(true_x)

def get_neg(user_food_train, user_food_label):
    # negtive sampling
    neg_x = []

    for pair, feature in user_food_train.items():
        if pair not in user_food_label:
            cnt, t_diffs = feature
            if len(t_diffs) >= time_cnt:
                neg_x.append([cnt, *(t_diffs[:time_cnt])])

    return np.array(neg_x)


def get_train(usr_ids_raw, food_ids_raw, dates):
    print(1)
    user_food_train, user_food_label = get_train_pair(usr_ids_raw, food_ids_raw, dates, 10)
    print(2)
    true_x, neg_x = get_true(user_food_train, user_food_label), get_neg(user_food_train, user_food_label)
    print(3)

    true_len = len(true_x)


    rand_idxs = np.random.randint(0, len(neg_x), true_len)

    true_y = [1 for i in range(true_len)]
    neg_x = neg_x[rand_idxs]
    neg_y = [0 for i in range(true_len)]

    train_x = np.concatenate((true_x, neg_x), axis = 0)
    train_y = np.concatenate((true_y, neg_y), axis = 0)

    # # time_diff normalize to 0-1
    # mn = min(train_x[:, 1])
    # mx = max(train_x[:, 1])

    # train_x[:, 1] = (train_x[:, 1] - mn) / (mx - mn)

    return train_x, train_y

def get_test(usr_ids_raw, food_ids_raw, dates):
    user_food_train, user_food_label = get_train_pair(usr_ids_raw, food_ids_raw, dates, 0)
    test_x = get_neg(user_food_train, user_food_label)

    mx = max(test_x[:, 1])
    mn = min(test_x[:, 1])

    print(4)
    user_food = dict()# {uid: [fids, features]}

    for pair, feature in user_food_train.items():
        uid, fid = pair.split(',')
        uid = int(uid)
        fid = int(fid)

        cnt, t_diffs = feature
        # mean_tDiff = (sum(t_diffs[:1]) / len(t_diffs[:1]) -  mn ) / (mx - mn)

        if len(t_diffs) >= time_cnt:
            if uid not in user_food:
                user_food[uid] = [[fid], [[cnt, *(t_diffs[:time_cnt])]]]
            else:
                user_food[uid][0].append(fid)
                user_food[uid][1].append([cnt, *(t_diffs[:time_cnt])])

    return user_food


#######    ensemble    #######
# clfs = []
# user_food = get_test(usr_ids_raw, food_ids_raw, dates)
# for i in range(10):
#   print("classification ", i)
#   train_x, train_y = get_train(usr_ids_raw, food_ids_raw, dates)
#   clf = LinearRegression(normalize = True)
#   clf.fit(train_x, train_y)

#   clfs.append(clf)

#######    train    #######
train_x, train_y = get_train(usr_ids_raw, food_ids_raw, dates)
# train_x = transformation(train_x)

user_food = get_test(usr_ids_raw, food_ids_raw, dates)
print(5)

# clf = LinearSVR(max_iter=5000)
# clf = GradientBoostingRegressor(n_estimators=200)
clf = LinearRegression(normalize = True)

clf.fit(train_x, train_y)
# print(6)



#######    predict    #######
user_ids = []
preds = []
for uid, item in user_food.items():
    user_ids.append(uid)

    fids, features = item
    # features = transformation(features)
    fids = np.array(fids)

    pred_poss = 0
    # for clf in clfs:
        # pred_poss += clf.predict(features)
    pred_poss = clf.predict(features)
        
    idxs = np.argsort(pred_poss)[::-1][:20]
    pred = fids[idxs]

    preds.append(pred)

write_csv(user_ids, preds, pred_path)








