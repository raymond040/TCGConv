import pandas as pd
import numpy as np

u_list, i_list, ts_list, label_list = [], [], [], []
feat_l = []
idx_list = []

with open('reddit.csv') as f:
    s = next(f)
    for idx, line in enumerate(f): #index, line per line reading
        e = line.strip().split(',')
        u = int(e[0])
        i = int(e[1])

        ts = float(e[2])
        label = float(e[3])  # int(e[3])

        feat = np.array([float(x) for x in e[4:]]) #all other features

        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(label)
        idx_list.append(idx)

        feat_l.append(feat)
        
feature_cnt = len(feat_l[0])
df_dct = {'u': u_list,
            'i': i_list,
            'ts': ts_list,
            'label': label_list,
            'idx': idx_list}
for i in range(feature_cnt):
    for j in range(len(feat_l)):
        if 'feat_'+str(i) in df_dct.keys():
            df_dct['feat_'+str(i)].append(feat_l[j][i])
        else:
            df_dct['feat_'+str(i)] = [feat_l[j][i]]
df = pd.DataFrame(df_dct)
df['i'] = df['i']+max(df['u'])
df.to_csv('reddit_processed.csv')
