# %% [markdown]
# ### Wifi features
# 
# This this is the code to generate the wifi features available in [this dataset](https://www.kaggle.com/devinanzelmo/indoor-navigation-and-location-wifi-features). Using these features can get a score below 14. For an example notebook using them see [this notebook](https://www.kaggle.com/devinanzelmo/wifi-features-lightgbm-starter). They only uses waypoints, wifi and timestamp data to generate solution. See this [forum post](https://www.kaggle.com/c/indoor-location-navigation/discussion/215445) for an outline of this solution method, and methods of improvement.
# 
# There are `break`'s inserted into loops which need to be removed to get this to run. Right now data is written to current working directory. This takes 2-4 hours to run depending on hard drive etc. There is a lot of room for improvement speeding up feature generation. 
# 
# **Update:** I added one line that creates a column for the path filename, this allows for a groupkfold crossvalidation. 
# 

# %% [code]
import pandas as pd
import numpy as np
import glob
import os
import gc
import json

# %% [code]
base_path = ''

# %% [code]
# pull out all the buildings actually used in the test set, given current method we don't need the other ones
ssubm = pd.read_csv('data/sample_submission.csv')

# only 24 of the total buildings are used in the test set, 
# this allows us to greatly reduce the intial size of the dataset

ssubm_df = ssubm["site_path_timestamp"].apply(lambda x: pd.Series(x.split("_")))
used_buildings = sorted(ssubm_df[0].value_counts().index.tolist())
# used_buildings = ["5a0546857ecc773753327266"]

# dictionary used to map the floor codes to the values used in the submission file. 
floor_map = {"B2": -2, "B1": -1, "F1": 0, "F2": 1, "F3": 2, "F4": 3, "F5": 4, "F6": 5, "F7": 6, "F8": 7, "F9": 8,
             "1F": 0, "2F": 1, "3F": 2, "4F": 3, "5F": 4, "6F": 5, "7F": 6, "8F": 7, "9F": 8}

# %% [code]
# get only the wifi bssid that occur over 1000 times(this number can be experimented with)
# these will be the only ones used when constructing features
bssid = dict()

for building in used_buildings:
    folders = sorted(glob.glob(os.path.join('data/'+building+'/*')))
    print(building)
    wifi = list()
    for folder in folders:
        floor = floor_map[folder.split('/')[-1]]
        files = glob.glob(os.path.join(folder, "*.txt"))
        for file in files:
            with open(file) as f:
                txt = f.readlines()
                for e, line in enumerate(txt):
                    tmp = line.strip().split()
                    if tmp[1] == "TYPE_WIFI":
                        wifi.append(tmp)
    df = pd.DataFrame(wifi)
    #top_bssid = df[3].value_counts().iloc[:500].index.tolist()
    value_counts = df[3].value_counts()
    top_bssid = value_counts[value_counts > 1000].index.tolist()
    print(len(top_bssid))
    bssid[building] = top_bssid
    del df
    del wifi
    gc.collect()

# %% [code]
with open("bssid_1000.json", "w") as f:
    json.dump(bssid, f)

with open("bssid_1000.json") as f:
    bssid = json.load(f)

# %% [code]
# generate all the training data 
building_dfs = dict()

for building in used_buildings:
    break
    folders = sorted(glob.glob(os.path.join('data/', building + '/*')))
    dfs = list()
    index = sorted(bssid[building])
    print(building)
    for folder in folders:
        floor = floor_map[folder.split('/')[-1]]
        files = glob.glob(os.path.join(folder, "*.txt"))
        print(floor)
        for file in files:
            wifi = list()
            waypoint = list()
            with open(file) as f:
                txt = f.readlines()
            for line in txt:
                line = line.strip().split()
                if line[1] == "TYPE_WAYPOINT":
                    waypoint.append(line)
                if line[1] == "TYPE_WIFI":
                    wifi.append(line)

            df = pd.DataFrame(np.array(wifi))

            # generate a feature, and label for each wifi block
            for gid, g in df.groupby(0):
                dists = list()
                for e, k in enumerate(waypoint):
                    dist = abs(int(gid) - int(k[0]))
                    dists.append(dist)
                nearest_wp_index = np.argmin(dists)

                g = g.drop_duplicates(subset=3)
                tmp = g.iloc[:, 3:5]
                feat = tmp.set_index(3).reindex(index).replace(np.nan, -999).T
                feat["x"] = float(waypoint[nearest_wp_index][2])
                feat["y"] = float(waypoint[nearest_wp_index][3])
                feat["f"] = floor
                feat["path"] = file.split('/')[-1].split('.')[0]  # useful for crossvalidation
                dfs.append(feat)

    building_df = pd.concat(dfs)
    building_dfs[building] = df
    building_df.to_csv(building + "_1000_train.csv")

# %% [code]
# Generate the features for the test set

ssubm_building_g = ssubm_df.groupby(0)
feature_dict = dict()

for gid0, g0 in ssubm_building_g:
    break
    index = sorted(bssid[g0.iloc[0, 0]])
    feats = list()
    print(gid0)
    for gid, g in g0.groupby(1):

        # get all wifi time locations,
        with open(os.path.join(base_path, 'test/' + g.iloc[0, 1] + '.txt')) as f:
            txt = f.readlines()

        wifi = list()

        for line in txt:
            line = line.strip().split()
            if line[1] == "TYPE_WIFI":
                wifi.append(line)

        wifi_df = pd.DataFrame(wifi)
        wifi_points = pd.DataFrame(wifi_df.groupby(0).count().index.tolist())

        for timepoint in g.iloc[:, 2].tolist():
            deltas = (wifi_points.astype(int) - int(timepoint)).abs()
            min_delta_idx = deltas.values.argmin()
            wifi_block_timestamp = wifi_points.iloc[min_delta_idx].values[0]

            wifi_block = wifi_df[wifi_df[0] == wifi_block_timestamp].drop_duplicates(subset=3)
            feat = wifi_block.set_index(3)[4].reindex(index).fillna(-999)

            feat['site_path_timestamp'] = g.iloc[0, 0] + "_" + g.iloc[0, 1] + "_" + timepoint
            feats.append(feat)
    feature_df = pd.concat(feats, axis=1).T
    feature_df.to_csv(gid0 + "_1000_test.csv")
    feature_dict[gid0] = feature_df