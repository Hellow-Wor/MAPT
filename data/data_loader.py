import os
import pickle
import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import Dataset


class Dataset_self(Dataset):
    def __init__(self, root_path, battery_list, flag='train', size=None, features=800, session=0): # data_type,
        """
            data_path: pkl file of dataset  eg.['NCM523-1.pkl', 'NCM523-2.pkl' ....]
        """
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.root_path = root_path
        self.battery_list = battery_list
        self.flag = flag
        self.features = features
        self.session = session
        self.__read_data__()

    def save_to_pkl(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    def load_from_pkl(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    def __get_data__(self, path):
        data_raw = self.load_from_pkl(path)
        capacity = data_raw['capacity'].reshape(-1, 1)
        rul = data_raw['rul'].reshape(-1, 1)
        voltage = data_raw['voltage']
        diff_voltage = data_raw['diff-voltage']
        data = np.concatenate([voltage, diff_voltage, capacity, rul], axis=1)
        return data

    def convert_to_windows(self, raw_data):
        data_x, data_y = [], []
        for i, data in enumerate(raw_data):
            if i>len(raw_data)-self.seq_len-self.pred_len:
                break
            x = raw_data[i:i+self.seq_len, :self.features]
            y = raw_data[i+self.seq_len-self.label_len:i+self.seq_len+self.pred_len, self.features:]
            data_x.append(x)
            data_y.append(y)
        data_x = np.stack(data_x, axis=0)
        data_y = np.stack(data_y, axis=0)
        return data_x, data_y
    def __read_data__(self):
        raw_data_all = []
        for id in self.battery_list:
            path = os.path.join(self.root_path, f'{id}.pkl')
            raw_data = self.__get_data__(path)
            raw_data_all.append(raw_data)
        raw_data_all = np.concatenate(raw_data_all, axis=0)
        self.data_x, self.data_y = self.convert_to_windows(raw_data_all)
    def __getitem__(self, index):
        seq_x, seq_y = np.squeeze(self.data_x[index, :, :]), np.squeeze(self.data_y[index, :, :])
        return seq_x, seq_y, index
    def __len__(self):
        return len(self.data_x)