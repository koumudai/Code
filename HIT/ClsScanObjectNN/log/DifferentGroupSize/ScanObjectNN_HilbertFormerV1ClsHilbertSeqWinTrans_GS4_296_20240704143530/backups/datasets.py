import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_data(root, split, num_points):
    h5_name = f'{root}/h5_files/main_split/{"training" if split=="train" else "test"}_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")
    all_data = f['data'][:].astype('float32')
    all_cls = f['label'][:].astype('int64')
    f.close()
    all_data = all_data[:, :num_points, :]
    return all_data, all_cls


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNN(Dataset):
    def __init__(self, data_path, split='train', num_points=1024):
        self.data, self.label = load_data(data_path, split, num_points)
        self.split = split

    def __getitem__(self, item):
        data = self.data[item]
        cls = self.label[item]
        if self.split == 'train':
            data = translate_pointcloud(data)
            np.random.shuffle(data)
        return data, cls

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(f'./data', split='train', num_points=2048)
    test = ScanObjectNN(f'./data', split='test', num_points=2048)

    save = []
    for i, (data, cls) in enumerate(train):
        if i > 10: break
        save.append({'data': data, 'cls': cls})
    np.save(f'./scanobjectnn_vis', save)