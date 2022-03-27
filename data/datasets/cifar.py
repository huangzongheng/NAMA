import torch
import torchvision
import os
import pickle
import numpy as np

from torchvision.datasets import CIFAR100

class CIFAR100_CF(CIFAR100):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR100_CF, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    # self.targets.extend(entry['fine_labels'])
                    self.targets.extend(list(zip(entry['coarse_labels'], entry['fine_labels'])))

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

if __name__ == "__main__":
    dataset = CIFAR100_CF('/data/', transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    # print(dataset[0])
    batch = next(iter(dataloader))
    print(batch)