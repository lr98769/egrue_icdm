import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision.transforms as transforms


class ClassificationDataset(Dataset):
    def __init__(self,
                 file_name,  # Full path of dataset file. string
                 input_dims,  # Input dimensions. list of ints
                 output_dim,  # Output dimension. int
                 transform_min=0., transform_max=1.,  # Min and max scaler for input data. floats
                 seed=None, # Seed to shuffle dataset. int or None (if no shuffling)
                 timing=False # To toggle if we include only testing samples
        ):  
        self.seed = seed
        if self.seed is not None:
            import random
            random.seed(seed)
            torch.manual_seed(self.seed)
            
        if file_name.endswith(".parquet"):
            df = pd.read_parquet(file_name).reset_index()
        else:
            df = pd.read_csv(file_name)

        # Dataset shapes
        self.n_data = int(df.iloc[df.shape[0] - 3][1])
        self.input_dims, self.output_dim = input_dims, output_dim
        input_dim = int(np.prod(self.input_dims))

        # Dataset preprocessing
        self.transform_min = transform_min
        self.transform_max = transform_max
        if len(self.input_dims) == 3:
            if self.input_dims[2] == 3:
                self.transform = transforms.Compose([transforms.ToPILImage(),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.RandomCrop(32, 4),
                                                     transforms.RandomRotation(degrees=15)])

        # Features and label data
        X, Y = df.values[:self.n_data, 1:input_dim + 1], df.values[:self.n_data, input_dim + 1: input_dim + 2]
        # Commented out to have no shuffling of the main dataframe before splitting
        # if self.seed is None:
        self.X, self.Y = X, Y
        # else:
        #     p = np.random.permutation(self.n_data)
        #     self.X, self.Y = X[p], Y[p]
        self.timing = timing

    def __len__(self):
        return self.n_data

    def __getitem__(self, item):
        if len(self.input_dims) == 1:  # Vector data
            x = (torch.from_numpy(self.X[item]).type(torch.DoubleTensor) - self.transform_min) / self.transform_max
        elif len(self.input_dims) == 3 and self.input_dims[2] == 1:  # Image data with one channel
            x = (torch.from_numpy(self.X[item]).type(torch.DoubleTensor) - self.transform_min) / self.transform_max
            x = x.view(self.input_dims[2], self.input_dims[0], self.input_dims[1])
        elif len(self.input_dims) == 3 and self.input_dims[2] == 3:  # Image data with three channel
            input_reshaped = np.uint8(np.transpose(self.X[item].reshape(self.input_dims[2], self.input_dims[0], self.input_dims[1]), (1, 2, 0)))
            x = torch.from_numpy(np.array(self.transform(input_reshaped), np.int32, copy=False)).type(torch.DoubleTensor)
            x = (x - self.transform_min) / self.transform_max
            x = x.permute((2, 0, 1)).contiguous()
        else:
            raise NotImplementedError

        y = torch.from_numpy(np.array(self.Y[item])).type(torch.LongTensor)
        return x, y

    def split(self, batch_size, split=[.6, .8], num_workers=0):
        indices = list(range(self.n_data))

        # Compute class counts in training set
        class_index, class_count = np.unique(self.Y[:int(split[0] * self.n_data)], return_counts=True)
        if self.output_dim is not None:
            N = np.zeros(self.output_dim)
            N[class_index.astype(int)] = class_count
            N = torch.tensor(N)
        else:
            N = None

        # Commented out so that we can use the split ratios to ensure the same ratio
        # Split (and shuffle) the data in train/val/test sets
        # if self.seed is None:
        #     n_test = 10000
        #     i_val = int(.8 * (self.n_data - n_test))
        #     train_indices, val_indices, test_indices = indices[:i_val], \
        #                                                indices[i_val:-n_test], \
        #                                                indices[-n_test:]
        # else:
        train_indices, val_indices, test_indices = indices[:round(split[0] * self.n_data)], \
                                                    indices[round(split[0] * self.n_data):round(split[1] * self.n_data)], \
                                                    indices[round(split[1] * self.n_data):]
        print(len(train_indices), len(val_indices), len(indices))

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        if self.timing:
            test_sampler = SequentialSampler(test_indices)
        else:
            test_sampler = SequentialSampler(indices) # Changed to preserve order and evaluate all samples
        train_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(self, batch_size=2 * batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(self, batch_size=2 * batch_size, sampler=test_sampler, 
                                                  num_workers=num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader, N
