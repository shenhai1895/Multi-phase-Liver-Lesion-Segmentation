import os
import torch
import SimpleITK as sitk
import numpy as np
from glob import glob
from torch.utils.data import Dataset

UPPER_BOUND = 190
LOWER_BOUND = -65


def MinMaxNormalization(img):
    lower_bound = LOWER_BOUND
    upper_bound = UPPER_BOUND
    img = torch.clamp(img, lower_bound, upper_bound)
    img = 2 * (img - lower_bound) / (upper_bound - lower_bound) - 1
    return img


class MultiPhaseMultiSliceInferenceDataset(Dataset):
    def __init__(self, img_path="/data0/raw_data/sss/nii/example_0000/",
                 normalize=MinMaxNormalization,
                 n_ctx=3):
        super().__init__()
        ct = sitk.ReadImage(sorted(glob(os.path.join(img_path, "CT_*.nii.gz"))))
        ct_array = sitk.GetArrayFromImage(ct)
        
        liver = sitk.ReadImage(sorted(glob(os.path.join(img_path, "liver_*.nii.gz"))))
        liver_array = sitk.GetArrayFromImage(liver)

        self.origin = ct.GetOrigin()[0:3]
        self.spacing = ct.GetSpacing()[0:3]
        self.shape = ct_array.shape
        self.n_ctx = n_ctx
        self.normalize = normalize
        _, s_z, s_x, s_y = np.where(liver_array > 0)
        min_z = max(min(s_z), 0)
        max_z = min(max(s_z), liver_array.shape[1] - 1)
        self.len = max_z - min_z - n_ctx + 2
        self.ct = torch.from_numpy(ct_array)
        self.base_num = min(s_z)
        self.liver = liver_array

    def __len__(self):
        return self.len

    def get_slice(self, slices, z):
        if z == 0:
            return torch.stack([slices[z], slices[z], slices[z + 1]], 1)
        elif z == max(slices.keys()):
            return torch.stack([slices[z - 1], slices[z], slices[z]], 1)
        return torch.stack([slices[z - 1], slices[z], slices[z + 1]], 1)

    def get_data(self, z):
        depth = self.ct.shape[1]
        start = z
        end = z + self.n_ctx
        if start == 0:
            end = start + self.n_ctx
        elif end == depth:
            start = end - self.n_ctx
        key_idx = z - start
        slices = {}
        for t in range(max(start - 1, 0), min(depth, end + 1)):
            slices[t] = self.ct[:, t]
        images = []
        for t in range(start, end):
            images.append(self.get_slice(slices, t))
        images = torch.stack(images, 1)
        return images, key_idx

    def __getitem__(self, item):
        z = item + self.base_num
        images, key_idx = self.get_data(z)
        if self.normalize is not None:
            images = self.normalize(images)
        images = torch.as_tensor(images, dtype=torch.float32)
        z = torch.as_tensor(z, dtype=torch.int64)

        return images[:, key_idx], images, key_idx, z
