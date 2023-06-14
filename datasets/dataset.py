import pickle
import random

import json

import os
import torch
import SimpleITK as sitk
import numpy as np
from glob import glob
from torch.utils.data import Dataset

UPPER_BOUND = 155
LOWER_BOUND = -55


def transforms_dla(img):
    lower_bound = LOWER_BOUND
    upper_bound = UPPER_BOUND
    lower_bound += random.randint(-5, 5)
    upper_bound += random.randint(-5, 5)
    img = torch.clamp(img, lower_bound, upper_bound)
    img = 2 * (img - lower_bound) / (upper_bound - lower_bound) - 1
    return img


def transforms_dla_test(img):
    lower_bound = LOWER_BOUND
    upper_bound = UPPER_BOUND
    img = torch.clamp(img, lower_bound, upper_bound)
    img = 2 * (img - lower_bound) / (upper_bound - lower_bound) - 1
    return img


class DualPhaseMultiSliceDataset(Dataset):
    def __init__(self, info_path="/data0/wulei/tr6.0_training/info/patch_tumor_256_256.json",
                 input_dir="/data0/wulei/tr6.0_training/train6.0_slice",
                 depth_info="/data0/wulei/tr6.0_training/info/depth_info.pkl",
                 transforms=transforms_dla,
                 same_slice=False,
                 closing_opt=False,
                 num_classes=5,
                 train_mode="tumor",
                 n_ctx=9,
                 ):
        super().__init__()
        assert train_mode in ["liver", "tumor", "all"]
        self.input_dir = input_dir
        self.same_slice = same_slice
        self.closing_opt = closing_opt
        self.cls = num_classes
        self.train_mode = train_mode
        self.n_ctx = n_ctx
        self.transforms = transforms
        with open(info_path, 'rb') as f:
            all_map = json.load(f)
            all_ = [(k, v) for k, v in all_map.items()]
        pos = []
        neg = []
        boosted_neg = []
        for k, v in all_:
            if v['cnt'] > 0:
                pos.append((k, v))
        self.pos = pos
        self.neg = neg
        self.boosted_neg = boosted_neg
        if not os.path.exists(depth_info):
            self.depth_info = {}
            for i in self.pos:
                name = i[0].split("_")[0]
                if name not in self.depth_info:
                    depth = len(glob(os.path.join(self.input_dir, name + '_' + '*.npz')))
                    self.depth_info[name] = depth
            with open(depth_info, "wb") as f:
                pickle.dump(self.depth_info, f)
        else:
            with open(depth_info, "rb") as f:
                self.depth_info = pickle.load(f)

    def __len__(self):
        return len(self.pos)

    def read_slice(self, case_id, z):
        path = os.path.join(self.input_dir, '%s_%d.npz' % (case_id, z))
        data = np.load(path, allow_pickle=True)
        raw = data['ct'].astype(np.int16)
        mask = data['mask'].astype(np.int16)
        if self.train_mode == 'liver':
            mask = mask > 0
        elif self.train_mode == 'tumor':
            mask[mask == 1] = 0
        return torch.from_numpy(raw), torch.from_numpy(mask)

    def get_slice(self, slices, x0, x1, y0, y1, z):
        try:
            if z == 0:
                return torch.stack(
                    [slices[z][0][:, x0: x1, y0: y1], slices[z][0][:, x0: x1, y0: y1],
                     slices[z + 1][0][:, x0: x1, y0: y1]],
                    1)
            elif z == max(slices.keys()):
                return torch.stack(
                    [slices[z - 1][0][:, x0: x1, y0: y1], slices[z][0][:, x0: x1, y0: y1],
                     slices[z][0][:, x0: x1, y0: y1]],
                    1)
            return torch.stack(
                [slices[z - 1][0][:, x0: x1, y0: y1], slices[z][0][:, x0: x1, y0: y1],
                 slices[z + 1][0][:, x0: x1, y0: y1]],
                1)
        except ValueError:
            print(0)

    def augmentation_3d_3s(self, images, mask):
        flip_num = np.random.randint(0, 8)
        if flip_num == 1:
            images = torch.flip(images, [3])
            mask = torch.flip(mask, [2])
        elif flip_num == 2:
            images = torch.flip(images, [4])
            mask = torch.flip(mask, [3])
        elif flip_num == 3:
            images = torch.rot90(images, k=1, dims=(4, 3))
            mask = torch.rot90(mask, k=1, dims=(3, 2))
        elif flip_num == 4:
            images = torch.rot90(images, k=3, dims=(4, 3))
            mask = torch.rot90(mask, k=3, dims=(3, 2))
        elif flip_num == 5:
            images = torch.flip(images, [4])
            mask = torch.flip(mask, [3])
            images = torch.rot90(images, k=1, dims=(4, 3))
            mask = torch.rot90(mask, k=1, dims=(3, 2))
        elif flip_num == 6:
            images = torch.flip(images, [4])
            mask = torch.flip(mask, [3])
            images = torch.rot90(images, k=3, dims=(4, 3))
            mask = torch.rot90(mask, k=3, dims=(3, 2))
        elif flip_num == 7:
            images = torch.flip(images, [3, 4])
            mask = torch.flip(mask, [2, 3])
        return images, mask

    def __getitem__(self, item):
        k, v = self.pos[item]
        temp = k.split('_')
        case_id = temp[0]
        x0, x1, y0, y1, z = v['pos']
        height = x1 - x0
        width = y1 - y0
        x0 += random.randint(-10, 10)
        x1 = x0 + height
        if x0 < 0:
            x0 = 0
            x1 = x0 + width
        elif x1 > 512:
            x1 = 512
            x0 = x1 - width
        y0 += random.randint(-10, 10)
        y1 = y0 + width
        if y0 < 0:
            y0 = 0
            y1 = y0 + height
        elif y1 > 512:
            y1 = 512
            y0 = y1 - height
        depth = self.depth_info[case_id]
        half = self.n_ctx // 2
        half = random.randint(-2, 2) + half
        start = max(z - half, 0)
        end = min(depth, z + self.n_ctx - half)
        if start == 0:
            end = start + self.n_ctx
        elif end == depth:
            start = end - self.n_ctx
        key_idx = z - start
        slices = {}
        for t in range(max(start - 1, 0), min(depth, end + 1)):
            try:
                slices[t] = self.read_slice(case_id, t)
            except Exception as e:
                print(e)
                print(case_id)
        img = []
        for t in range(start, end):
            img.append(self.get_slice(slices, x0, x1, y0, y1, t))
        img = torch.stack(img, 1)
        mask = []
        for t in range(start, end):
            mask.append(slices[t][1])
        mask = torch.stack(mask, 1)
        mask = mask[:, :, x0: x1, y0: y1]
        if self.train_mode == "all":
            if self.cls == 5:
                mask[(mask == 2) | (mask == 3) | (mask == 6)] = 2
                mask[mask == 4] = 3
                mask[mask == 5] = 4
            elif self.cls == 3:
                mask[mask > 1] = 2
        elif self.train_mode == "tumor":
            if self.cls == 5:
                mask[(mask == 2) | (mask == 3) | (mask == 6)] = 1
                mask[mask == 4] = 2
                mask[mask == 5] = 3
            elif self.cls == 3:
                mask[mask > 1] = 1

        if self.transforms is not None:
            img = self.transforms(img)
        img, mask = self.augmentation_3d_3s(img, mask)
        img = torch.as_tensor(img, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.long)
        return img[[1, 2]], (mask[1], mask[2])


class TestDualPhaseMultiSliceDataset(Dataset):
    def __init__(self, img_path="/data0/wulei/example_3",
                 reverse_slice=False,
                 transforms=transforms_dla_test,
                 n_ctx=3):
        super().__init__()
        img_a = sitk.ReadImage(glob(os.path.join(img_path, "*_img_a.nii.gz")))
        img_a_array = sitk.GetArrayFromImage(img_a)
        img_v = sitk.ReadImage(glob(os.path.join(img_path, "*_img_v.nii.gz")))
        img_v_array = sitk.GetArrayFromImage(img_v)
        liver = sitk.ReadImage(glob(os.path.join(img_path, "*_liver_v.nii.gz")))
        liver_array = sitk.GetArrayFromImage(liver)
        ct = np.concatenate([img_a_array, img_v_array], 0)
        self.origin = img_a.GetOrigin()
        self.spacing = img_a.GetSpacing()
        self.shape = ct.shape
        self.n_ctx = n_ctx
        self.transforms = transforms
        _, s_z, s_x, s_y = np.where(liver_array > 0)
        min_z = max(min(s_z), 0)
        max_z = min(max(s_z), ct.shape[1] - 1)
        self.len = max_z - min_z - n_ctx + 2
        self.ct = torch.from_numpy(ct)
        self.base_num = min(s_z)
        self.liver = liver_array
        self.reverse_slice = reverse_slice

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
        if self.reverse_slice:
            images = torch.flip(images, [0])
        if self.transforms is not None:
            images = self.transforms(images)
        images = torch.as_tensor(images, dtype=torch.float32)
        z = torch.as_tensor(z, dtype=torch.int64)

        return images[:, key_idx], images, key_idx, z


if __name__ == '__main__':
    m = TestDualPhaseMultiSliceDataset()
    for i in m:
        pass
