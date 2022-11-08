import argparse
import os
import sys
from collections import Counter
from glob import glob
from multiprocessing import Pool

import numpy as np
import torch
import torchvision
from skimage.measure import label, regionprops
from skimage.morphology import opening, disk

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1


def vote(array):
    l, n = label(array > 0, return_num=True)
    for i in range(n):
        idx = l == (i + 1)
        array[idx] = Counter(array[idx]).most_common(1)[0][0]
    return array


def max_region(array):
    l, n = label(array > 0, return_num=True)
    regs = regionprops(l)
    if len(regs) == 0:
        return array
    regs.sort(key=lambda x: x.filled_area, reverse=True)
    _max_region = regs[0]
    coords = _max_region.coords
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    array[:, :, :] = 0
    array[x, y, z] = 1
    return array


def post_tumor(path_pred, args):
    series_index = args.series_index
    out_dir = args.out_dir
    gt_dir = args.path_gt

    base_name = os.path.basename(path_pred)

    seg = np.load(path_pred, allow_pickle=True)
    series_segs = np.array(seg.get('arrays', seg.get('arr_0')))
    meta = seg.get('seg_nums', seg.get('arr_1'))
    header = seg.get("header")
    series_segs = series_segs.reshape((len(series_segs), -1, 512, 512))
    liver = series_segs & 1
    tumor = series_segs >> 1

    gt_seg = np.load(os.path.join(gt_dir, base_name), allow_pickle=True)
    gt_series_segs = np.array(gt_seg.get('arrays', seg.get('arr_0')))
    gt_series_segs = gt_series_segs.reshape((len(gt_series_segs), -1, 512, 512))
    liver = gt_series_segs & 1

    liver[liver > 1] = 1
    liver = liver.transpose((0, 2, 3, 1))
    tumor[tumor == 1] = 0
    tumor = tumor.transpose((0, 2, 3, 1))

    if series_index == -1:
        for i in range(len(gt_series_segs)):
            liver[i] = max_region(liver[i])
            idx = (liver[i] > 0) | (tumor[i] > 0)
            drop = max_region(idx) == 0
            tumor[i][drop] = 0
            tumor[i] = vote(tumor[i])
            for j in range(tumor[i].shape[2]):
                tumor[i][:, :, j] = opening(tumor[i][:, :, j], disk(1))
            # tumor[i] = opening(tumor[i], disk(3))
    else:
        liver[series_index] = max_region(liver[series_index])
        idx = (liver[series_index] > 0) | (tumor[series_index] > 0)
        drop = max_region(idx) == 0
        tumor[series_index][drop] = 0
        tumor[series_index] = vote(tumor[series_index])

    if out_dir is not None:
        liver = liver.transpose((0, 3, 1, 2)).astype(np.uint8)
        tumor = tumor.transpose((0, 3, 1, 2)).astype(np.uint8)

        mask = liver.astype(np.uint8) | (tumor.astype(np.uint8) << 1)
        out = []
        for x in mask:
            out.append(x.reshape(-1))
        np.savez_compressed(os.path.join(out_dir, base_name),
                            arrays=out, seg_nums=meta, version='2')
    print(base_name, "done~")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="post")
    parser.add_argument('--path_gt', type=str, default=r"/data0/wulei/ts_4.0_new/")
    # parser.add_argument('--path_gt', type=str, default=r"/data0/dataset/liver_CT3_Z2_ts4.0/")
    parser.add_argument('--path_root', type=str, default=r"/data0/wulei/train_log/segmentor_1")
    parser.add_argument('--out_dir', type=str, default='/data0/wulei/train_log/segmentor_1')
    parser.add_argument('--series_index', type=int, default=-1)
    parser.add_argument('--start_epoch', type=int, default=25)
    parser.add_argument('--end_epoch', type=int, default=35)
    args = parser.parse_args()

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    for epoch_i in range(args.start_epoch, args.end_epoch):
        args.path_pred = os.path.join(args.path_root, "pred-%.2d" % epoch_i)
        args.out_dir = args.path_pred + "-post"
        # args.path_pred = "/data6/wulei/train_log/nnunet"
        # args.out_dir = args.path_pred + "-post"
        print(args.path_pred)
        print(args.out_dir)
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        inp = [(i, args) for i in glob(os.path.join(args.path_pred, '*.npz')) if "ct" not in i]
        pool = Pool(10)
        pool.starmap(post_tumor, inp)
        pool.close()
        pool.join()
