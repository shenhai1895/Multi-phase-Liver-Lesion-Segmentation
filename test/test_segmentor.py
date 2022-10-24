import argparse
import os
import sys
import time

import torch
import torchvision
import numpy as np
import SimpleITK as sitk
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from datasets.dataset import TestDualPhaseMultiSliceDataset, transforms_dla_test
from models.MPLLSeg import DualSeg
from utils.utils import mkdir, setup, cleanup


def test(rank, world_size, args):
    setup(rank, world_size, port=args.port)
    test_file_list = args.test_file_list[rank]
    num_classes = args.num_classes
    model = DualSeg(cls=num_classes, num_tokens=16, n_context=args.n_ctx, bn=torch.nn.SyncBatchNorm).cuda(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    checkpoint = torch.load(os.path.join(args.checkpoint_path), map_location='cpu')
    model.module.load_state_dict(checkpoint['model'])
    model.eval()
    for name in test_file_list:
        dataset = TestDualPhaseMultiSliceDataset(os.path.join(args.test_dir, name),
                                                 transforms=transforms_dla_test, n_ctx=args.n_ctx)
        d = DataLoader(dataset, args.batch_size, num_workers=args.num_workers)
        s, z, w, h = dataset.shape
        pred_tumor_mask = torch.zeros((s, z, num_classes, w, h))
        pred_tumor_num = torch.zeros((s, z, num_classes, w, h))
        time_start = time.time()
        for images, images_bag, key_idx, z in d:
            images_bag = images_bag.to(rank)
            with torch.no_grad():
                mask_1, mask_2 = model(images_bag)  # (b, c, z, h, w)
                mask_1 = torch.softmax(mask_1, 1).permute((0, 2, 1, 3, 4))
                mask_2 = torch.softmax(mask_2, 1).permute((0, 2, 1, 3, 4))
                for mask_1_i, mask_2_i, z_i in zip(mask_1, mask_2, z):
                    pred_tumor_mask[0, z_i:z_i + args.n_ctx] += mask_1_i.cpu()
                    pred_tumor_mask[1, z_i:z_i + args.n_ctx] += mask_2_i.cpu()
                    pred_tumor_num[:, z_i:z_i + args.n_ctx] += 1
        s = dataset.base_num
        e = dataset.base_num + dataset.len + args.n_ctx - 1
        pred_tumor_mask[:, s:e] = pred_tumor_mask[:, s:e] / pred_tumor_num[:, s:e]
        pred_tumor_mask[:, :, 0, :, :] += 1e-15
        pred_tumor_mask = torch.argmax(pred_tumor_mask, 2)
        time_end = time.time()
        seg_a = pred_tumor_mask[0].cpu().numpy().astype(np.int16)
        savedImg = sitk.GetImageFromArray(seg_a)
        savedImg.SetSpacing(dataset.spacing)
        savedImg.SetOrigin(dataset.origin)
        sitk.WriteImage(savedImg, os.path.join(args.test_dir, name, name + "_seg_a_pred.nii.gz"))

        seg_v = pred_tumor_mask[1].cpu().numpy().astype(np.int16)
        savedImg = sitk.GetImageFromArray(seg_v)
        savedImg.SetSpacing(dataset.spacing)
        savedImg.SetOrigin(dataset.origin)
        sitk.WriteImage(savedImg, os.path.join(args.test_dir, name, name+"_seg_v_pred.nii.gz"))
        print(name, str(time_end - time_start) + "s")
    cleanup()


def run_test(test_fn, args, world_size):
    """
    predict all test cases on multiple gpu.
    Args:
        test_fn:
        args:
        world_size:

    Returns:

    """
    test_file_list = [i for i in os.listdir(args.test_dir) if "example_" in i]
    test_file_list = sorted(test_file_list)
    test_file_list = np.array_split(test_file_list, world_size)
    args.test_file_list = test_file_list
    mp.spawn(test_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Multi-phase Liver Lesion Segmentation')
    parser.add_argument('--test_dir', type=str, default="/data0/wulei/")
    parser.add_argument('--checkpoint_path', type=str, default="/data0/wulei/train_log/segmentor_1/model_33.pth")
    parser.add_argument('--train_mode', type=str, default="tumor", choices=["tumor", "liver", "all"])
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--n_ctx', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--same_slice', type=bool, default=False)
    parser.add_argument('--devices', type=list, default=[0])
    parser.add_argument('--port', type=str, default="1894")

    args = parser.parse_args()

    devices = ','.join([str(s) for s in args.devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    n_gpus = torch.cuda.device_count()
    run_test(test, args, n_gpus)
