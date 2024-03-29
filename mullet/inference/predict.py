import argparse
import os
import time

import torch
import numpy as np
import SimpleITK as sitk
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from mullet.datasets.dataset import MultiPhaseMultiSliceInferenceDataset
from mullet.models.mullet import MULLET
from mullet.scripts.post import max_region, vote
from mullet.utils.utils import mkdir, setup, cleanup


def inference(rank, world_size, args):
    setup(rank, world_size, port=args.port)
    test_file_list = args.test_file_list[rank]
    num_classes = args.num_classes
    model = MULLET(cls=num_classes, num_tokens=24,
                   n_context=args.n_ctx, bn=torch.nn.SyncBatchNorm).cuda(rank)
    model = DDP(model, device_ids=[
                rank], output_device=rank, find_unused_parameters=True)
    checkpoint = torch.load(os.path.join(
        args.checkpoint_path), map_location='cpu')
    model.module.load_state_dict(checkpoint['model'])
    model.eval()
    for name in test_file_list:
        dataset = MultiPhaseMultiSliceInferenceDataset(
            os.path.join(args.i, name), n_ctx=args.n_ctx)
        d = DataLoader(dataset, args.batch_size, num_workers=args.num_workers)
        s, z, w, h = dataset.shape
        pred_tumor_mask = torch.zeros((s, z, num_classes, w, h))
        pred_tumor_num = torch.zeros((s, z, num_classes, w, h))
        time_start = time.time()
        phase_idx = [0, 1, 2, 3]
        for images, images_bag, key_idx, z in d:
            images_bag = images_bag.to(rank)
            with torch.no_grad():
                mask_0, mask_1, mask_2, mask_3 = model(
                    images_bag)  # (b, c, z, h, w)
                mask_0 = torch.softmax(mask_0, 1).permute((0, 2, 1, 3, 4))
                mask_1 = torch.softmax(mask_1, 1).permute((0, 2, 1, 3, 4))
                mask_2 = torch.softmax(mask_2, 1).permute((0, 2, 1, 3, 4))
                mask_3 = torch.softmax(mask_3, 1).permute((0, 2, 1, 3, 4))
                for mask_0_i, mask_1_i, mask_2_i, mask_3_i, z_i in zip(mask_0, mask_1, mask_2, mask_3, z):
                    pred_tumor_mask[phase_idx[0], z_i:z_i +
                                    args.n_ctx] += mask_0_i.cpu()
                    pred_tumor_mask[phase_idx[1], z_i:z_i +
                                    args.n_ctx] += mask_1_i.cpu()
                    pred_tumor_mask[phase_idx[2], z_i:z_i +
                                    args.n_ctx] += mask_2_i.cpu()
                    pred_tumor_mask[phase_idx[3], z_i:z_i +
                                    args.n_ctx] += mask_3_i.cpu()
                    pred_tumor_num[phase_idx, z_i:z_i + args.n_ctx] += 1
        s = dataset.base_num
        e = dataset.base_num + dataset.len + args.n_ctx - 1
        pred_tumor_mask[:, s:e] = pred_tumor_mask[:, s:e] / \
            pred_tumor_num[:, s:e]
        pred_tumor_mask[:, :, 0, :, :] += 1e-15
        pred_tumor_mask = torch.argmax(pred_tumor_mask, 2)
        time_end = time.time()

        pred_tumor_mask = pred_tumor_mask.numpy().astype(np.int16)
        for i in range(len(pred_tumor_mask)):
            idx = (dataset.liver[0] > 0) | (pred_tumor_mask[i] > 0)
            drop = max_region(idx) == 0
            pred_tumor_mask[i][drop] = 0
            pred_tumor_mask[i] = vote(pred_tumor_mask[i])

        mkdir(os.path.join(args.o, name))

        seg_p = pred_tumor_mask[0]
        savedImg = sitk.GetImageFromArray(seg_p)
        savedImg.SetSpacing(dataset.spacing)
        savedImg.SetOrigin(dataset.origin)
        sitk.WriteImage(savedImg, os.path.join(
            args.o, name, "segmentation_1.nii.gz"))

        seg_a = pred_tumor_mask[1]
        savedImg = sitk.GetImageFromArray(seg_a)
        savedImg.SetSpacing(dataset.spacing)
        savedImg.SetOrigin(dataset.origin)
        sitk.WriteImage(savedImg, os.path.join(
            args.o, name, "segmentation_2.nii.gz"))

        seg_v = pred_tumor_mask[2]
        savedImg = sitk.GetImageFromArray(seg_v)
        savedImg.SetSpacing(dataset.spacing)
        savedImg.SetOrigin(dataset.origin)
        sitk.WriteImage(savedImg, os.path.join(
            args.o, name, "segmentation_3.nii.gz"))

        seg_d = pred_tumor_mask[3]
        savedImg = sitk.GetImageFromArray(seg_d)
        savedImg.SetSpacing(dataset.spacing)
        savedImg.SetOrigin(dataset.origin)
        sitk.WriteImage(savedImg, os.path.join(
            args.o, name, "segmentation_4.nii.gz"))

        print(name, str(time_end - time_start) + "s")
        print("Saved the segmentation in ", os.path.join(
            args.o, name))
    cleanup()


def run_inference(test_fn, args, world_size):
    test_file_list = [i for i in os.listdir(args.i) if "example_" in i]
    test_file_list = sorted(test_file_list)
    test_file_list = np.array_split(test_file_list, world_size)
    args.test_file_list = test_file_list
    mp.spawn(test_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)


def predict_entry_point():
    parser = argparse.ArgumentParser('Multi-phase Liver Lesion Segmentation')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct format for your files!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created.')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_ctx', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--port', type=str, default="6666")

    args = parser.parse_args()

    devices = ','.join([str(s) for s in args.devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    n_gpus = torch.cuda.device_count()
    run_inference(inference, args, n_gpus)


if __name__ == '__main__':
    predict_entry_point()
