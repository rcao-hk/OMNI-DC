# -*- coding: utf-8 -*-
from config import args as args_config
import os
# 仍然按 demo 的做法读取 GPU / 通讯端口（即使这里我们只用 DataParallel，也不影响）
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

import utility
from model.ognidc import OGNIDC


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

# ----------------------------
# 固定随机种子（与 demo 风格一致）
# ----------------------------
def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# 读取 / 覆盖 ckpt 参数（沿用 demo）
# ----------------------------
def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), "file not found: {}".format(args.pretrain)
        if args.resume:
            checkpoint = torch.load(args.pretrain)
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            new_args.start_epoch = checkpoint['epoch'] + 1
    return new_args


import argparse
def merge_args(cfg, cli):
    """用命令行参数覆盖 config.py 的默认参数"""
    for k, v in vars(cli).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg

def get_cli_args():
    parser = argparse.ArgumentParser(description="Multi-dataset inference for OMNI-DC")

    # 允许外部 .sh 传入的关键参数
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--dataset_root', type=str, help='Root directory of dataset')
    parser.add_argument('--split', type=str, help='Split file path')
    parser.add_argument('--method', type=str, help='Output method name')
    parser.add_argument('--camera', type=str, choices=['l515', 'd435', 'tof'], help='Camera type')
    parser.add_argument('--output_root', type=str, help='Root dir to save predictions')
    parser.add_argument('--img_height', type=int, help='Input height')
    parser.add_argument('--img_width', type=int, help='Input width')

    # ⚠️ 关键：允许未知参数通过（比如 config.py 中定义的其它超参）
    args_cli, _ = parser.parse_known_args()
    return args_cli


# ----------------------------
# 工具：从 split 读取 rgb 相对路径
# 每行形如：<rgb_rel_path> [<gt_depth_rel_path> ...]
# 我们只取第一列
# ----------------------------
def read_split_rgb_list(split_file):
    rgb_list = []
    with open(split_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            rgb_rel = line.strip().split()[0]
            rgb_list.append(rgb_rel)
    return rgb_list

# ----------------------------
# 工具：按数据集规则推导 raw/sparse depth 路径与 scale
# 返回 depth_path, depth_scale(->米)
# ----------------------------
def infer_depth_path_and_scale(dataset, dataset_root, rgb_rel_path, camera):
    # 统一：最终都读取 PNG/16U，再乘以 depth_scale（毫米/尺度）* 1e-3 -> 米
    if dataset == 'HAMMER':
        scene = rgb_rel_path.split('/')[0]
        frame = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
        dpth = os.path.join(dataset_root, dataset, scene, 'polarization', f'depth_{camera}', f'{frame:06d}.png')
        scale = 1.0  # PNG单位毫米 → 米需要 *1e-3（后面统一乘）
    elif dataset == 'HouseCat6D':
        scene = rgb_rel_path.split('/')[0]
        frame = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
        dpth = os.path.join(dataset_root, dataset, scene, 'depth', f'{frame:06d}.png')
        scale = 1.0
    elif dataset == 'PhoCAL':
        scene = rgb_rel_path.split('/')[0]
        frame = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
        dpth = os.path.join(dataset_root, dataset, scene, 'depth', f'{frame:06d}.png')
        scale = 1.0
    elif dataset == 'TransCG':
        scene = rgb_rel_path.split('/')[1]
        frame = int(rgb_rel_path.split('/')[-2])
        if camera == 'd435':
            dpth = os.path.join(dataset_root, dataset, 'scenes', scene, f'{frame}', 'depth1.png')
        elif camera == 'l515':
            dpth = os.path.join(dataset_root, dataset, 'scenes', scene, f'{frame}', 'depth2.png')
        scale = 1.0
    elif dataset == 'GN-Trans':
        scene = rgb_rel_path.split('/')[1]
        frame = int(rgb_rel_path.split('/')[-1].split('_')[0])
        dpth = os.path.join(dataset_root, dataset, 'scenes', scene, f'{frame:04d}_depth_sim.png')
        scale = 1.0
    elif dataset == 'XYZ-IBD':
        scene = rgb_rel_path.split('/')[1]
        frame = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
        dpth = os.path.join(dataset_root, dataset, 'val', scene, 'depth_xyz', f'{frame:06d}.png')
        scale = 0.09999999747378752  # 数据集自带比例
    elif dataset == 'YCB-V':
        scene = rgb_rel_path.split('/')[1]
        frame = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
        dpth = os.path.join(dataset_root, dataset, 'test', scene, 'depth', f'{frame:06d}.png')
        scale = 0.1  # YCB-V: 0.1 毫米单位
    elif dataset == 'T-LESS':
        scene = rgb_rel_path.split('/')[1]
        frame = int(os.path.splitext(os.path.basename(rgb_rel_path))[0])
        dpth = os.path.join(dataset_root, dataset, 'test_primesense', scene, 'depth', f'{frame:06d}.png')
        scale = 0.1
    elif dataset == 'ROBI':
        scale = 0.03125
        dpth = os.path.join(dataset_root, dataset, rgb_rel_path.replace("Stereo", "Depth").replace('LEFT_', 'DEPTH_').replace('.bmp', '.png'))
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")
    return dpth, scale

import time
def benchmark_inference(model,
                        example_inputs,
                        device='cuda',
                        n_warmup=10,
                        n_iters=50):
    """
    精确测 PyTorch 模型单次 forward 时间（不含数据加载等开销）

    model: 已构建好的 nn.Module
    example_inputs: 和真实推理时 shape 一致的输入 tensor（或 tuple/list of tensors）
    device: 'cuda' or 'cpu'
    n_warmup: 预热次数（不计时）
    n_iters: 正式计时的迭代次数
    """
    assert device in ['cuda', 'cpu']

    # 关闭梯度
    torch.set_grad_enabled(False)

    # -------- warm-up，不计时 --------
    for _ in range(n_warmup):
        _ = model(example_inputs)
    if device == 'cuda':
        torch.cuda.synchronize()

    times_ms = []

    # -------- 正式计时 --------
    if device == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)

        for _ in range(n_iters):
            starter.record()
            _ = model(example_inputs)
            ender.record()
            torch.cuda.synchronize()              # 等 GPU 完成
            times_ms.append(starter.elapsed_time(ender))  # 单位: ms
    else:
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model(example_inputs)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)   # s -> ms

    times = torch.tensor(times_ms)
    mean_ms = times.mean().item()
    std_ms  = times.std(unbiased=False).item()

    # batch 维度（用于算 per-image 时间 / FPS）
    if torch.is_tensor(example_inputs):
        batch_size = example_inputs.size(0)
    elif isinstance(example_inputs, (list, tuple)) and torch.is_tensor(example_inputs[0]):
        batch_size = example_inputs[0].size(0)
    else:
        batch_size = 1

    print(f"[{device}] {n_iters} runs, batch_size={batch_size}")
    print(f"  mean  : {mean_ms:.3f} ms / batch")
    print(f"  std   : {std_ms:.3f} ms")
    print(f"  per-img: {mean_ms / batch_size:.3f} ms")
    print(f"  FPS   : {1000.0 / (mean_ms / batch_size):.2f}")

    return mean_ms, std_ms, times_ms

# ----------------------------
# 主推断（多数据集，读取 split）
# ----------------------------
def test_mixed(args):
    # 1) 模型
    if args.model == 'OGNIDC':
        net = OGNIDC(args)
    else:
        raise TypeError(args.model, ['OGNIDC'])
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), f"file not found: {args.pretrain}"
        ckpt = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(ckpt['net'], strict=False)
        if key_u:
            print('[OMNI-DC] Unexpected keys:', key_u)
        if key_m:
            print('[OMNI-DC] Missing keys:', key_m)
            raise KeyError
        print(f'[OMNI-DC] Checkpoint loaded: {args.pretrain}')

    # 简化：DP 即可
    net = nn.DataParallel(net).eval()

    # 2) 读取 split（每行第一个字段是 rgb 相对路径）
    rgb_rel_paths = read_split_rgb_list(args.split)
    print(f'[INFO] Loaded {len(rgb_rel_paths)} samples from split: {args.split}')

    # 3) 预处理
    t_rgb = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # depth 直接 ToTensor（H,W → 1,H,W）
    t_dep = T.ToTensor()

    # 4) 推断循环
    depth_factor_mm = 1000.0  # 保存为 uint16 毫米

    diviser = int(4 * 2 ** (args.num_resolution - 1))  # 与 demo 中一致

    for rgb_rel in tqdm(rgb_rel_paths, desc=f'Infer {args.dataset}'):
        rgb_path = os.path.join(args.dataset_root, args.dataset, rgb_rel)
        if not os.path.exists(rgb_path):
            print(f'[Warn] RGB not found: {rgb_path}')
            continue

        # 对应 raw/sparse depth
        try:
            depth_path, dscale = infer_depth_path_and_scale(args.dataset, args.dataset_root, rgb_rel, args.camera)
        except Exception as e:
            print('[Warn] depth path infer failed:', rgb_rel, e)
            continue
        if not os.path.exists(depth_path):
            print(f'[Warn] DEP not found: {depth_path}')
            continue

        # 读图
        try:
            rgb_np = np.array(Image.open(rgb_path))
            if rgb_np.ndim == 3 and rgb_np.shape[2] == 4:
                rgb_np = rgb_np[:, :, :3]
            elif rgb_np.ndim == 2:
                rgb_np = np.repeat(rgb_np[..., None], 3, axis=2)
            dep_raw = np.array(Image.open(depth_path)).astype(np.float32) * dscale * 1e-3  # -> 米
        except Exception as e:
            print('[Error] read image/depth failed:', rgb_path, depth_path, e)
            continue

        H_raw, W_raw = rgb_np.shape[:2]

        # 可选 resize 到 args.img_height/width（与 demo 不强制一致，方便多数据集）
        H_in, W_in = (args.img_height, args.img_width)
        if (H_in is not None and W_in is not None) and (H_raw != H_in or W_raw != W_in):
            rgb_np_resz = np.array(Image.fromarray(rgb_np).resize((W_in, H_in), Image.BICUBIC))
            dep_resz = np.array(Image.fromarray(dep_raw).resize((W_in, H_in), Image.NEAREST)).astype(np.float32)
        else:
            rgb_np_resz = rgb_np
            dep_resz = dep_raw
            H_in, W_in = H_raw, W_raw

        # To tensor
        rgb = t_rgb(Image.fromarray(rgb_np_resz)).unsqueeze(0).cuda()   # [1,3,H,W]
        dep = t_dep(dep_resz).unsqueeze(0).cuda()                       # [1,1,H,W]
        K = torch.eye(3).reshape(1, 3, 3).cuda()  # not used, keep API

        sample = {'rgb': rgb, 'dep': dep, 'K': K, 'pattern': 0}

        # Padding（对齐至 diviser）
        _, _, H, W = rgb.shape
        H_pad = ( (H // diviser + 1) * diviser - H ) if (H % diviser != 0) else 0
        W_pad = ( (W // diviser + 1) * diviser - W ) if (W % diviser != 0) else 0
        if H_pad > 0 or W_pad > 0:
            rgb = F.pad(rgb, (0, W_pad, 0, H_pad))
            dep = F.pad(dep, (0, W_pad, 0, H_pad))
            sample['rgb'] = rgb
            sample['dep'] = dep

        # 前向
        # print(sample['rgb'].shape)
        # benchmark_inference(net, sample, 'cuda', n_warmup=10, n_iters=50)
        with torch.no_grad():
            out = net(sample)

        # 去 padding
        if H_pad > 0 or W_pad > 0:
            pred = out['pred'][..., :H, :W]
        else:
            pred = out['pred']

        # 如果 resize 过，resize 回原图大小
        pred_np = pred.squeeze(1).squeeze(0).detach().cpu().numpy()
        if (H_raw != H_in) or (W_raw != W_in):
            pred_np = np.array(Image.fromarray(pred_np).resize((W_raw, H_raw), Image.NEAREST)).astype(np.float32)

        # 保存为 uint16 毫米
        pred_mm = np.clip(pred_np * depth_factor_mm, 0, 65535).astype(np.uint16)

        # 组织保存路径
        # 例：<dataset_root>/<dataset>/.../<scene>/<frame>.png
        # 我们按相对路径构造一个层级
        # 默认保存到：output_root/dataset/method/<scene>/XXXXXX_depth.png
        # scene 推断：
        parts = rgb_rel.split('/')
        # 各数据集的 scene 位置不同，这里尽量通用处理：
        if args.dataset in ['HAMMER', 'HouseCat6D', 'PhoCAL']:
            scene = parts[0]
            frame_id = int(os.path.splitext(parts[-1])[0])
        elif args.dataset in ['TransCG']:
            scene = parts[1]
            frame_id = int(parts[-2])
        elif args.dataset in ['GN-Trans']:
            scene = parts[1]
            frame_id = int(parts[-1].split('_')[0])
        elif args.dataset in ['XYZ-IBD', 'YCB-V', 'T-LESS']:
            scene = parts[1]
            frame_id = int(os.path.splitext(parts[-1])[0])
        elif args.dataset in ['ROBI']:
            scene = parts[1] + '_' + parts[2]
            frame_id = int(parts[-1].split('.')[0].split('_')[-1])
        else:
            scene = parts[0]
            frame_id = int(os.path.splitext(parts[-1])[0])

        if args.dataset in ['HAMMER', 'TransCG']:
            save_dir = os.path.join(args.output_root, args.dataset, args.camera, args.method, scene)
        else:
            save_dir = os.path.join(args.output_root, args.dataset, args.method, scene)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{frame_id:06d}_depth.png')
        Image.fromarray(pred_mm).save(save_path)

    print('[DONE] Inference finished.')

def main(args):
    init_seed()
    # 打印主要参数
    print('\n\n=== Arguments (from config.py) ===')
    cnt = 0
    for key in sorted(vars(args)):
        print(key, ':', getattr(args, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    # 走混合数据集推断
    test_mixed(args)


if __name__ == '__main__':
    args_cli = get_cli_args()
    args_main = merge_args(args_config, args_cli)
    main(args_main)