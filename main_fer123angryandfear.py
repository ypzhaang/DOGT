import argparse
import os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt

from logger import Logger
from parse import parse_method, parser_add_main_args
from torch_geometric.utils import add_self_loops, remove_self_loops
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from dataset_new import *
from torchvision.utils import save_image

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate(model, loader, eval_func, device, args):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x = batch.x.float()
            out, _ , _ , _= model(batch.x, batch.edge_index, batch.batch)

            if args.dataset in ['zinc', 'zinc-mini']:
                y_true.append(batch.y.view(-1).cpu())
                y_pred.append(out.view(-1).cpu())
            elif args.dataset == 'abide':
                label = batch.y.view(-1) - 1
                label = label.view(-1, 1)
                y_true.append(label.cpu())
                y_pred.append(out.cpu())
            else:
                y_true.append(batch.y.cpu())
                y_pred.append(out.cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    return eval_func(y_pred, y_true)

# ------------------- 解析命令行 -------------------
parser = argparse.ArgumentParser(description='Training on molhiv')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

# ------------------- 固定随机种子 -------------------
fix_seed(args.seed)

# ------------------- 设备选择 -------------------
device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print("device", device)

# ------------------- 新增：最优验证跟踪初始化 -------------------
if args.dataset in ['molhiv', 'molbace', 'abide']:
    # 二分类任务 AUC 越大越好
    best_test = 0.0
    cmp_better = lambda new, best: new > best
else:
    # 回归任务 MAE/RMSE 越小越好
    best_test = float('inf')
    cmp_better = lambda new, best: new < best
best_epoch = -1
# ------------------------------------------------------

# === Dataset loading ===
if args.dataset == 'molhiv':
    dataset, train_loader, valid_loader, test_loader, num_features, num_classes = load_molhiv_dataset(args)
elif args.dataset == 'molbace':
    dataset, train_loader, valid_loader, test_loader, num_features, num_classes = load_molbace_dataset(args)
elif args.dataset == 'zinc':
    dataset, train_loader, valid_loader, test_loader, num_features, num_classes = load_zinc_dataset(args)
elif args.dataset == 'zinc-mini':
    dataset, train_loader, valid_loader, test_loader, num_features, num_classes = load_zinc12k_dataset(args)
elif args.dataset == 'abide':
    dataset, train_loader, valid_loader, test_loader, num_features, num_classes = load_abide_dataset(args)
elif args.dataset == 'molesol':
    dataset, train_loader, valid_loader, test_loader, num_features, num_classes = load_molesol_dataset(args)
elif args.dataset == 'fer123':
    dataset, train_loader, valid_loader, test_loader, num_features, num_classes = load_fer2013_graph_dataset(args)
elif args.dataset == 'fer123happyandsad':
    dataset, train_loader, valid_loader, test_loader, num_features, num_classes = load_fer2013_graph_dataset_happyandsad(args)
elif args.dataset == 'fer123angryandfear':
    dataset, train_loader, valid_loader, test_loader, num_features, num_classes = load_fer2013_graph_dataset_angryandfear(args)
else:
    raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

print(f"Loaded {args.dataset} | #graphs: {len(dataset)}, #features: {num_features}, #classes: {num_classes}")


model = parse_method(args, num_classes, num_features, device)
model = model.to(device)

if args.dataset in ['molhiv', 'molbace', 'abide']:
    criterion = nn.BCEWithLogitsLoss()
    eval_func = lambda pred, label: torchmetrics.functional.auroc(pred.sigmoid(), label.int(), task='binary')
elif args.dataset == 'zinc':
    criterion = nn.L1Loss()  # MAE
    eval_func = lambda pred, label: torchmetrics.functional.mean_absolute_error(pred, label)
elif args.dataset == 'zinc-mini':
    criterion = nn.L1Loss()  # MAE
    eval_func = lambda pred, label: torchmetrics.functional.mean_absolute_error(pred, label)
elif args.dataset == 'molesol':
    criterion = nn.MSELoss()  
    eval_func = lambda pred, label: torch.sqrt(nn.functional.mse_loss(pred, label)) 
elif args.dataset == 'fer123':
    criterion = nn.CrossEntropyLoss()
    eval_func = lambda pred, label: torchmetrics.functional.auroc(
        pred.softmax(dim=-1), 
        label.int(),           
        task='multiclass',
        num_classes=7,
        average='macro'        # 可选 'macro' / 'weighted' / 'none'
    )
elif args.dataset in ['fer123happyandsad', 'fer123angryandfear']:
    criterion = nn.CrossEntropyLoss()
    eval_func = lambda pred, label: torchmetrics.functional.auroc(
        pred.softmax(dim=1)[:, 1], label.int(), task='binary'
    )
else:
    raise ValueError("Unsupported dataset for loss/eval_func")

logger = Logger(args.runs, args)

mask_dir = "/home/dell/sx/sgformer/SGFormer/large/new/masks"
# ------------------- 主训练循环 -------------------
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.trans_weight_decay},
        {'params': model.params2, 'weight_decay': args.gnn_weight_decay},
        {'params': model.params3, 'weight_decay': args.gnn_for_origin_weight_decay},
        {'params': model.params4, 'weight_decay': args.hyper_weight_decay}
    ], lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        # 每个 epoch，记录最后一个 batch 的 masks
        last_masks = None

        for batch in train_loader:
            start = time.time()
            batch = batch.to(device)
            batch.x = batch.x.float()
            optimizer.zero_grad()

            out, diverse_loss, masks,_ = model(batch.x, batch.edge_index, batch.batch)
            last_masks = masks  # 保存当前 batch 的 masks

            if args.dataset in ['zinc', 'zinc-mini']:
                label = batch.y.view(-1).float()
                loss = criterion(out.view(-1), label)
            elif args.dataset == 'abide':
                label = batch.y.view(-1) - 1
                label = label.view(-1, 1)
                loss = criterion(out, label)
            elif args.dataset in ['fer123', 'fer123happyandsad', 'fer123angryandfear']:
                label = batch.y.long()
                loss = criterion(out, label)
            else:
                label = batch.y.float()
                loss = criterion(out, label)

            # 前 50 个 epoch 只用主 loss，之后加 diverse_loss
            if epoch >= 50:
                loss = loss + args.hyper_diverse_rate * diverse_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 定期评估
        if epoch % args.eval_step == 0:
            train_result = evaluate(model, train_loader, eval_func, device, args)
            valid_result = evaluate(model, valid_loader, eval_func, device, args)
            test_result  = evaluate(model, test_loader,  eval_func, device, args)
            logger.add_result(run, (train_result, valid_result, test_result))

            # 初始化保存目录
            eval_dir = os.path.join(mask_dir, args.dataset, f"epoch{epoch:03d}")
            os.makedirs(eval_dir, exist_ok=True)

            # label 0 和 1 最多保存 10 个图
            label_counts = {0: 0, 1: 0}
            max_per_label = 5

            # 遍历 batch 中每张图（graph）
            num_graphs = batch.num_graphs  # 图数量
            for graph_id in range(num_graphs):
                label = batch.y[graph_id].item()
                if label not in label_counts:
                    continue
                if label_counts[label] >= max_per_label:
                    continue

                # 找到该图的节点
                node_mask = (batch.batch == graph_id)  # shape: [total_nodes]
                node_indices = node_mask.nonzero(as_tuple=False).view(-1)  # 节点索引
                if node_indices.shape[0] != 2304:
                    continue  # 跳过异常图（非 48x48）

                # 获取当前图对应的 mask 矩阵：[8, 2304]
                selected_mask = last_masks[graph_id]

                # 保存 mask
                mask_path = os.path.join(eval_dir, f"label{label}_img{label_counts[label]}_mask.txt")
                np.savetxt(mask_path, selected_mask.detach().cpu().numpy(), fmt="%.6f")

                # 构建图像
                pixels = batch.x[node_indices].view(48, 48).detach().cpu().numpy()
                img_path = os.path.join(eval_dir, f"label{label}_img{label_counts[label]}.png")
                plt.imsave(img_path, pixels, cmap='gray')

                label_counts[label] += 1

                # 提前结束：两类各有 10 张
                if all(c >= max_per_label for c in label_counts.values()):
                    break

            # 打印当前状态
            if epoch % args.display_step == 0:
                print(f"Epoch {epoch:03d} | Loss {total_loss:.4f} | "
                      f"Train {train_result:.4f} | Valid {valid_result:.4f} | Test {test_result:.4f}")

    logger.print_statistics(run)

logger.print_statistics()
print(f"Best validation was {best_valid:.4f} at epoch {best_epoch}")
