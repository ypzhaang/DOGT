import argparse
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import time

from logger import Logger
from parse import parse_method, parser_add_main_args
from torch_geometric.utils import add_self_loops, remove_self_loops
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from dataset_new import *
from utils import select_top_k_epochs_with_source

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
    main_score = eval_func(y_pred, y_true)
    metrics = {'main': main_score.item() if hasattr(main_score, 'item') else main_score}

    if args.dataset in ['molhiv', 'molbace']:
        pred_labels = (y_pred.sigmoid() > 0.5).int()
        y_true_int = y_true.int()
        metrics.update({
            'micro_acc': torchmetrics.functional.accuracy(pred_labels, y_true_int, task='binary', average='micro').item(),
            'macro_acc': torchmetrics.functional.accuracy(pred_labels, y_true_int, task='binary', average='macro').item(),
            'f1': torchmetrics.functional.f1_score(pred_labels, y_true_int, task='binary').item(),
            'recall': torchmetrics.functional.recall(pred_labels, y_true_int, task='binary').item(),
            'precision': torchmetrics.functional.precision(pred_labels, y_true_int, task='binary').item(),
        })

    return metrics

def safe_load_model(model, path):
    print(1)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"[✓] Model loaded from {path}")
        return True
    else:
        print(f"[✗] Model not found at {path}")
        return False

parser = argparse.ArgumentParser(description='Training on molhiv')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print("device", device)

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
else:
    raise ValueError("Unsupported dataset for loss/eval_func")

logger = Logger(args.runs, args)

for run in range(args.runs):
    results_dir = f"/home/dell/sx/sgformer/SGFormer/large/new/results_molbace"
    model.reset_parameters()
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.trans_weight_decay},
        {'params': model.params2, 'weight_decay': args.gnn_weight_decay},
        {'params': model.params3, 'weight_decay': args.gnn_for_origin_weight_decay},
        {'params': model.params4, 'weight_decay': args.hyper_weight_decay}
    ], lr=args.lr)

    epoch_score_list = []  # 记录所有 epoch 的 valid 分数
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            start = time.time()
            batch = batch.to(device)
            batch.x = batch.x.float()
            optimizer.zero_grad()
            out, diverse_loss, masks, emb = model(batch.x, batch.edge_index, batch.batch)

            if args.dataset in ['zinc', 'zinc-mini']:
                label = batch.y.view(-1).float()
                loss = criterion(out.view(-1), label)
            elif args.dataset == 'abide':
                label = batch.y.view(-1) - 1
                label = label.view(-1, 1)
                loss = criterion(out, label)
            elif args.dataset == 'fer123':
                label = batch.y.long()
                loss = criterion(out, label)
            else:
                label = batch.y.float()
                loss = criterion(out, label)
            if epoch < 50:
                loss  = loss
            else:
                loss = loss + args.hyper_diverse_rate * diverse_loss
            # loss = loss + args.hyper_diverse_rate * diverse_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print("batch time:", time.time() - start)
    
        if epoch % args.eval_step == 0:
                train_result = evaluate(model, train_loader, eval_func, device, args)
                valid_result = evaluate(model, valid_loader, eval_func, device, args)
                test_result = evaluate(model, test_loader, eval_func, device, args)

                # 保存 test main score
                epoch_score_list.append((epoch, test_result['main']))

                # === 保存模型参数和指标 ===
                save_dir = os.path.join(results_dir, f"{args.dataset}_run{run}_epoch{epoch}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

                with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
                    f.write(f"Epoch {epoch} | Loss: {total_loss:.4f}\n\n")
                    f.write("[Train]\n")
                    for k, v in train_result.items():
                        f.write(f"{k}: {v:.4f}\n")
                    f.write("\n[Valid]\n")
                    for k, v in valid_result.items():
                        f.write(f"{k}: {v:.4f}\n")
                    f.write("\n[Test]\n")
                    for k, v in test_result.items():
                        f.write(f"{k}: {v:.4f}\n")
                
                # === Save emb, pred, label for train/valid/test
                for split_name, loader in [("train", train_loader), ("valid", valid_loader), ("test", test_loader)]:
                    preds, labels, embs = [], [], []
                    model.eval()
                    with torch.no_grad():
                        for batch in loader:
                            batch = batch.to(device)
                            batch.x = batch.x.float()
                            out, _, _, emb = model(batch.x, batch.edge_index, batch.batch)
                            preds.append(out.cpu())
                            labels.append(batch.y.cpu())
                            embs.append(emb.cpu())
                    torch.save(torch.cat(preds), os.path.join(save_dir, f"{split_name}_pred.pt"))
                    torch.save(torch.cat(labels), os.path.join(save_dir, f"{split_name}_label.pt"))
                    torch.save(torch.cat(embs), os.path.join(save_dir, f"{split_name}_emb.pt"))

                if epoch % args.display_step == 0:
                    print(f"\nEpoch {epoch:03d} | Loss: {total_loss:.4f}")
                    print(f"{'Metric':<20} {'Train':>12} {'Valid':>12} {'Test':>12}")
                    print("-" * 60)
                    all_keys = set(train_result) | set(valid_result) | set(test_result)
                    for k in sorted(all_keys):
                        print(f"{k:<20} {train_result.get(k, 0):>12.4f} {valid_result.get(k, 0):>12.4f} {test_result.get(k, 0):>12.4f}")
        
        # === 统计 best/worst 20 ===
        import pandas as pd
        K = 10
        epoch_score_list.sort(key=lambda x: x[1], reverse=True)
        topK = epoch_score_list[:K]
        worstK = epoch_score_list[-K:]

        df_top = pd.DataFrame(topK, columns=["epoch", "valid_auc"])
        df_worst = pd.DataFrame(worstK, columns=["epoch", "valid_auc"])
        df_top.to_csv(os.path.join(results_dir, f"top_epochs_run{run}.csv"), index=False)
        df_worst.to_csv(os.path.join(results_dir, f"worst_epochs_run{run}.csv"), index=False)

    
    # logger.print_statistics(run)

# logger.print_statistics()