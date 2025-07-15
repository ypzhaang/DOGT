# d
import os
import h5py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data, Dataset as GeoDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.datasets import ZINC
from abide.Rest.dataloader import BrainDataset
from utils import get_grid_edge_index, _build_edge_index
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split

def load_molhiv_dataset(args):
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=os.path.join(args.data_dir, 'ogbg'))
    split_idx = dataset.get_idx_split()
    data_list = dataset[:]

    train_loader = GeoDataLoader([data_list[i] for i in split_idx['train']],
                              batch_size=args.batch_size, shuffle=True)
    valid_loader = GeoDataLoader([data_list[i] for i in split_idx['valid']],
                               batch_size=args.batch_size, shuffle=False)
    test_loader = GeoDataLoader([data_list[i] for i in split_idx['test']],
                              batch_size=args.batch_size, shuffle=False)

    num_classes = 1  # Binary classification
    num_features = dataset.num_node_features

    return dataset, train_loader, valid_loader, test_loader, num_features, num_classes

def load_molbace_dataset(args):
    dataset = PygGraphPropPredDataset(name='ogbg-molbace', root=os.path.join(args.data_dir, 'ogbg'))
    split_idx = dataset.get_idx_split()
    data_list = dataset[:]

    train_loader = GeoDataLoader([data_list[i] for i in split_idx['train']],
                              batch_size=args.batch_size, shuffle=True)
    valid_loader = GeoDataLoader([data_list[i] for i in split_idx['valid']],
                               batch_size=args.batch_size, shuffle=False)
    test_loader = GeoDataLoader([data_list[i] for i in split_idx['test']],
                              batch_size=args.batch_size, shuffle=False)

    num_classes = 1  # Binary classification
    num_features = dataset.num_node_features

    return dataset, train_loader, valid_loader, test_loader, num_features, num_classes

def load_zinc_dataset(args):
    # 加载完整 ZINC（非 subset）
    dataset = ZINC(root=os.path.join(args.data_dir, 'zinc'), subset=False)

    # 按照 PyG 的默认顺序切分，自己划分
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_dataset = dataset[:train_len]
    valid_dataset = dataset[train_len:train_len + val_len]
    test_dataset = dataset[train_len + val_len:]

    train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeoDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = GeoDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    num_classes = 1  # 回归任务
    num_features = dataset.num_node_features

    return dataset, train_loader, valid_loader, test_loader, num_features, num_classes

def load_zinc12k_dataset(args):
    root = os.path.join(args.data_dir, 'zinc')

    # 1. 分别读取 train/val/test
    train_ds = ZINC(root=root, subset=True, split='train')
    val_ds   = ZINC(root=root, subset=True, split='val')
    test_ds  = ZINC(root=root, subset=True, split='test')

    # 2. 合并到一个 dataset_all
    dataset = ConcatDataset([train_ds, val_ds, test_ds])
    assert len(dataset) == len(train_ds) + len(val_ds) + len(test_ds)  # 应该 = 12 000

    # 3. 根据合并后的索引，划分出三个子集
    n_train = len(train_ds)
    n_val   = len(val_ds)
    # train:   [0, n_train)
    # valid:   [n_train, n_train + n_val)
    # test:    [n_train + n_val, n_train + n_val + len(test_ds))
    train_idx = list(range(0,           n_train))
    val_idx   = list(range(n_train,    n_train + n_val))
    test_idx  = list(range(n_train + n_val, n_train + n_val + len(test_ds)))

    train_loader = GeoDataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True)
    valid_loader = GeoDataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size, shuffle=False)
    test_loader  = GeoDataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size, shuffle=False)

    num_classes  = 1
    num_features = train_ds.num_node_features

    return dataset, train_loader, valid_loader, test_loader, num_features, num_classes

def load_molesol_dataset(args):
    dataset = PygGraphPropPredDataset(name='ogbg-molesol', root=os.path.join(args.data_dir, 'ogbg'))
    data_list = dataset[:]

    # 设置随机种子保证可复现
    torch.manual_seed(args.seed)

    # 计算划分数量
    total_len = len(data_list)
    train_len = int(0.7 * total_len)
    valid_len = int(0.2 * total_len)
    test_len = total_len - train_len - valid_len 
    
    # 随机划分数据
    train_data, valid_data, test_data = torch.utils.data.random_split(data_list, [train_len, valid_len, test_len])

    # 构造 DataLoader
    train_loader = GeoDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeoDataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    num_classes = 1  # 回归任务
    num_features = dataset.num_node_features

    return dataset, train_loader, valid_loader, test_loader, num_features, num_classes

class BrainGraphDataset(GeoDataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x = sample["x"]  # [N, N] Pearson correlation matrix
        y = sample["y"]

        # 使用阈值筛选边（只保留相关性 > 0.5 的边）
        threshold = 0.5
        edge_index = (x > threshold).nonzero(as_tuple=False).t().contiguous()

        # 特征设为对角线（自连接值）或度信息（你可以根据需要改）
        node_feature = torch.diag(x).unsqueeze(1)  # shape [N, 1]

        return Data(x=node_feature, edge_index=edge_index, y=y)

def load_abide_dataset(args):
    data_file_path = "/home/dell/sx/sgformer/SGFormer/large/new/abide/Rest/abide_data.h5"
    split_base_path = "/home/dell/sx/sgformer/SGFormer/large/new/abide/Rest"

    dataset_name = "abide_data"
    task_name = "rest"
    file = h5py.File(data_file_path, "r")
    dataset = BrainDataset(
        file=file,
        y_key="class",  # 你现在用的是 'class' 字段作为 label
        dataset_name=dataset_name,
        task_name=task_name,
    )

    subjects = list(file.keys())

    def read_split(filename):
        with open(os.path.join(split_base_path, filename), "r") as f:
            subject_ids = f.read().strip().split("\n")
            return [subjects.index(sid) for sid in subject_ids if sid in subjects]

    train_ids = read_split("train.split")
    test_ids = read_split("test.split")

    val_split = int(0.1 * len(train_ids))
    val_ids = train_ids[-val_split:]
    train_ids = train_ids[:-val_split]

    train_dataset = Subset(dataset, train_ids)
    valid_dataset = Subset(dataset, val_ids)
    test_dataset  = Subset(dataset, test_ids)

    # 封装为 torch_geometric 的 Data 格式
    train_dataset = BrainGraphDataset(train_dataset)
    valid_dataset = BrainGraphDataset(valid_dataset)
    test_dataset  = BrainGraphDataset(test_dataset)

    train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = GeoDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = GeoDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 获取节点特征维度
    sample = train_dataset[0]
    num_features = sample.x.shape[-1]
    num_classes = 1  # 二分类

    dataset_all = ConcatDataset([train_dataset, valid_dataset, test_dataset])

    return dataset_all, train_loader, valid_loader, test_loader, num_features, num_classes

class FER2013GraphDataset(GeoDataset):
    def __init__(self, csv_path, usage='Training', target_emotions=None, label_map=None, connect_diag=False):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df[self.df['Usage'] == usage].reset_index(drop=True)

        if target_emotions is not None:
            self.df = self.df[self.df['emotion'].isin(target_emotions)].reset_index(drop=True)

        self.label_map = label_map
        self.edge_index = get_grid_edge_index(48, 48, connect_diag=connect_diag)

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.float32) / 255.0
        x = torch.tensor(pixels, dtype=torch.float32).unsqueeze(1)
        label = int(row['emotion'])
        y = torch.tensor([self.label_map[label] if self.label_map else label], dtype=torch.long)
        return Data(x=x, edge_index=self.edge_index, y=y)

def load_fer2013_graph_dataset(args):
    csv_path = "/home/dell/sx/sgformer/SGFormer/data/Fer2013/icml_face_data.csv"

    train_dataset = FER2013GraphDataset(csv_path, usage='Training')
    valid_dataset = FER2013GraphDataset(csv_path, usage='PublicTest')
    test_dataset  = FER2013GraphDataset(csv_path, usage='PrivateTest')

    train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeoDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = GeoDataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    num_features = 1       # 每个像素值作为一个特征
    num_classes = 7        # 表情有7类

    return train_dataset, train_loader, valid_loader, test_loader, num_features, num_classes

def load_fer2013_graph_dataset_happyandsad(args):
    csv_path = "/home/dell/sx/sgformer/SGFormer/data/Fer2013/icml_face_data.csv"
    target_emotions = [3, 4]  # 3 = Happy, 4 = Sad
    label_map = {3: 0, 4: 1}  # Happy → 0, Sad → 1
    max_samples_per_label = 1000

    def limit(df):
        return pd.concat([
            df[df['emotion'] == 3].iloc[:max_samples_per_label],
            df[df['emotion'] == 4].iloc[:max_samples_per_label]
        ]).reset_index(drop=True)

    train_dataset = FER2013GraphDataset(csv_path, 'Training', target_emotions, label_map)
    train_dataset.df = limit(train_dataset.df)

    valid_dataset = FER2013GraphDataset(csv_path, 'PublicTest', target_emotions, label_map)
    valid_dataset.df = limit(valid_dataset.df)

    test_dataset  = FER2013GraphDataset(csv_path, 'PrivateTest', target_emotions, label_map)
    test_dataset.df = limit(test_dataset.df)

    train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeoDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = GeoDataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    num_features = 1
    num_classes = 2

    return train_dataset, train_loader, valid_loader, test_loader, num_features, num_classes

def load_fer2013_graph_dataset_angryandfear(args):
    csv_path = "/home/dell/sx/sgformer/SGFormer/data/Fer2013/icml_face_data.csv"
    target_emotions = [0, 2]  # 0 = Angry, 2 = Fear
    label_map = {0: 0, 2: 1}  # Angry → 0, Fear → 1
    max_samples_per_label = 1000

    def limit(df):
        return pd.concat([
            df[df['emotion'] == 0].iloc[:max_samples_per_label],
            df[df['emotion'] == 2].iloc[:max_samples_per_label]
        ]).reset_index(drop=True)

    train_dataset = FER2013GraphDataset(csv_path, 'Training', target_emotions, label_map)
    train_dataset.df = limit(train_dataset.df)

    valid_dataset = FER2013GraphDataset(csv_path, 'PublicTest', target_emotions, label_map)
    valid_dataset.df = limit(valid_dataset.df)

    test_dataset  = FER2013GraphDataset(csv_path, 'PrivateTest', target_emotions, label_map)
    test_dataset.df = limit(test_dataset.df)

    train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = GeoDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = GeoDataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    num_features = 1
    num_classes = 2

    return train_dataset, train_loader, valid_loader, test_loader, num_features, num_classes

def load_mnist_dataset(args):
    # ------------- 1. 随机种子 & 变换 -------------
    torch.manual_seed(args.seed)
    transform = transforms.ToTensor()
    root_dir = os.path.join(args.data_dir, "mnist")

    # ------------- 2. 读取官方 train / test 并合并 -------------
    train_set = MNIST(root=root_dir, train=True, transform=transform, download=True)
    test_set = MNIST(root=root_dir, train=False, transform=transform, download=True)

    images = torch.cat([train_set.data, test_set.data])          # [N, 28, 28] uint8
    labels = torch.cat([train_set.targets, test_set.targets])    # [N]

    # ------------- 3. 仅保留数字 2 & 7，并重映射标签 -------------
    mask = (labels == 2) | (labels == 7)
    images = images[mask]
    labels = labels[mask]
    labels = (labels == 7).long()  # 2 → 0, 7 → 1

    # ------------- 4. 预计算 edge_index -------------
    edge_index = _build_edge_index()  # 28×28 固定，可复用

    # ------------- 5. 将每张图片转为 Data 对象 -------------
    data_list = []
    for img, y in zip(images, labels):
        x = img.float().view(-1, 1) / 255.0          # [784, 1], 归一化到 0-1
        data = Data(x=x, edge_index=edge_index, y=y.unsqueeze(0))
        data_list.append(data)

    # ------------- 6. 数据划分 -------------
    total_len = len(data_list)
    train_len = int(0.7 * total_len)
    valid_len = int(0.2 * total_len)
    test_len = total_len - train_len - valid_len

    train_data, valid_data, test_data = random_split(
        data_list,
        [train_len, valid_len, test_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # ------------- 7. DataLoader -------------
    train_loader = GeoDataLoader(train_data,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=4)
    valid_loader = GeoDataLoader(valid_data,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4)
    test_loader = GeoDataLoader(test_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=4)

    # ------------- 8. 其他信息 -------------
    num_features = 1
    num_classes = 2

    return data_list, train_loader, valid_loader, test_loader, num_features, num_classes