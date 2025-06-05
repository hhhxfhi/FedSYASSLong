import os.path

import numpy as np
import ujson

import torch


def check(args, config_path):
    """
    判断当前配置文件是否匹配已有设置，主要是判断当前状态是否是debug状态
    :param args: 模型参数
    :param config_path: 配置文件路径
    """
    if not args.debug:
        return False

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)

        if config["train_config"]["distribution"] == args.distribution:
            if args.distribution == "dirichlet":
                if "alpha" in config["train_config"] and config["train_config"]["alpha"] == args.alpha and \
                        config["train_config"]["client"]["num_clients"] == args.client["num_clients"]:
                    return True
            elif args.distribution == "non_balanced":
                if config["train_config"]["client"]["num_clients"] == args.client["num_clients"] and \
                        config["train_config"]["client"]["num_classes_per_client"] == args.client["num_classes_per_client"]:
                    return True
            elif args.distribution == "iid":
                if config["train_config"]["client"]["num_clients"] == args.client["num_clients"]:
                    return True
    return False


def save_data(train_config, statistic, train_subsets, test_subsets, config_path, train_path, test_path):
    config = {
        "train_config": train_config,
        "statistic": statistic
    }
    with open(config_path, 'w') as f:
        ujson.dump(config, f, indent=2)

    torch.save(train_subsets, train_path)
    torch.save(test_subsets, test_path)


def load_data(train_path, test_path):
    train_subsets = torch.load(train_path)
    test_subsets = torch.load(test_path)
    return train_subsets, test_subsets


def read_data(dataset, idx, is_train=True):
    # rawdata_path = os.path.join("dataset/rawdata/", dataset)
    if is_train:
        train_data_dir = os.path.join("dataset/Cifar100/", 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join("dataset/Cifar100/", 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

