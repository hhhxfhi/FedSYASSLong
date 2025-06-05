import os

# 定义命令参数
datasets = ['kronoDroid']
algorithms = ['FedProto', 'FedDistill','FedAvg','DBE','FedGH','FedPHP']
prefix = '630_final'
model_name = 'LeNet5'
benign_alone = '--benignAlone'

# 生成并运行命令
for dataset in datasets:
    for algorithm in algorithms:
        command = f"python main.py --dataset {dataset} --algorithm {algorithm} --prefix {prefix} --modelName {model_name} {benign_alone}"
        print(f"Running command: {command}")
        os.system(command)
