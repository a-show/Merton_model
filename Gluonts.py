from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import Dataset

wiki2k_download_link: str = "https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz"  # noqa: E501
import os
import tarfile
from pathlib import Path
from urllib import request
from gluonts.dataset.common import load_datasets
from gluonts.dataset.repository.datasets import get_dataset, get_download_path
default_dataset_path: Path = get_download_path() / "datasets"
dataset_name = "exchange_rate_nips"
if dataset_name == "wiki2000_nips":
    wiki_dataset_path = default_dataset_path / "wiki2000_nips"
    Path(default_dataset_path).mkdir(parents=True, exist_ok=True)
    if not wiki_dataset_path.exists():
        tar_file_path = wiki_dataset_path.parent / f"{"wiki2000_nips"}.tar.gz"
        request.urlretrieve(
            wiki2k_download_link,
            tar_file_path,
        )

        with tarfile.open(tar_file_path) as tar:
            tar.extractall(path=wiki_dataset_path.parent)

        os.remove(tar_file_path)
    datasets = load_datasets(
        metadata=wiki_dataset_path / "metadata",
        train=wiki_dataset_path / "train",
        test=wiki_dataset_path / "test",
    )
    train_list = np.concatenate([data.get("target") for data in datasets.train])
    test_list = np.concatenate([data.get("target") for data in datasets.test])
else:
    datasets = get_dataset(dataset_name)
    train_list = list(datasets.train)
    print(train_list[0])
    test_list = list(datasets.test)

from gluonts.evaluation import Evaluator
from Merton_model import MertonModel, COSMethodBasedDensityRecovery

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = []
x = []
N_train = len(train_list)
for i in range(N_train):
    x.append(torch.tensor(train_list[i]["target"]).reshape(-1))
    model.append(MertonModel(x[i], recovery_method=COSMethodBasedDensityRecovery(N_freq=500,device=device), device=device))
N = 100
N_test = len(test_list)
predict = []
test_data = []
for i in range(N_test):
    cat = i%N_train
    train_size = len(train_list[cat]["target"])
    test_data.append(torch.tensor(test_list[cat]["target"][train_size:]).reshape(-1))
    predict.append(model[cat].sample_path(x_0=train_list[cat]["target"][train_size-1], t=test_data[i].shape[0], n_sample_paths=N).cpu())

M = 10
plt.plot(torch.arange(-x[0].shape[0]+ 1, 1), x[0], label=f'Observed Path', color="#1E5EFF")
for i in range(M):
    plt.plot(torch.arange(predict[0].shape[1]), predict[0][i,:], label=f'Path {i+1}')

plt.plot(torch.arange(test_data[0].shape[0]), test_data[0], color="#1E5EFF") #label=f'Valid Path'

plt.xlabel('Time Steps')
plt.ylabel('Asset Price')  
plt.title('Sample Paths from Merton Model')
plt.legend()
plt.show()

from gluonts.model.forecast import SampleForecast
reference_series = []
forecast = []
for i in range(len(test_list)):
    cat = i%N_train
    samples = predict[i].numpy()
    start = test_list[i]['start']
    prediction_length = predict[i].shape[1]
    freq = test_list[i]['start'].freq
    forecast.append(SampleForecast(
        samples=samples,
        start_date = start + len(train_list[cat]['target']),
    ))
    if isinstance(start, pd.Period):
        start = start.to_timestamp()
    period_index = forecast[i].index
    reference_series.append(pd.Series(
        test_list[i]['target'][-prediction_length:],
        index=period_index
    ))

evaluator = Evaluator()
agg_metrics, item_metrics = Evaluator()( reference_series, forecast )
print(agg_metrics)

