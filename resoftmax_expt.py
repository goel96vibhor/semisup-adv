# %%
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch


# %%
softmax = torch.nn.Softmax(dim=1)


# %%
def plot_histogram(cifar_conf_vals, noncifar_conf_vals, dataset):
    fig, axs = plt.subplots(1, 2, constrained_layout=True)

    mu, std = norm.fit(cifar_conf_vals)
    axs[0].hist(cifar_conf_vals, bins=100)
    axs[0].set_title('P1: $\mu=%.2f, \sigma=%.2f$ \ncount=%d' % (mu, std, len(cifar_conf_vals)))
    axs[0].set_xlabel('Confidence')
    axs[0].set_ylabel('Number of examples')

    mu, std = norm.fit(noncifar_conf_vals)
    axs[1].hist(noncifar_conf_vals, bins=100)
    axs[1].set_title('Non-CIFAR: $\mu=%.2f, \sigma=%.2f$ \ncount=%d' % (mu, std, len(noncifar_conf_vals)))
    axs[1].set_xlabel('Confidence')
    axs[1].set_ylabel('Number of examples')

    fig.suptitle('Histogram for {} Dataset confidence distribution'.format(dataset), fontsize=14)
    
    plt.show()


# %%
df = pd.read_csv('cifar10-vs-ti/tinyimages.csv')

confs, is_c10 = df.iloc[:, 2:13], df.iloc[:, -2]
last_less_8 = confs.iloc[:, -1] < .8
p1_more_1 = confs.iloc[:, :-1].max(axis=1) > .1

print(len(last_less_8), len(p1_more_1))
# print(last_less_8)
print(last_less_8.isin([True]).sum(axis=0))
print(last_less_8.isin([False]).sum(axis=0))

# print(p1_more_1)
print(p1_more_1.isin([True]).sum(axis=0))
print(p1_more_1.isin([False]).sum(axis=0))


# %%
selection_logic = last_less_8 & p1_more_1
print('Trues count:', selection_logic.isin([True]).sum(axis=0))
print('Falses count:', selection_logic.isin([False]).sum(axis=0))


# %%
selected_ex = confs[selection_logic]
p1_confs = selected_ex.iloc[:, :-1]
last_confs = selected_ex.iloc[:, -1]


# %%
# Resoftmax
p1_confs = torch.tensor(p1_confs.values)
p1_confs = softmax(p1_confs)
p1_max_conf, p1_preds = torch.max(p1_confs, dim=1)


# %%
print(p1_max_conf)
print(p1_max_conf.shape)


# %%
plot_histogram(p1_max_conf, last_confs, 'TinyImages')

# %%
