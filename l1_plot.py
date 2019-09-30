import matplotlib.pyplot as plt
import pandas as pd

datasets = ["mnist", "cifar"]
retrains = list(range(-5, -2))
versions = range(3, 4)
for dataset in datasets:
    for version in versions:
        acc_ax = plt.subplot(3, 1, 1)
        sparsity_ax = plt.subplot(3, 1, 2)
        time_ax = plt.subplot(3, 1, 3)
        acc_ax.set_xlabel("log of L1 coefficient")
        acc_ax.set_ylabel("accuracy [%]")
        sparsity_ax.set_xlabel("log of L1 coefficient")
        sparsity_ax.set_ylabel("sparsity [%]")
        time_ax.set_xlabel("log of L1 coefficient")
        time_ax.set_ylabel("time [s]")

        for retrain in retrains:
            base_name = "{}_{}_{}".format(dataset, retrain, version)
            csv_name = "output_{}.csv".format(base_name)
            df = pd.read_csv(csv_name)
            acc_ax.plot(df["l1"], df["acc"]*100, label="{}".format(retrain))
            sparsity_ax.plot(df["l1"], df["sparsity"]*100, label="{}".format(retrain))
            time_ax.plot(df["l1"], df["time"], label="{}".format(retrain))

        acc_ax.set_title("accuracy of {}".format(dataset))
        sparsity_ax.set_title("sparsity of {}".format(dataset))
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}.png".format(base_name))
        plt.close()
"""
accs = []
with open("test_acc_3.txt", "r") as f:
    for line in f:
        accs.append(float(line))
sparcity = []
with open("model_sparcity_3.txt", "r") as f:
    for line in f:
        sparcity.append(float(line))
l1 = [0.1**t for t in range(3, 8)]
plt.plot(l1, accs, label="accuracy")
plt.plot(l1, sparcity, label="sparcity")
plt.xscale("log")
plt.xlabel("L1 coefficient")
plt.ylabel("accuracy")
plt.title("accuracy of cifar")
plt.legend()
plt.savefig("test_acc_cifar.png")
plt.close()
"""
