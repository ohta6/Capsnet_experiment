import matplotlib.pyplot as plt

accs = []
with open("test_acc_0.txt", "r") as f:
    for line in f:
        accs.append(float(line))
sparcity = []
with open("model_sparcity_0.txt", "r") as f:
    for line in f:
        sparcity.append(float(line))
l1 = [0.1**t for t in range(3, 8)]
plt.plot(l1, accs, label="accuracy")
plt.plot(l1, sparcity, label="sparcity")
plt.xscale("log")
plt.xlabel("L1 coefficient")
plt.ylabel("accuracy")
plt.title("accuracy of mnist")
plt.legend()
plt.savefig("test_acc_mnist.png")
plt.close()

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
