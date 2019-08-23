import matplotlib.pyplot as plt
import pandas as pd

def plot_log(foldername):
    df = pd.read_csv(foldername+'/log.csv')

    plt.subplot(1, 2, 1)
    plt.title(foldername+'_acc')
    plt.plot(df['epoch'], df['capsnet_acc'], label='capsnet_acc')
    plt.plot(df['epoch'], df['val_capsnet_acc'], label='val_capsnet_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(foldername+'_loss')
    plt.plot(df['epoch'], df['capsnet_loss'], label='capsnet_loss')
    plt.plot(df['epoch'], df['decoder_loss'], label='decoder_loss')
    plt.plot(df['epoch'], df['loss'], label='loss')
    plt.plot(df['epoch'], df['val_capsnet_loss'], label='val_capsnet_loss')
    plt.plot(df['epoch'], df['val_decoder_loss'], label='val_decoder_loss')
    plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(foldername+'/acc_loss.png')
    plt.clf()
def plot_log2(foldernames):
    df1 = pd.read_csv(foldernames[0]+'/log.csv')
    df2 = pd.read_csv(foldernames[1]+'/log.csv')
    df3 = pd.read_csv(foldernames[2]+'/log.csv')
    df2['epoch'] += 50
    df3['epoch'] += 100
    df = pd.concat([df1, df2, df3])
    plt.title(foldernames[0]+'_acc')
    plt.plot(df['epoch'], df['capsnet_acc'], label='capsnet_acc')
    plt.plot(df['epoch'], df['val_capsnet_acc'], label='val_capsnet_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    #plt.ylim(0.8, 1.0)
    plt.savefig('cifar_acc.png')
    plt.clf()
if __name__=='__main__':
    """
    plot_log('result_mnist')
    plot_log('result_mnist_l1')
    plot_log('result_mnist_retrain')
    plot_log('result_fashion_mnist')
    plot_log('result_fashion_mnist_l1')
    plot_log('result_fashion_mnist_retrain')
    plot_log('result_svhn')
    plot_log('result_svhn_l1')
    plot_log('result_svhn_retrain')
    plot_log2(['result_mnist', 'result_mnist_l1', 'result_mnist_retrain'])
    plot_log2(['result_fashion_mnist', 'result_fashion_mnist_l1', 'result_fashion_mnist_retrain'])
    plot_log2(['result_svhn', 'result_svhn_l1', 'result_svhn_retrain'])
    plot_log2(['result_cifar10', 'result_cifar10_l1', 'result_cifar10_retrain'])
    """
    
    plot_log('result/food_batch30')
