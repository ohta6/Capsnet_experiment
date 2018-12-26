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
if __name__=='__main__':
    #plot_log('result_mnist')
    #plot_log('result_mnist_l1')
    #plot_log('result_fashion_mnist')
    plot_log('result_fashion_mnist_l1')

