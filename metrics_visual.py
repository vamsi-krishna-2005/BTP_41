import matplotlib.pyplot as plt
import torch
import pandas as pd

def plot_metrics(history, model_name):
    epochs = [h['epoch'] for h in history]
    
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    
    train_miou = [h['train_mIoU'] for h in history]
    val_miou = [h['val_mIoU'] for h in history]
    
    avg_gaf = [h['avg_gaf'] for h in history]
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss over Epochs')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_miou, label='Train mIoU')
    plt.plot(epochs, val_miou, label='Val mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.title(f'{model_name} mIoU over Epochs')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, avg_gaf, label='Avg GAF', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('GAF Score')
    plt.title(f'{model_name} GAF Score over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/Metrics_visual_after_albumentation_{model_name}.png")


unet_history = pd.read_csv('results/Albumented_unet_gpu_metrics.csv').to_dict('records')
swin_history = pd.read_csv('results/Albumented_swin_gpu_metrics.csv').to_dict('records')
plot_metrics(unet_history, 'U-Net')
#plot_metrics(swin_history, 'Swin-UNet')
