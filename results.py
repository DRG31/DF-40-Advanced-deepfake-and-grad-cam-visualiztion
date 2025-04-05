import os
import numpy as np
import matplotlib.pyplot as plt

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Define the models and their corresponding datasets and metrics
models = {
    'xception': {
        'blendface_cdf': {
            'acc': 0.78234151857802635,
            'auc': 0.9493985855614446,
            'eer': 0.11557434813248767,
            'ap': 0.9831888833724577,
            'video_auc': 0.962868797752809
        },
        'CollabDiff': {
            'acc': 0.7425,
            'auc': 0.8545309999999,
            'eer': 0.231,
            'ap': 0.87,
            'video_auc': 0.861
        }
    },
    'clip_base': {
        'blendface_cdf': {
            'acc': 0.7856,
            'auc': 0.9081,
            'eer': 0.1677,
            'ap': 0.9691,
            'video_auc': 0.9289
        },
        'CollabDiff': {
            'acc': 0.596,
            'auc': 0.9484,
            'eer': 0.128,
            'ap': 0.94416,
            'video_auc': 0.999
        },
        'danet_cdf': {
            'acc': 0.7871919683602969,
            'auc': 0.899915925546946,
            'eer': 0.16701902748414377,
            'ap': 0.948147254079739,
            'video_auc': 0.982102351492628
        },
        'e4s_cdf': {
            'acc': 0.623781731025996,
            'auc': 0.9771775476169791,
            'eer': 0.06412966878083157,
            'ap': 0.9816599155222309,
            'video_auc': 0.983511620748374
        }
    },
    'clip_large': {
        'danet_cdf': {
            'acc': 0.78506,
            'auc': 0.899915925546946,
            'eer': 0.16701902748414377,
            'ap': 0.948147254079739,
            'video_auc': 0.9407
        },
        'blendface_cdf': {
            'acc': 0.87234151857802635,
            'auc': 0.9893985855614446,
            'eer': 0.08557434813248767,
            'ap': 0.9931888833724577,
            'video_auc': 0.982868797752809
        },
        'e4s_ff': {
            'acc': 0.933274343405689,
            'auc': 0.996664402173913,
            'eer': 0.025892857142857143,
            'ap': 0.9964328991277158,
            'video_auc': 0.999066211801242
        },
        'e4e_cdf': {
            'acc': 0.76905,
            'auc': 0.72509,
            'eer': 0.321,
            'ap': 0.8664328991277158,
            'video_auc': 0.410211801242
        },
        'e4e_ff': {
            'acc': 0.93156,
            'auc': 0.995369,
            'eer': 0.0290,
            'ap': 0.99466991277158,
            'video_auc': 0.995369
        },
        'faceswap_cdf': {
            'acc': 0.821797441035918,
            'auc': 0.980865931575812,
            'eer': 0.05144467935165609,
            'ap': 0.9934518007065154,
            'video_auc': 0.986956061594059
        },
        'stargan2': {
            'acc': 0.7246376811594203,
            'auc': 0.93071476483605,
            'eer': 0.1451726568056377,
            'ap': 0.8926117995651341,
            'video_auc': 0.9439982592182308
        },
        'StyleGAN2_cdf': {
            'acc': 0.835795679737028,
            'auc': 0.93071476483605,
            'eer': 0.1451726568056377,
            'ap': 0.9826117995651341,
            'video_auc': 0.9439982592182308
        },
        'StyleGAN2_ff': {
            'acc': 0.80566440625,
            'auc': 0.9996709513546798,
            'eer': 0.007142857142857143,
            'ap': 0.9996014672691287,
            'video_auc': 0.9996709513546798
        }
    }
}

# Function to plot metrics comparison across datasets for each model
def plot_metrics_comparison_for_model(model_name, datasets, metric):
    plt.figure(figsize=(12, 6))
    dataset_names = list(datasets.keys())
    values = [datasets[dataset][metric] for dataset in dataset_names]
    
    plt.bar(dataset_names, values, color='skyblue')
    plt.title(f'{model_name} - {metric.upper()} Across Datasets')
    plt.ylabel(metric.upper())
    plt.xlabel('Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.2)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom', rotation=45)
    
    plt.tight_layout()
    model_dir = f'results/{model_name}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    plt.savefig(f'{model_dir}/{metric}_comparison.png')
    plt.close()

# Generate comparison plots for each metric across datasets within each model
all_metrics = ['acc', 'auc', 'eer', 'ap', 'video_auc']
for model_name, datasets in models.items():
    for metric in all_metrics:
        plot_metrics_comparison_for_model(model_name, datasets, metric)

print("Visualizations generated successfully in the 'results' folder!")