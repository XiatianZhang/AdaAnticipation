from main_GCN_head import  val_GCN
import logging
import wandb
import pprint

logging.basicConfig(level=logging.DEBUG,
                    filename='train.log',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )


sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric

parameters_dict = {
        'num_classes_phase':
            {'values':[6,]},
        'epochs':{
            'values': [60]
        },
        'rep_channels':{
            'values': [8]
        },
        't_k': {
            'values': [13]
        },
        'stage_t': {
            'values': [2]
        },
        'stage_p': {
            'values': [2]
        },
        'layer_t': {
            'values': [5]
        },
        'layer_p': {
            'values': [8]
        },
        'out_channel_t': {
            'values': [64]
        },
        'out_channel_p': {
            'values': [32]
        },
        'dropout': {
            'values': [0.3]
        },
        'weight1':{
            'values': [1,]
        },
        'weight2':{
            'values': [1,]
        },
        'weight3':{
            'values': [1,]
        },
        'weight4':{
            'values': [0,]
        },
    "batch_size": {
        'values': [2]
        },
    "learning_rate": {
        'values': [0.002]
        },
    "weight_decay": {
        'values': [0.01]
        },
    "alpha": {
        'values':[0],
        },
    "seed": {
    'values': [3072023]
    },
}

sweep_config['parameters'] = parameters_dict


sweep_id = wandb.sweep(sweep_config, project="C80-Right-h7-final")

Val = val_GCN(14062022)

wandb.agent(sweep_id, Val.val, count=1)
