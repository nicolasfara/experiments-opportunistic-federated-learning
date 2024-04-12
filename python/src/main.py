import copy
import torch


def average_weights(models):
    ''' Averages the weights

    Args:
        models (list): a list of state_dict

    Returns:
        state_dict: the average state_dict
    ''' 
    w_avg = copy.deepcopy(models[0])
    for key in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[key] += models[i][key]
        w_avg[key] = torch.div(w_avg[key], len(models))
    return w_avg