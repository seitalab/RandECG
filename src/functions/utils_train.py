import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix as confusion_matrix_calc
from sklearn.metrics import f1_score

def prepare_optim(optim_type, model, lr):
    if optim_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_type == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optim_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optim_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optim_type == "asgd":
        optimizer = optim.ASGD(model.parameters(), lr=lr)
    else:
        print("Invalid optimizer chosen")
        raise
    return optimizer

# Class weights ***********************************************************
cw_patterns_3class = {
    "same": [1, 1, 1],
}

cw_patterns_9class = {
    "same": [1, 1, 1, 1, 1, 1, 1, 1, 1],
}

# Loss function preparation ***********************************************
def prepare_lossfunc_3class(args, label_count, num_samples):
    if args.cw == "auto":
        weight = [1, label_count["N"]/label_count["A"],
                  label_count["N"]/label_count["O"]]
        C = torch.Tensor(weight)
    else:
        C = torch.Tensor(cw_patterns_3class[args.cw])
    print("Using class weight of {}".format(C))

    loss_func = nn.CrossEntropyLoss(weight=C, reduction="none")
    loss_func = loss_func.to(args.device)
    return loss_func

def prepare_lossfunc_9class(args, label_count, num_samples):
    if args.cw == "auto":
        weights = [num_samples / count for count in label_count]
    else:
        weights = torch.Tensor(cw_patterns_9class[args.cw])

    _w = [round(w, 2) for w in weights]
    print("Positive class weight:", _w)
    weights = torch.tensor(weights).float()
    loss_func = nn.BCEWithLogitsLoss(reduction="none",
                                     pos_weight=weights).to(args.device)
    return loss_func

# Score calculation *******************************************************
def calculate_f1score_3class(confusion_matrix, index):
    score = confusion_matrix[index, index]
    total_ref = confusion_matrix[index].sum()
    total_pred = confusion_matrix[:, index].sum()
    f1score  = (2 * score) / (total_ref + total_pred)
    return f1score

def calculate_score_3class(y_probs, y_trues):
    y_preds = np.argmax(y_probs, axis=1)
    result = confusion_matrix_calc(y_trues, y_preds)
    f1_N = calculate_f1score_3class(result, 0)
    f1_A = calculate_f1score_3class(result, 1)
    f1_O = calculate_f1score_3class(result, 2)

    # in case no `T` class in batch
    avgF1 = (f1_N + f1_A + f1_O) / 3
    return avgF1, (f1_N, f1_A, f1_O)

# 9class
def calculate_f1score_9class(confusion_matrix):
    TP = confusion_matrix[1, 1]
    total_ref = confusion_matrix[1].sum()
    total_pred = confusion_matrix[:, 1].sum()
    f1score  = (2 * TP) / (total_ref + total_pred)
    return f1score

def calculate_score_9class(y_probs, y_trues):
    y_preds = (y_probs > 0.5).astype("int")
    result = multilabel_confusion_matrix(y_trues, y_preds)

    f1scores = []
    for i in range(9):
        f1scores.append(calculate_f1score_9class(result[i]))

    avgF1 = sum(f1scores) / 9
    return avgF1, f1scores
