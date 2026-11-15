import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: Weight of recall relative to precision (beta > 1 favours recall)
    :return: F-Score rounded to three decimal places
    """
    TP, FN, FP = 0, 0, 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_t == 1 and y_p ==1:
            TP +=1
        elif y_t==1 and y_p == 0:
            FN+=1
        elif y_t==0 and y_p==1:
            FP+=1
    recall = TP / (TP+FN)
    precision = TP / (TP+FP)
    Fscore = (1+beta**2) * (precision*recall/(beta**2*precision + recall))
    return round(Fscore,3)

