import numpy as np

def ROC_curve(y_true:np.ndarray, y_score:np.ndarray, thresholds:np.ndarray):
    """Compute Receiver Operating Characteristic(ROC) curve.

    Args:
        y_true (np.ndarray): binary 1d array for true category labels.
        y_score (np.ndarray): 1d array of score, can be probability measure.
        thresholds (np.ndarray): array of thresholds for binary classification.

    Raises:
        TypeError: non boolean type element in y_true.

    Returns:
        np.ndarray : false_positive
            false positive rate.
        np.ndarray : true_positive
            true positive rate.
    """
    if y_true.dtype != bool:
        raise TypeError('y_true.dtype should be boolean.')
    false_positive = np.array([np.sum((y_score>threshold)*(~y_true))/np.sum(~y_true)
                     for threshold in thresholds])
    true_positive  = np.array([np.sum((y_score>threshold)*(y_true))/np.sum(y_true)
                     for threshold in thresholds])
    return false_positive, true_positive

def Youden_Index(fpr:np.ndarray, tpr:np.ndarray)->int:
    """Compute Youden's Statistics(Youden Index) of ROC curve.

    Args:
        fpr (np.ndarray): false positive rate(specificity)
        tpr (np.ndarray): true positive rate(sensitivity)

    Returns:
        int: Youden index
    """
    y = tpr - fpr
    return np.argmax(y)  # Only the first occurrence is returned.

def AUC(fpr:np.ndarray, tpr:np.ndarray)->float:
    """Calculate AUC of ROC_curve. Numerical scheme: Trapezoid Rule.

    Args:
        fpr (np.ndarray): false positive rate
        tpr (np.ndarray): true positive rate

    Returns:
        float: area under the curve
    """
    return -np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2)

def scan_auc_threshold(tdmi_data_flatten:np.ndarray, 
                       weight_flatten:np.ndarray, 
                       w_thresholds:list):
    log_tdmi_data = np.log10(tdmi_data_flatten)
    log_tdmi_range = [log_tdmi_data.min(), log_tdmi_data.max()]

    # compute ROC curves for different w_threshold values
    aucs = np.zeros_like(w_thresholds)
    roc_thresholds = np.linspace(*log_tdmi_range,100)
    for iidx, threshold in enumerate(w_thresholds):
        answer = weight_flatten.copy()
        answer = (answer>threshold).astype(bool)
        fpr, tpr = ROC_curve(answer, log_tdmi_data, roc_thresholds)
        aucs[iidx] = AUC(fpr, tpr)
    opt_threshold = roc_thresholds[Youden_Index(fpr, tpr)]
    return aucs, opt_threshold