import numpy as np

def pred_ints(model, X, y, percentile=95):
    err_down = []
    err_up = []
    preds = []
    for pred in model.estimators_:
        preds.append(pred.predict(X))
    preds = np.array(preds)
    for x in range(len(X)):
        err_down.append(np.percentile(preds[:,x], (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds[:,x], 100 - (100 - percentile) / 2.))
    correct = 0.

    for i, val in enumerate(y):
      if err_down[i] <= val and val <= err_up[i]:
        correct += 1
    return correct/len(y),err_down,err_up
