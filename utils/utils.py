import numpy as np
import matplotlib.pyplot as plt

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


def plot_predictions(pred_test_xgb, pred_test_rf, pred_test_dtr, pred_test_lreg, y_test):
    fig, axs = plt.subplots(2, 2, figsize=(15,12))

    axs[0, 0].scatter(pred_test_xgb, y_test, label='xgboost')
    axs[0, 0].plot([-1,10],[-1,10],'r-')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(-1, 7.5)
    axs[0, 0].set_xlim(-1, 7.5)
    axs[0, 0].set_xlabel('predicted prices')
    axs[0, 0].set_ylabel('true prices')

    axs[0, 1].scatter(pred_test_rf, y_test, label='random forest')
    axs[0, 1].plot([-1,10],[-1,10],'r-')
    axs[0, 1].legend()
    axs[0, 1].set_ylim(-1, 7.5)
    axs[0, 1].set_xlim(-1, 7.5)
    axs[0, 1].set_xlabel('predicted prices')
    axs[0, 1].set_ylabel('true prices')

    axs[1, 0].scatter(pred_test_dtr, y_test, label='decision tree regressor')
    axs[1, 0].plot([-1,10],[-1,10],'r-')
    axs[1, 0].legend()
    axs[1, 0].set_ylim(-1, 7.5)
    axs[1, 0].set_xlim(-1, 7.5)
    axs[1, 0].set_xlabel('predicted prices')
    axs[1, 0].set_ylabel('true prices')

    axs[1, 1].scatter(pred_test_lreg, y_test, label='linear regression')
    axs[1, 1].plot([-1,10],[-1,10],'r-')
    axs[1, 1].legend()
    axs[1, 1].set_ylim(-1, 7.5)
    axs[1, 1].set_xlim(-1, 7.5)
    axs[1, 1].set_xlabel('predicted prices')
    axs[1, 1].set_ylabel('true prices')

    plt.title('')
    plt.show()
