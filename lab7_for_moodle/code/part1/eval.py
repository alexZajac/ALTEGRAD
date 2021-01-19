"""
Learning on Sets - ALTEGRAD - Jan 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc': [], 'mae': []}, 'lstm': {'acc': [], 'mae': []}}

for i in range(len(cards)):
    print(f"Cardinality: {cards[i]}, i: {i}")
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):

        # Task 6

        ##################
        x_test_batch = torch.tensor(
            X_test[i][j:j+batch_size]).to(device, dtype=torch.long)
        y_pred_ds = deepsets(x_test_batch)
        y_pred_deepsets.append(y_pred_ds)
        y_pred_lstm_value = lstm(x_test_batch)
        y_pred_lstm.append(y_pred_lstm_value)
        ##################

    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()

    acc_deepsets = accuracy_score(
        y_test[i], y_pred_deepsets.round())  # your code here
    mae_deepsets = mean_absolute_error(
        y_test[i], y_pred_deepsets)  # your code here
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)

    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()

    acc_lstm = accuracy_score(y_test[i], y_pred_lstm.round())  # your code here
    mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm)  # your code here
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


# Task 7

##################
plt.plot(cards, results['deepsets']['acc'], label="Deepsets")
plt.plot(cards, results['lstm']['acc'], label="LSTM")
plt.legend()
plt.title(f"Accuracy of DeepSets & LSTM models")
plt.xlabel("Cardinalities")
plt.ylabel("Accuracy")
plt.show()
plt.savefig("deepsets_lstm_accuracy_plot.png")
##################
