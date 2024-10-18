# Number of clients
num_clients = 10
import numpy as np
import matplotlib.pyplot as plt
# Create a figure for the plots

def load_client_metrics(client_id):
    precision = np.load(f"client{client_id}_precision.npy")
    recall = np.load(f"client{client_id}_recall.npy")
    accuracy = np.load(f"client{client_id}_accuracy.npy")
    return precision, recall, accuracy
plt.figure(figsize=(13, 9))

# Plot Precision for all clients
plt.subplot(3, 1, 1)
for client_id in range(0, num_clients):
    precision, _, _ = load_client_metrics(client_id)
    plt.plot(precision, label=f"Client {client_id}")
server_precision=np.load("server_precision.npy")
plt.plot(server_precision,'-+',label='Server precision')
plt.title("Precision over Rounds")
plt.xlabel("Rounds")
plt.ylabel("Precision")
plt.grid()
plt.legend(fontsize='small')

# Plot Recall for all clients
plt.subplot(3, 1, 2)
for client_id in range(0, num_clients):
    _, recall, _ = load_client_metrics(client_id)
    plt.plot(recall, label=f"Client {client_id}")
server_recall=np.load("server_recall.npy")
plt.plot(server_recall,'-+',label='Server recall')
plt.title("Recall over Rounds")
plt.xlabel("Rounds")
plt.ylabel("Recall")
plt.grid()
plt.legend(fontsize='small')

# Plot Accuracy for all clients
plt.subplot(3, 1, 3)
for client_id in range(0, num_clients):
    _, _, accuracy = load_client_metrics(client_id)
    plt.plot(accuracy, label=f"Client {client_id}")
server_accuracy=np.load("server_accuracy.npy")
plt.plot(server_accuracy,'-+',label='Server accuracy')
plt.title("Accuracy over Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.grid()
plt.legend(fontsize='small')

# Show the plots
plt.tight_layout()
plt.show()