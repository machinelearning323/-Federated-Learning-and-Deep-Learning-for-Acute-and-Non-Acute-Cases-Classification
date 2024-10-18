import flwr as fl
import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score

# Load dataset from the specified directory
def load_data(data_dir):
    data = []
    labels = []
    class_names = os.listdir(data_dir)

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32, 32))  # Resize image
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            data.append(img_array)
            labels.append(class_index)

    return np.array(data), np.array(labels)


# Partition the dataset for multiple clients
def partition_data(data, labels, num_clients):
    partition_size = len(data) // num_clients
    partitions = []

    for i in range(num_clients):
        start = i * partition_size
        end = start + partition_size if i != num_clients - 1 else len(data)  # Last partition takes the remainder
        client_data = data[start:end]
        client_labels = labels[start:end]
        partitions.append((client_data, client_labels))

    return partitions
# Shuffle data and labels together
def shuffle_data(data, labels):
    data, labels = shuffle(data, labels, random_state=42)  # Shuffle with a fixed seed for reproducibility
    return data, labels

# Create a simple CNN model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Change to the number of classes
    ])
    return model




# Custom training loop using TQDM for progress visualization
def train_model_custom_loop(model, train_data, train_labels, val_data, val_labels, epochs=1, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
    
    # Optimizer and loss function
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # Track the training process
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Create a progress bar for this epoch
        prog_bar = tqdm(dataset, desc="Training", unit="batch")

        for step, (x_batch, y_batch) in enumerate(prog_bar):
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, logits)
            
            # Compute gradients and update weights
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            # Update progress bar
            prog_bar.set_postfix({"loss": loss_value.numpy()})

        # Validation step
        val_logits = model(val_data, training=False)
        val_loss = loss_fn(val_labels, val_logits)
        val_accuracy = np.mean(np.argmax(val_logits, axis=1) == val_labels)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")


# Flower client for federated learning
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_data, train_labels, val_data, val_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.model = create_model()

        # Initialize arrays to store precision, recall, and accuracy for each round
        self.precision_history = []
        self.recall_history = []
        self.accuracy_history = []

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # Train the model with custom loop and show progress
        train_model_custom_loop(self.model, self.train_data, self.train_labels, self.val_data, self.val_labels, epochs=1)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # Evaluate on the validation data
        val_logits = self.model(self.val_data, training=False)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = loss_fn(self.val_labels, val_logits).numpy()
        print(loss)
        precision = precision_score(self.val_labels, np.argmax(val_logits, axis=1), average='binary')
        recall = recall_score(self.val_labels, np.argmax(val_logits, axis=1), average='binary')
        accuracy = np.mean(np.argmax(val_logits, axis=1) == self.val_labels)

        # Save metrics to history arrays
        self.precision_history.append(precision)
        self.recall_history.append(recall)
        self.accuracy_history.append(accuracy)
        return float(loss), int(len(self.val_data)), {"accuracy": accuracy}
    def save_metrics(self):
        # Save precision, recall, and accuracy history to .npy files
        np.save("client0_precision.npy", np.array(self.precision_history))
        np.save("client0_recall.npy", np.array(self.recall_history))
        np.save("client0_accuracy.npy", np.array(self.accuracy_history))

        print(f"Metrics saved for client 0")

import threading
from sklearn.model_selection import train_test_split

def start_flower_client(client_id, partitions):
    # Split the partitioned data into training and validation sets for the client
    train_data, val_data, train_labels, val_labels = train_test_split(partitions[client_id][0], partitions[client_id][1], test_size=0.2)

    # Create and run the Flower client
    client = FlowerClient(train_data, train_labels, val_data, val_labels)

    # Start Flower client (replace the port with the one your server is running on)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

# Set up the Flower clients and server#
if __name__ == "__main__":
    # Load the full dataset
    data_dir = r"C:\Users\Abdussalam\Desktop\final_package\final_package\Annotated_stroke_and_non_stroke_Dataset"  # Replace with your dataset path
    data, labels = load_data(data_dir)
    # Shuffle the dataset before partitioning
    data, labels = shuffle_data(data, labels)
    # Partition the dataset into 10 parts for 10 clients
    num_clients = 10
    partitions = partition_data(data, labels, num_clients)

    # For each client, split its data into training and validation sets
    client_id = 0  # Change this for each client
    train_data, val_data, train_labels, val_labels = train_test_split(partitions[client_id][0], partitions[client_id][1], test_size=0.2)

    # Create and run the Flower client
    client = FlowerClient(train_data, train_labels, val_data, val_labels)

    # Start Flower client (replace the port with the one your server is running on)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client,grpc_max_message_length = 1024*1024*1024)
    client.save_metrics()

