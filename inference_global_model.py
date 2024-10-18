import flwr as fl
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score


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

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # Change to the number of classes
    ])
    return model

data_dir = r"C:\Users\Abdussalam\Desktop\final_package\final_package\test_set" # Replace with your dataset path
data, labels = load_data(data_dir)
server_accuracy=[]
server_precision=[]
server_recall=[]
for idx in range(1,101):
    saved_round_path= f"C:\\Users\\Abdussalam\\Desktop\\final_package\\final_package\\Weight_FL_server\\round-{idx}-weights.npz"
    npzfile= np.load(saved_round_path)
    model= create_model()

    conv_weights = npzfile['arr_0']
    conv_biases = npzfile['arr_1']
    dense1_weights = npzfile['arr_2']
    dense1_biases = npzfile['arr_3']
    dense2_weights = npzfile['arr_4']
    dense2_biases = npzfile['arr_5']

    # Set the weights manually to the model layers
    model.layers[0].set_weights([conv_weights, conv_biases])  # Conv2D layer
    model.layers[3].set_weights([dense1_weights, dense1_biases])  # First Dense layer
    model.layers[4].set_weights([dense2_weights, dense2_biases])  # Output Dense layer

    val_logits = model(data, training=False)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = loss_fn(labels, val_logits).numpy()
    precision = precision_score(labels, np.argmax(val_logits, axis=1), average='binary')
    recall = recall_score(labels, np.argmax(val_logits, axis=1), average='binary')
    accuracy = np.mean(np.argmax(val_logits, axis=1) == labels)

    server_precision.append(precision)
    server_accuracy.append(accuracy)
    server_recall.append(recall)
    
    np.save("server_precision.npy", np.array(server_precision))
    np.save("server_recall.npy", np.array(server_recall))
    np.save("server_accuracy.npy", np.array(server_accuracy))
