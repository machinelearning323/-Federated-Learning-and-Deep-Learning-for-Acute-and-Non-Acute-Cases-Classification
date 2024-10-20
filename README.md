Federated-Learning-and-Deep-Learning-for-Acute-and-Non-Acute-Cases-Classification
A simple neural network model was employed for classifying acute and non-acute stroke cases, designed to handle binary classification based on medical imaging data.
The federated learning approach was implemented using the FLOWER framework, which enables decentralized training across multiple clients while maintaining privacy by avoiding raw data sharing.
 In the experiment, a virtual federated learning environment was created, consisting of 10 clients and a central server. Each client trained the neural network model on its local data, sending only the model updates (gradients) to the server.
The server aggregated the model updates from all 10 clients to update the global model, ensuring that the learned features from different clients contributed to improving the classification performance.
The model's performance was validated using standard metrics, including precision, accuracy, F1-score, and recall. These metrics helped assess the model's ability to correctly classify stroke cases.
