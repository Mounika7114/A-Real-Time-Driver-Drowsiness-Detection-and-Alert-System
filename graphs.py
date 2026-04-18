import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 16))

# Training values from your logs
train_accuracy = [0.8693, 0.9622, 0.9730, 0.9774, 0.9874, 0.9884, 0.9901, 0.9924, 0.9931, 0.9932, 0.9956, 0.9966, 0.9931, 0.9937, 0.9976]
val_accuracy   = [0.5432, 0.5342, 0.5878, 0.9286, 0.9598, 0.8348, 0.9643, 0.9241, 0.9568, 0.9955, 0.9167, 0.7128, 0.9643, 0.9836, 0.9747]

train_loss = [0.4089, 0.1063, 0.0703, 0.0657, 0.0403, 0.0366, 0.0274, 0.0277, 0.0253, 0.0199, 0.0150, 0.0156, 0.0241, 0.0191, 0.0079]
val_loss   = [0.6230, 1.9475, 1.8781, 0.1412, 0.0975, 0.4692, 0.0961, 0.2222, 0.1128, 0.0119, 0.2515, 1.0128, 0.1343, 0.0578, 0.0537]

# Accuracy plot
plt.figure(figsize=(12,5))
plt.plot(epochs, train_accuracy, 'o-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 's-', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# Loss plot
plt.figure(figsize=(12,5))
plt.plot(epochs, train_loss, 'o-', label='Training Loss')
plt.plot(epochs, val_loss, 's-', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()
