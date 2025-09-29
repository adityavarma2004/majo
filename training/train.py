import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from models.cnn_crnn import get_model

# Constants
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
]

def load_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y

def prepare_data(X, y):
    """Split data into train, validation, and test sets."""
    # First split: (train + validation) and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Second split: train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def compute_class_weights(y):
    """Compute class weights for imbalanced dataset."""
    class_weights = {}
    total_samples = len(y)
    classes, counts = np.unique(y, return_counts=True)
    
    for cls, count in zip(classes, counts):
        class_weights[cls] = total_samples / (len(classes) * count)
    
    return class_weights

def plot_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training/training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('training/confusion_matrix.png')
    plt.close()

def main():
    # Load data
    X, y = load_data()
    print(f"Data loaded - X shape: {X.shape}, y shape: {y.shape}")
    
    # Prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(X, y)
    
    # Get model
    model = get_model(input_shape=X_train.shape[1:])
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Generate predictions
    y_pred = model.predict(X_test).argmax(axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    # Plot training history and confusion matrix
    plot_history(history)
    plot_confusion_matrix(y_test, y_pred)
    
    # Save model
    model.save('models/urban_sound_cnn_crnn.h5')
    print("\nModel saved to models/urban_sound_cnn_crnn.h5")

if __name__ == "__main__":
    main()