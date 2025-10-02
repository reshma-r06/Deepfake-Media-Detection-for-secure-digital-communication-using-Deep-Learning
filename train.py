import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

def build_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    # Load data
    X, y = load_features('audio_features.csv')
    print(f"Training data shape: {X.shape}")

    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, 'models/scaler.save')

    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print("Class weights:", class_weight_dict)

    # Train model with class weights
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, validation_split=0.2, class_weight=class_weight_dict)

    # Evaluate on test set
    y_probs = model.predict(X_test)
    # Try different thresholds
    for threshold in [0.5, 0.4, 0.6]:
        y_pred = (y_probs > threshold).astype(int)
        from sklearn.metrics import confusion_matrix, classification_report
        print(f"\nResults for threshold {threshold}:")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    # Show some sample predictions
    print("\nSample predictions (probability, predicted label, true label):")
    for i in range(min(10, len(y_test))):
        print(f"{y_probs[i][0]:.3f}\t{int(y_probs[i][0]>0.5)}\t{y_test[i]}")

    # Save test set for further analysis
    np.savez('models/test_set.npz', X_test=X_test, y_test=y_test, y_probs=y_probs)

    # Save model
    model.save('models/audio_model.h5')

if __name__ == '__main__':
    train()