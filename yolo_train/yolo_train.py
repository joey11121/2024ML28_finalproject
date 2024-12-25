from staff_detector import train_staff_detector

if __name__ == "__main__":
    # Configure training parameters
    data_yaml_path = "data.yaml"
    epochs = 300
    
    # Train the model
    print(f"Starting training with {epochs} epochs...")
    results = train_staff_detector(data_yaml_path, epochs)
    print("Training completed!")
    print(f"Best model saved at: {results.best}")
