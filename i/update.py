import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_update_model():
    # Create a Sequential model
    model = Sequential()
    
    # Add a Dense layer with 20,000 units
    # Assuming a linear activation function (default)
    model.add(Dense(20000, input_shape=(20000,)))

    return model

def main():
    # Get the model
    model = get_update_model()

    # Summary of the model to verify its structure
    model.summary()

if __name__ == "__main__":
    main()
