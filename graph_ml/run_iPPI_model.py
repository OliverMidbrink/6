from iPPI_prediction_model import create_iPPI_prediction_model
from load_iPPI_data import train_generator

def main():
    model = create_iPPI_prediction_model()
    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    model.fit(train_generator())

if __name__ == "__main__":
    main()