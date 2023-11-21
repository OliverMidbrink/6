from iPPI_prediction_model import create_iPPI_prediction_model
from load_iPPI_data import iPPIDataset
from spektral.data import DisjointLoader
from tensorflow.keras.callbacks import ModelCheckpoint

def main():
    model = create_iPPI_prediction_model()
    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    train_dataset = iPPIDataset(mode="train")
    val_dataset = iPPIDataset(mode="val")

    train_loader = DisjointLoader(train_dataset, batch_size=4, epochs=100)
    val_loader = DisjointLoader(val_dataset, batch_size=4)

    checkpoint_path = "graph_ml/checkpoint_iPPI_model.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model.fit(
        train_loader.load(),
        validation_data=val_loader.load(),
        validation_steps=val_loader.steps_per_epoch,
        steps_per_epoch=train_loader.steps_per_epoch,
        epochs=100,
        callbacks=[checkpoint] 
    )

if __name__ == "__main__":
    main()