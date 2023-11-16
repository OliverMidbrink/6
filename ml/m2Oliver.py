import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Add, GlobalAveragePooling3D, Multiply
from tensorflow.keras import regularizers
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

class M2OliverModel(Model):
    def __init__(self, protein_input_shape=(4, 100, 100, 100), molecule_input_shape=(9, 30, 30, 30), num_classes=2):
        super(M2OliverModel, self).__init__()
        self.protein_input_shape = protein_input_shape
        self.molecule_input_shape = molecule_input_shape
        self.num_classes = num_classes
        self.model, self.callbacks = self._build_model()

    def _protein_pipe(self, input_shape):
        input_layer = Input(shape=input_shape)
        x = Conv3D(4096, (30, 30, 30), strides=(2, 2, 2), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((8, 8, 8), padding='same')(x)
        x = Conv3D(2048, (20, 20, 20), strides=(2, 2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        return Model(inputs=input_layer, outputs=Flatten()(x))

    def _molecule_pipe(self, input_shape): # Molecule pip is basically disabled during testing 2023-11-16
        input_layer = Input(shape=input_shape)
        x = Conv3D(1024, (20, 20, 20), strides=(10, 10, 10), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((8, 8, 8), padding='same')(x)
        x = Conv3D(num_filters, (12, 12, 12), strides=(2, 2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        return Model(inputs=input_layer, outputs=Flatten()(x))

    def _build_model(self):
        # Protein pipelines
        protein_pipe1 = self._protein_pipe(self.protein_input_shape)
        protein_pipe2 = self._protein_pipe(self.protein_input_shape)

        # Molecule pipeline
        molecule_pipe = self._molecule_pipe(self.molecule_input_shape)

        # Merging pipelines
        merged = Concatenate()([protein_pipe1.output, protein_pipe2.output, molecule_pipe.output])

        # Deep Neural Network after merging
        z = Dense(16384)(merged)
        z = Activation('relu')(z)
        z = Dense(8192)(z)
        z = Activation('relu')(z)
        z = Dense(4096)(z)
        z = Activation('relu')(z)

        # Output Layer
        output = Dense(self.num_classes, activation='sigmoid')(z)

        # Create model
        model = Model(inputs=[protein_pipe1.input, protein_pipe2.input, molecule_pipe.input], outputs=output)
    
    def call(self, inputs):
        return self.model(inputs)

    def summary(self):
        return self.model.summary()