import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, AveragePooling3D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Add, GlobalAveragePooling3D, Multiply
from tensorflow.keras import regularizers
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam

class M2OliverModel16GBMemory(Model):
    def __init__(self, protein_input_shape=(4, 100, 100, 100), molecule_input_shape=(9, 30, 30, 30), num_classes=2):
        super(M2OliverModel16GBMemory, self).__init__()
        self.protein_input_shape = protein_input_shape
        self.molecule_input_shape = molecule_input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _protein_pipe(self, input_shape):
        input_layer = Input(shape=input_shape)

        x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(input_layer)

        x = Conv3D(1, (3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv3D(1, (3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return Model(inputs=input_layer, outputs=Flatten()(x))

    def _molecule_pipe(self, input_shape): # Molecule pip is basically disabled during testing 2023-11-16
        input_layer = Input(shape=input_shape)

        x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(input_layer)

        x = Conv3D(1, (2, 2, 2), strides=(1, 1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

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
        z = Dense(2)(merged)
        z = Activation('relu')(z)
        z = Dense(1)(z)
        z = Activation('relu')(z)

        # Output Layer
        output = Dense(self.num_classes, activation='sigmoid')(z)

        # Create model
        model = Model(inputs=[protein_pipe1.input, protein_pipe2.input, molecule_pipe.input], outputs=output)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def call(self, inputs):
        return self.model(inputs)

    def summary(self):
        return self.model.summary()