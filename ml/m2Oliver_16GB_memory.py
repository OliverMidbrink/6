import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Add, GlobalAveragePooling3D, Multiply
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

        x = Conv3D(300, (10, 10, 10), strides=(2, 2, 2), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((4, 4, 4), padding='same')(x)

        x = Conv3D(200, (10, 10, 10), strides=(1, 1, 1), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((8, 8, 8), padding='same')(x)
        
        x = Conv3D(100, (10, 10, 10), strides=(2, 2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        
        return Model(inputs=input_layer, outputs=Flatten()(x))

    def _molecule_pipe(self, input_shape): # Molecule pip is basically disabled during testing 2023-11-16
        input_layer = Input(shape=input_shape)

        x = Conv3D(248, (5, 5, 5), strides=(1, 1, 1), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)

        x = Conv3D(100, (12, 12, 12), strides=(2, 2, 2), padding='same')(x)
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
        z = Dense(100)(merged)
        z = Activation('relu')(z)
        z = Dense(50)(z)
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