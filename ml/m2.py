import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Add, GlobalAveragePooling3D, Multiply
from tensorflow.keras import regularizers
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

class M2Model(Model):
    def __init__(self, protein_input_shape=(4, 100, 100, 100), molecule_input_shape=(9, 30, 30, 30), num_classes=2):
        super(ProteinMolInteractionModel, self).__init__()
        self.protein_input_shape = protein_input_shape
        self.molecule_input_shape = molecule_input_shape
        self.num_classes = num_classes
        self.model, self.callbacks = self._build_model()

    def _conv_block(self, inputs, num_filters):
        x = Conv3D(num_filters, (3, 3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def _residual_block(self, inputs, num_filters):
        x = self._conv_block(inputs, num_filters)
        x = Conv3D(num_filters, (3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        
        # Shortcut connection
        shortcut = Conv3D(num_filters, (1, 1, 1), padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    def _attention_block(self, inputs):
        x = GlobalAveragePooling3D()(inputs)
        x = Dense(inputs.shape[-1], activation='relu')(x)
        x = Dense(inputs.shape[-1], activation='sigmoid')(x)
        return Multiply()([inputs, x])

    def _protein_pipe(self, input_shape):
        input_layer = Input(shape=input_shape)
        x = self._conv_block(input_layer, 64)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = self._residual_block(x, 128)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = self._attention_block(x)
        x = self._residual_block(x, 256)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        return Model(inputs=input_layer, outputs=Flatten()(x))

    def _molecule_pipe(self, input_shape):
        input_layer = Input(shape=input_shape)
        y = self._conv_block(input_layer, 64)
        y = MaxPooling3D((2, 2, 2), padding='same')(y)
        y = self._residual_block(y, 128)
        y = MaxPooling3D((2, 2, 2), padding='same')(y)
        return Model(inputs=input_layer, outputs=Flatten()(y))

    def _build_model(self):
        # Protein pipelines
        protein_pipe1 = self._protein_pipe(self.protein_input_shape)
        protein_pipe2 = self._protein_pipe(self.protein_input_shape)

        # Molecule pipeline
        molecule_pipe = self._molecule_pipe(self.molecule_input_shape)

        # Merging pipelines
        merged = Concatenate()([protein_pipe1.output, protein_pipe2.output, molecule_pipe.output])

        # Deep Neural Network after merging
        z = Dense(1024, kernel_regularizer=regularizers.l2(0.001))(merged)
        z = Activation('relu')(z)
        z = Dropout(0.5)(z)
        z = Dense(512, kernel_regularizer=regularizers.l2(0.001))(z)
        z = Activation('relu')(z)
        z = Dropout(0.5)(z)
        z = Dense(256, kernel_regularizer=regularizers.l2(0.001))(z)
        z = Activation('relu')(z)

        # Output Layer
        output = Dense(self.num_classes, activation='sigmoid')(z)

        # Create model
        model = Model(inputs=[protein_pipe1.input, protein_pipe2.input, molecule_pipe.input], outputs=output)

        # Compile the model with Adam optimizer and learning rate schedule
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Setup callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)

        return model, [reduce_lr, checkpoint]
    
    def call(self, inputs):
        return self.model(inputs)

    def summary(self):
        return self.model.summary()

# Usage
protein_input_shape = (4, 100, 100, 100)
molecule_input_shape = (9, 30, 30, 30)
num_classes = 2