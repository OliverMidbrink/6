from keras.models import Model
from keras.layers import Input, Conv3D, AveragePooling3D, Flatten, Dense, concatenate, Dropout, BatchNormalization, Add
from keras.layers import GlobalAveragePooling3D, Multiply, ELU
from keras import regularizers
from keras.optimizers import Adam

class M3Model(Model):
    def __init__(self, protein_input_shape=(4, 100, 100, 100), molecule_input_shape=(9, 30, 30, 30), num_classes=2):
        super(M3Model, self).__init__()
        self.protein_input_shape = protein_input_shape
        self.molecule_input_shape = molecule_input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _improved_conv_block(self, inputs, num_filters, kernel_size=(3, 3, 3), stride=1):
        x = Conv3D(num_filters, kernel_size, strides=stride, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ELU()(x)
        return x

    def _residual_block(self, inputs, num_filters):
        x = self._improved_conv_block(inputs, num_filters)
        x = Conv3D(num_filters, (3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        
        # Shortcut connection
        shortcut = Conv3D(num_filters, (1, 1, 1), padding='same')(inputs)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = ELU()(x)
        return x

    def _attention_block(self, inputs):
        x = GlobalAveragePooling3D()(inputs)
        x = Dense(inputs.shape[-1], activation='relu')(x)
        x = Dense(inputs.shape[-1], activation='sigmoid')(x)
        return Multiply()([inputs, x])

    def _protein_pipe(self, input_shape):
        input_layer = Input(shape=input_shape)
        x = self._improved_conv_block(input_layer, 64)
        x = AveragePooling3D((2, 2, 2), padding='same')(x)
        x = self._residual_block(x, 128)
        x = AveragePooling3D((2, 2, 2), padding='same')(x)
        x = self._attention_block(x)
        x = self._residual_block(x, 256)
        x = AveragePooling3D((2, 2, 2), padding='same')(x)
        return Model(inputs=input_layer, outputs=Flatten()(x))

    def _molecule_pipe(self, input_shape):
        input_layer = Input(shape=input_shape)
        y = self._improved_conv_block(input_layer, 64)
        y = AveragePooling3D((2, 2, 2), padding='same')(y)
        y = self._residual_block(y, 128)
        y = AveragePooling3D((2, 2, 2), padding='same')(y)
        return Model(inputs=input_layer, outputs=Flatten()(y))

    def _build_model(self):
        # Protein pipelines
        protein_pipe1 = self._protein_pipe(self.protein_input_shape)
        protein_pipe2 = self._protein_pipe(self.protein_input_shape)

        # Molecule pipeline
        molecule_pipe = self._molecule_pipe(self.molecule_input_shape)

        # Merging pipelines
        merged = concatenate([protein_pipe1.output, protein_pipe2.output, molecule_pipe.output])

        # Deep Neural Network after merging
        z = Dense(1024, kernel_regularizer=regularizers.l2(0.0001))(merged)
        z = ELU()(z)
        z = Dropout(0.3)(z)
        z = Dense(512, kernel_regularizer=regularizers.l2(0.0001))(z)
        z = ELU()(z)
        z = Dropout(0.3)(z)
        z = Dense(256, kernel_regularizer=regularizers.l2(0.0001))(z)
        z = ELU()(z)

        # Output Layer
        output = Dense(self.num_classes, activation='sigmoid')(z)

        # Create model
        model = Model(inputs=[protein_pipe1.input, protein_pipe2.input, molecule_pipe.input], outputs=output)

        # Compile the model
        model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def call(self, inputs):
        return self.model(inputs)

    def summary(self):
        return self.model.summary()
