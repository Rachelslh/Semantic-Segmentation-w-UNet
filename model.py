from typing import List

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, concatenate
from tensorflow.keras import Input, Model

import numpy as np

from utils import plot, create_mask

class UNet_Model:
    def __init__(self, input_size, n_filters, n_classes, activation, initializer) -> None:
        self.input_size = input_size
        self.n_filters = n_filters
        self.n_classes = n_classes
        
        self.activation = activation
        self.initializer = initializer
    
    
    def make_network(self, ):
        inputs = Input(shape=self.input_size)
        
        downsampled_activations = self.encode(inputs)
        upsampled_activations = self.decode(downsampled_activations)

        #conv9 = Conv2D(self.n_filters,
        #    3,
        #    activation=self.activation,
        #    padding='same',
        #    kernel_initializer=self.initializer)(upsampled_activations)

        conv10 = Conv2D(self.n_classes, 1, padding='same')(upsampled_activations)
        
        model = Model(inputs=inputs, outputs=conv10)
        
        return model
    
    
    def encode(self, inputs):
        # Contracting Path (encoding)
        cblock1 = self.conv_block(inputs, self.n_filters)
        cblock2 = self.conv_block(cblock1[0], self.n_filters * 2)
        cblock3 = self.conv_block(cblock2[0], self.n_filters * 4)
        cblock4 = self.conv_block(cblock3[0], self.n_filters * 8, dropout_prob=0.3)
        cblock5 = self.conv_block(cblock4[0], self.n_filters * 16, dropout_prob=0.3, max_pooling=False) 
        
        return cblock1, cblock2, cblock3, cblock4, cblock5
        
    def decode(self, inputs):
        cblock1, cblock2, cblock3, cblock4, cblock5 = inputs
        # Expanding Path (decoding)
        # Use the cblock5[0] as expansive_input and cblck4[1] as contractive_input and n_filters * 8
        dblock1 = self.deconv_block(cblock5[0], cblock4[1],  self.n_filters * 8)
        # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
        # Note that you must use the second element of the contractive block i.e before the maxpooling layer. 
        # At each step, use half the number of filters of the previous block 
        dblock2 = self.deconv_block(dblock1, cblock3[1],  self.n_filters * 4)
        dblock3 = self.deconv_block(dblock2, cblock2[1],  self.n_filters * 2)
        dblock4 = self.deconv_block(dblock3, cblock1[1],  self.n_filters)
        
        return dblock4

    def conv_block(self, inputs, n_filters, dropout_prob=0, max_pooling=True):
        conv = Conv2D(filters=n_filters, # Number of filters
            kernel_size=3,   # Kernel size   
            activation=self.activation,
            padding='same',
            kernel_initializer=self.initializer)(inputs)
        conv = Conv2D(n_filters,
            3,
            activation=self.activation,
            padding='same',
            kernel_initializer=self.initializer)(conv)
        
        # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
        if dropout_prob > 0:
            conv = Dropout(dropout_prob)(conv)
            
        # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
        if max_pooling:
            activations = MaxPooling2D(pool_size=(2, 2))(conv) # stride=None defaults to pool size
            
        else:
            activations = conv
            
        skip_connection = conv
        
        return activations, skip_connection, 
        
    
    def deconv_block(self, expansive_input, contractive_input, n_filters):
        up = Conv2DTranspose(
                 n_filters,    # number of filters
                 3,    # Kernel size
                 strides=2,
                 padding='same')(expansive_input)
    
        # Merge the previous output and the contractive_input
        merge = concatenate([up, contractive_input], axis=3)
        conv = Conv2D(n_filters,
                    3,
                    activation=self.activation,
                    padding='same',
                    kernel_initializer=self.initializer)(merge)
        conv = Conv2D(n_filters,
                    3,
                    activation=self.activation,
                    padding='same',
                    kernel_initializer=self.initializer)(conv)
        
        return conv
    
    
def infer(model: UNet_Model, checkpoint_path: str, dataset: List[np.array], num: int = 1):
    # Loads the weights
    model.load_weights(checkpoint_path)
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            plot([image[0], mask[0], create_mask(pred_mask)])
