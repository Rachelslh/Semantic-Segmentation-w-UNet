
import os

import tensorflow as tf
from omegaconf import OmegaConf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

from model import UNet_Model, infer
from data_loader import DataLoader


if __name__=="__main__":
    # Read from config
    config = OmegaConf.load("Unet/config.yaml")
    dataloader = DataLoader('Unet/data/images/', 'Unet/data/masks/', config.INPUT_SIZE)
    unet_architecture = UNet_Model(tuple(config.INPUT_SIZE), n_filters=32, n_classes=23, activation='relu', initializer='he_normal')
    model = unet_architecture.make_network()
    model.summary()
    
    checkpoint_path = "model.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)   
    model.compile(optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint",
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)

    train_dataset = dataloader.processed_image_ds.cache().shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
    model_history = model.fit(train_dataset, epochs=config.EPOCHS, callbacks=[cp_callback])
    
    plt.plot(model_history.history["accuracy"])
    
    # Plot predicted masks against the true mask
    infer(model, "checkpoint", train_dataset, 6)