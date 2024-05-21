
import os

import tensorflow as tf
from omegaconf import OmegaConf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

from src.model import UNet_Model, infer
from src.configs.data.data_loader import DataLoader


if __name__=="__main__":
    # Read from config
    config = OmegaConf.load("src/configs/config.yaml")
    dataloader = DataLoader('data/images/', 'data/masks/', config.INPUT_SIZE)
    unet_architecture = UNet_Model(tuple(config.INPUT_SIZE), n_filters=32, n_classes=23, activation='relu', initializer='he_normal')
    model = unet_architecture.make_network()
    model.summary()
    
    checkpoint_path = "model.ckpt"
    model.compile(optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)

    train_dataset = dataloader.train_dataset.cache().shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
    val_dataset = dataloader.val_dataset.cache().shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
    model_history = model.fit(train_dataset, validation_data=val_dataset, epochs=config.EPOCHS, callbacks=[cp_callback])
    
    plt.plot(model_history.history["accuracy"])
    
    # Plot predicted masks against the true mask
    infer(model, checkpoint_path, dataloader.test_dataset.batch(config.BATCH_SIZE), 6)