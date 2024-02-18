import os
from typing import Tuple

import tensorflow as tf


class DataLoader:
    def __init__(self, image_dir: str, masks_dir: str, input_size: Tuple[int, int, int]) -> None:
        image_filenames = os.listdir(image_dir)
        self.images = [image_dir+i for i in image_filenames]
        self.masks = [masks_dir+i for i in image_filenames]
        self.input_size = input_size
        
        self.split()
        self.preprocess()
        
    def split(self, ):
        image_filenames = tf.constant(self.images)
        masks_filenames = tf.constant(self.masks)

        self.dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
            

    def preprocess(self):
        image_ds = self.dataset.map(self.process_path)
        self.processed_image_ds = image_ds.map(self.preprocess_sample)
        for image, mask in self.processed_image_ds.take(1):
            print(image)
            print(mask)
    
    
    def process_path(self, image_path, mask_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=self.input_size[-1])
        img = tf.image.convert_image_dtype(img, tf.float32)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=self.input_size[-1])
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True) #TODO Understand this
        return img, mask
    
    
    def preprocess_sample(self, image, mask):
        input_image = tf.image.resize(image, self.input_size[:2], method='nearest')
        input_mask = tf.image.resize(mask, self.input_size[:2], method='nearest')

        return input_image, input_mask
    