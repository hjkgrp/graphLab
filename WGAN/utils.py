import os
import argparse
import numpy as np
from PIL import Image
from keras import backend as K
from keras.layers.merge import _Merge


def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for
    display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class RandomWeightedAverage(_Merge):
    """
    Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated.
    """

    def __init__(self, BATCH_SIZE):
        _Merge.__init__(self)
        self.BATCH_SIZE = BATCH_SIZE

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def generate_images(generator_model, output_dir, epoch):
    """
    Feeds random seeds into the generator and tiles and saves the output to a PNG
    file.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    test_image_stack = generator_model.predict(np.random.rand(10, 100))
    test_image_stack = (test_image_stack * 127.5) + 127.5
    test_image_stack = np.squeeze(np.round(test_image_stack).astype(np.uint8))
    tiled_output = tile_images(test_image_stack)
    tiled_output = Image.fromarray(tiled_output, mode='L')  # L specifies greyscale
    outfile = os.path.join(output_dir, 'epoch_{}.png'.format(epoch))
    tiled_output.save(outfile)


def parsing():
    parser = argparse.ArgumentParser(description="Improved Wasserstein GAN "
                                                 "implementation for Keras.")
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Directory to output generated files to")
    args = parser.parse_args()
    return args
