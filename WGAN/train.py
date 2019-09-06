import os
import numpy as np
import pickle
from functools import partial
from PIL import Image
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from model import make_generator, make_discriminator
from loss import wloss, gradient_penalty_wloss
from utils import tile_images, generate_images, parsing, RandomWeightedAverage
from prepare_data import get_mnist


def main():
    args = parsing()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    BATCH_SIZE = 64
    TRAINING_RATIO = 5
    GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
    X_train = get_mnist()

    generator = make_generator()
    discriminator = make_discriminator()

    # The generator_model is used when we want to train the generator layers.
    # As such, we ensure that the discriminator layers are not trainable.
    # Note that once we compile this model, updating .trainable will have no effect within
    # it. As such, it won't cause problems if we later set discriminator.trainable = True
    # for the discriminator_model, as long as we compile the generator_model first.
    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False
    generator_input = Input(shape=(100,))
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input],
                            outputs=[discriminator_layers_for_generator])
    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=wloss)

    # Now that the generator_model is compiled, we can make the discriminator
    # layers trainable.
    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    # The discriminator_model is more complex. It takes both real image samples and random
    # noise seeds as input. The noise seed is run through the generator model to get
    # generated images. Both real and generated images are then run through the
    # discriminator. Although we could concatenate the real and generated images into a
    # single tensor, we don't (see model compilation for why).
    real_samples = Input(shape=X_train.shape[1:])
    generator_input_for_discriminator = Input(shape=(100,))
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    # We also need to generate weighted-averages of real and generated samples,
    # to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage(BATCH_SIZE)([real_samples,
                                                          generated_samples_for_discriminator])
    # We then run these samples through the discriminator as well. Note that we never
    # really use the discriminator output for these samples - we're only running them to
    # get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator(averaged_samples)

    # The gradient penalty loss function requires the input averaged samples to get
    # gradients. However, Keras loss functions can only have two arguments, y_true and
    # y_pred. We get around this by making a partial() of the function with the averaged
    # samples here.
    partial_gp_loss = partial(gradient_penalty_wloss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = 'gradient_penalty'

    # Keras requires that inputs and outputs have the same number of samples. This is why
    # we didn't concatenate the real samples and generated samples before passing them to
    # the discriminator: If we had, it would create an output with 2 * BATCH_SIZE samples,
    # while the output of the "averaged" samples for gradient penalty
    # would have only BATCH_SIZE samples.

    # If we don't concatenate the real and generated samples, however, we get three
    # outputs: One of the generated samples, one of the real samples, and one of the
    # averaged samples, all of size BATCH_SIZE. This works neatly!
    discriminator_model = Model(inputs=[real_samples,
                                        generator_input_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
    # the real and generated samples, and the gradient penalty loss for the averaged samples
    discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                loss=[wloss,
                                      wloss,
                                      partial_gp_loss])
    # We make three label vectors for training. positive_y is the label vector for real
    # samples, with value 1. negative_y is the label vector for generated samples, with
    # value -1. The dummy_y vector is passed to the gradient_penalty loss function and
    # is not used.
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    d_loss = []
    discriminator_loss = []
    generator_loss = []
    print('Training...')
    for epoch in range(100):
        print("---epoch: %d---" % epoch)
        np.random.shuffle(X_train)
        print("Epoch: ", epoch)
        print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
        minibatches_size = BATCH_SIZE * TRAINING_RATIO
        for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
            print("batch: ", i)
            discriminator_minibatches = X_train[i * minibatches_size:
                                                (i + 1) * minibatches_size]
            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * BATCH_SIZE:
                                                        (j + 1) * BATCH_SIZE]
                noise = np.random.rand(BATCH_SIZE, 100).astype(np.float32)
                discriminator_loss.append(discriminator_model.train_on_batch(
                    [image_batch, noise],
                    [positive_y, negative_y, dummy_y]))
            d_loss.append(np.mean(np.array(discriminator_loss[-5:]), axis=0))
            generator_loss.append(generator_model.train_on_batch(np.random.rand(BATCH_SIZE, 100),
                                                                 positive_y))
        generate_images(generator, args.output_dir, epoch)
        loss_dict = {"g_loss": generator_loss, "d_loss": d_loss}
        with open("loss.pkl", "wb") as fo:
            pickle.dump(loss_dict, fo)


if __name__ == '__main__':
    main()
