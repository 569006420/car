from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



class KerasPilot:

    def load(self, model_path):
        self.model = load_model(model_path)

    def shutdown(self):
        pass

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):

        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=steps * (1.0 - train_split) / train_split)
        return hist


class KerasLinear(KerasPilot):
    def __init__(self, model=None, num_outputs=None, *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = default_linear()
        else:
            self.model = default_linear()

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        # print(len(outputs), outputs)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


def default_linear():
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in

    # Convolution2D class name is an alias for Conv2D
    x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu')(x)
    x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='elu')(x)
    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='elu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='elu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='elu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    x = Dense(units=50, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    # categorical output of the angle
    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)
    sess = tf.Session()
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    Tensorboard = TensorBoard(log_dir="/home/wei/parts/logs", histogram_freq=1, write_grads=True)
    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})
    with tf.name_scope('train'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=angle_out,
                                                              logits=logits)
        loss = tf.reduce_mean(xentropy,name='loss')
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)

 #   tf.summary.scalar("angle_out", angle_out )
 #   tf.summary.scalar("throttle_out", throttle_out )
    tf.summary.scalar("loss", loss )
    merge_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/home/wei/mycar/logs", sess.graph)
    for i in range(100):
        train_summary = sess.run(merge_summary, feed_dict= {loss:i+1})
        writer.add_summary(train_summary, i)
    #history=model.fit(steps_per_epoch=steps_per_epoch, callbacks=[Tensorboard])


    return model
