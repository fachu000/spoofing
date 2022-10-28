import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import scipy.signal
import os
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from sklearn.cluster import KMeans

import gsim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

# Imports modified to please pylint
#from tensorflow.keras.layers import Dense, Flatten, Conv2D
#from tensorflow.keras import Model
from keras.api._v2.keras.layers import Dense, Flatten, Conv2D
from keras.api._v2.keras import Model
import keras.api._v2.keras as keras
from tqdm.keras import TqdmCallback

print("TensorFlow version:", tf.__version__)


class InferenceBlock:

    pass


class SameOrNotClassifier(InferenceBlock):

    def __init__(self, verbosity=0):
        self.verbosity = verbosity

    @staticmethod
    def build_dataset(t_power, num_entries):
        # It returns a tensor t_dataset of shape (num_entries, 2, num_feat) and
        # a vector v_same of length num_entries. Each entry of t_dataset is a
        # matrix of the form:
        #
        # [ t_power[ind_tx_1, :, ind_sample_1]; t_power[ind_tx_2, :,
        # ind_sample_2 ]
        #
        # if ind_tx_1 == ind_tx_2, then the corresponding entry of v_same is
        # True, else it is False.
        #
        # v_same contains num_entries/2 entries = True and num_entries/2 entries
        # = False
        #

        def draw_no_replacement(high):
            # Returns two different uniformly distributed random integers
            # between 0 and high-1.
            ind_1 = gsim.rs.randint(high)
            ind_2 = gsim.rs.randint(high - 1)
            if ind_2 >= ind_1:
                ind_2 += 1
            return ind_1, ind_2

        num_entries_each_class = int(np.floor(num_entries / 2))

        num_tx, _, num_samples = t_power.shape
        lv_feat_pairs = []
        l_same = []
        for ind_entry in range(num_entries):

            if ind_entry < num_entries_each_class:
                ind_tx_1, ind_tx_2 = draw_no_replacement(num_tx)
                same = False
            else:
                ind_tx_1 = gsim.rs.randint(num_tx)
                ind_tx_2 = ind_tx_1
                same = True
            ind_sample_1, ind_sample_2 = gsim.rs.randint(num_samples,
                                                         size=(2, ))
            t_feat_pairs = np.stack(
                (t_power[ind_tx_1, :, ind_sample_1], t_power[ind_tx_2, :,
                                                             ind_sample_2]))
            lv_feat_pairs.append(t_feat_pairs)
            l_same.append(same)

        # Shuffle
        t_feat_pairs = np.array(lv_feat_pairs)
        v_same = np.array(l_same)
        v_ind = gsim.rs.permutation(len(lv_feat_pairs))
        return t_feat_pairs[v_ind], v_same[v_ind]

    def train(self, t_power, num_pairs_train, num_pairs_val, val_split=0.2):
        """ num_pairs_train training examples are constructed from a fraction
        (1-validation_split) of the outer slices of t_power. Conversely, num_pairs_val
        validation examples are constructed from the remaining fraction
        validation_split of the outer slices of t_power.
        
        """

        num_tx_val = int(np.floor(val_split * len(t_power)))
        if self.verbosity >= 2:
            print(
                f'Using {len(t_power)-num_tx_val} tx. positions for actual training and {num_tx_val} for validation.'
            )
        if num_tx_val == 0:
            raise ValueError
        v_ind = gsim.rs.permutation(len(t_power))
        t_power_train = t_power[v_ind[num_tx_val:]]
        t_power_val = t_power[v_ind[:num_tx_val]]

        t_feat_pairs_train, v_same_train = self.build_dataset(
            t_power_train, num_pairs_train)
        t_feat_pairs_val, v_same_val = self.build_dataset(
            t_power_val, num_pairs_val)

        self._train(t_feat_pairs_train=t_feat_pairs_train,
                    v_same_train=v_same_train,
                    t_feat_pairs_val=t_feat_pairs_val,
                    v_same_val=v_same_val)


class DnnSameOrNotClassifier(SameOrNotClassifier):

    class SymmetricNet(Model):

        learning_rate = 1e-4  #0.00005
        activation = 'leaky_relu'
        neurons_per_layer = 512

        def __init__(self):
            super().__init__()
            self.flatten = Flatten()
            self.d1 = Dense(self.neurons_per_layer,
                            activation=self.activation,
                            kernel_regularizer=tf.keras.regularizers.L1(.14))
            self.d2 = Dense(self.neurons_per_layer, activation=self.activation)
            self.d3 = Dense(self.neurons_per_layer, activation=self.activation)
            self.d4 = Dense(self.neurons_per_layer, activation=self.activation)
            self.d5 = Dense(self.neurons_per_layer, activation=self.activation)
            self.d6 = Dense(self.neurons_per_layer, activation=self.activation)
            self.d7 = Dense(self.neurons_per_layer, activation=self.activation)
            self.dout = Dense(1)

        def call(self, x):
            x_reversed = tf.reverse(x, [1])
            return (self.asymmetric_call(x) +
                    self.asymmetric_call(x_reversed)) / 2

        def asymmetric_call(self, x):
            # v_ means batch of vectors

            if True:
                # debug
                #batch_size = x.shape[0]
                v_x_vec = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2],
                                               1))  #batch of xbec
                vec_x_vec_x_t = v_x_vec @ tf.transpose(
                    v_x_vec, (0, 2, 1))  # outer product
                v_feat_1 = tf.reshape(
                    vec_x_vec_x_t,
                    (-1, vec_x_vec_x_t.shape[1] * vec_x_vec_x_t.shape[2]))

                v_feat_2 = x[:, 0, :] - x[:, 1, :]

                v_feat_3 = tf.reshape(x, (-1, x.shape[1] * x.shape[2]))

                #x = tf.concat((v_feat_1, v_feat_2, v_feat_3), axis=1)
                x = tf.concat((v_feat_2, v_feat_3), axis=1)  #
                #x = tf.concat((v_feat_1, v_feat_3), axis=1) # 0.98
            else:
                x = self.flatten(x)

            x = self.d1(x)
            x = self.d2(x)
            x = self.d3(x)
            #
            # x = self.d4(x)
            # x = self.d5(x)
            # x = self.d6(x)
            return self.dout(x)

    class CallbackStop(tf.keras.callbacks.Callback):
        min_accuracy = 0.97

        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_accuracy') > self.min_accuracy):
                # print(
                #     f"\nReached {self.min_accuracy} accuracy, so stopping training!!"
                # )
                self.model.stop_training = True

    def __init__(self, num_epochs, run_eagerly=False, **kwargs):

        super().__init__(**kwargs)
        self.model = self.SymmetricNet()
        self.num_epochs = num_epochs
        self.run_eagerly = run_eagerly

    def __str__(self):
        return "DNNC"

    def _train(self, t_feat_pairs_train, v_same_train, t_feat_pairs_val,
               v_same_val):

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.model.learning_rate),
            #loss='binary_crossentropy')
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=keras.metrics.BinaryAccuracy(name='accuracy',
                                                 threshold=0.))

        self.model.run_eagerly = self.run_eagerly
        history = self.model.fit(t_feat_pairs_train,
                                 v_same_train.astype(int),
                                 epochs=self.num_epochs,
                                 verbose=0,
                                 validation_data=(t_feat_pairs_val,
                                                  v_same_val.astype(int)),
                                 batch_size=32,
                                 callbacks=[
                                     TqdmCallback(verbose=0),
                                     tf.keras.callbacks.EarlyStopping(
                                         monitor='val_accuracy',
                                         patience=30,
                                         restore_best_weights=True,
                                     ),
                                     DnnSameOrNotClassifier.CallbackStop()
                                 ])

        def plot_loss(history):
            plt.subplot(211)
            plt.plot(history.history['loss'], '--', label='loss')
            plt.plot(history.history['val_loss'], '-', label='val_loss')
            #plt.ylim([0, 10])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            print(
                f"{len(t_feat_pairs_train)} training examples, {len(t_feat_pairs_val)} validation examples"
            )
            plt.grid(True)

            plt.subplot(212)
            plt.plot(history.history['accuracy'], '--', label='accuracy')
            plt.plot(history.history['val_accuracy'],
                     '-',
                     label='val_accuracy')
            plt.ylim([0, 1.1])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

        if self.verbosity:
            plot_loss(history)

            plt.figure()
            v_weights = np.sort(
                np.abs(self.model.d1.weights[0].numpy().ravel()))
            plt.hist(v_weights, bins=40)
            plt.title("Histogram of the weights of the first layer")
            print("abs(weights 1st layer) = ", v_weights)
            return

    def are_the_same(self, t_feat_pairs):
        # t_feat_pairs is num_pairs x 2 x num_feat returns a vector v_same of
        # length num_pairs where v_same[i] is True if t_feat_pairs[i][0] and
        # t_feat_pairs[i][1] are determined to belong to the same transmitter
        return np.ravel(self.model(t_feat_pairs).numpy() > 0)


class DistSameOrNotClassifier(SameOrNotClassifier):
    """ Distance-based classifier"""

    thresh = 1

    def __init__(self,
                 num_thresh=30,
                 feature_selection=None,
                 norm='l2',
                 **kwargs):
        """
        `num_thresh` is the number of thresholds to try when training

        `feature_selection` can be either None or a vector of indices between
        and num_feat - 1. If not None, then only the features with indices in
        that vector are used. 

        `norm` can be "l1" or "l2". 
        """
        super().__init__(**kwargs)
        self.num_thresh = num_thresh
        self.feature_selection = feature_selection
        self.norm = norm

    def __str__(self):
        return f"DBC({self.norm})"

    def _train(self,
               t_feat_pairs_train,
               v_same_train,
               t_feat_pairs_val=None,
               v_same_val=None):

        def accuracy_for_thresh(thresh, v_statistic, v_same):
            return np.mean(
                v_same == self._statistic_to_class(v_statistic, thresh))

        v_statistic_train = self._reduce_statistic(t_feat_pairs_train)
        v_thresh = np.linspace(np.min(v_statistic_train),
                               np.max(v_statistic_train), self.num_thresh)

        v_accuracy = [
            accuracy_for_thresh(thresh, v_statistic_train, v_same_train)
            for thresh in v_thresh
        ]

        ind_max = np.argmax(v_accuracy)

        self.thresh = v_thresh[ind_max]

        if t_feat_pairs_val is not None:
            accuracy_train = v_accuracy[ind_max]
            accuracy_val = accuracy_for_thresh(
                self.thresh, self._reduce_statistic(t_feat_pairs_val),
                v_same_val)
            if self.verbosity:
                print(
                    f"Training of {self} completed. Training accuracy = {accuracy_train}, val accuracy = {accuracy_val}"
                )

    def _reduce_statistic(self, t_feat):
        return np.array([self._statistic(m_feat) for m_feat in t_feat])

    def _statistic(self, m_feats):
        """`m_feats` is a 2 x num_feat matrix"""

        if self.feature_selection is not None:
            m_feats = m_feats[:, self.feature_selection]

        v_feat_diff = m_feats[0] - m_feats[1]
        if self.norm == "l2":
            return np.linalg.norm(v_feat_diff)
        elif self.norm == "l1":
            return np.sum(abs(v_feat_diff))
        else:
            raise ValueError

    @staticmethod
    def _statistic_to_class(v_statistic, thresh):
        return v_statistic < thresh

    def are_the_same(self, t_feat_pairs):
        v_statistic = self._reduce_statistic(t_feat_pairs)
        return self._statistic_to_class(v_statistic, self.thresh)


class KMeansSameOrNotClassifier(DistSameOrNotClassifier):

    num_clusters = None

    def __init__(self, num_clusters=None, num_samples_to_cluster=50, **kwargs):
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.num_samples_to_cluster = num_samples_to_cluster

    def __str__(self):
        return "KMC"

    def train(self, t_power, num_pairs_train, num_pairs_val, val_split=0.2):
        """ 
        Each column of each outer slice of t_power is a data vector. 
        
        A fraction (1-validation_split) of the outer slices of t_power give rise
        to the training data vectors. 

        Kmeans is used to cluster `num_samples_to_cluster` training data vectors
        selected uniformly at random into num_clusters clusters. 

        A feature vector comprises the num_clusters distances between a given
        data vector and all the centroids. 

        Two data vectors are decided to come from the same location if the
        distance between these feature vectors is below `thresh`.
        
        num_pairs_train pairs of feature vectors are constructed from the
        training data vectors. Similarly, with num_pairs_val for validation
        pairs. 

        The threshold is adjusted from the training pairs. 

        't_power' is num_tx x num_channels * num_rx  x num_samples
                
        """

        # Obtain data vectors
        num_tx_val = int(np.floor(val_split * len(t_power)))
        if self.verbosity >= 1:
            print(
                f'Using {len(t_power)-num_tx_val} tx. positions for actual training and {num_tx_val} for validation.'
            )
        if num_tx_val == 0:
            raise ValueError
        v_ind = gsim.rs.permutation(len(t_power))
        t_power_train = t_power[v_ind[num_tx_val:]]
        t_power_val = t_power[v_ind[:num_tx_val]]

        num_tx_train, num_feat, num_samp = t_power_train.shape
        # the following is num_samp * num_tx_train x num_feat
        m_data_train = np.reshape(
            np.transpose(t_power_train, (0, 2, 1)),
            (num_tx_train * num_samp, num_feat))  # each row is a data vector

        # Cluster data vectors
        v_ind = gsim.rs.permutation(num_tx_train * num_samp)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(
            m_data_train[v_ind[:self.num_samples_to_cluster]])
        self.m_centroids = kmeans.cluster_centers_  # self.num_clusters x num_feat

        # Training and validation pairs of DATA vectors
        t_data_pairs_train, v_same_train = self.build_dataset(
            t_power_train, num_pairs_train)
        t_data_pairs_val, v_same_val = self.build_dataset(
            t_power_val, num_pairs_val)

        # To feature vector pairs
        t_feat_pairs_train = np.array(
            [self._data_to_feat_pair(pair) for pair in t_data_pairs_train])
        t_feat_pairs_val = np.array(
            [self._data_to_feat_pair(pair) for pair in t_data_pairs_val])

        self._train(t_feat_pairs_train=t_feat_pairs_train,
                    v_same_train=v_same_train,
                    t_feat_pairs_val=t_feat_pairs_val,
                    v_same_val=v_same_val)

    def _data_to_feat_pair(self, m_data_pair):
        """"
        `m_data_pair` is 2 x num_ch * num_rx

        It returns a 2 x `self.num_centroids` matrix.
        """

        return np.array([
            self._data_to_feat_vec(m_data_pair[0]),
            self._data_to_feat_vec(m_data_pair[1]),
        ])

    def _data_to_feat_vec(self, v_data):
        """
            `v_data` is of shape (num_ch * num_rx,)

            Returns

                vector of length self.num_centroids where the i-th entry
                contains the distance from `v_data` to the i-th centroid.
        """
        return np.array([
            np.linalg.norm(v_data - v_centroid)
            for v_centroid in self.m_centroids
        ])