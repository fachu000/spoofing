import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import scipy.signal
import os
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import gsim
from inference_blocks.inference_blocks import SameOrNotClassifier

global rs


class Simulator:

    # Analyze different tx classifier
    @staticmethod
    def analyze_same_or_not_classifiers(t_power,
                                        l_classifiers,
                                        num_pairs_train,
                                        num_pairs_val,
                                        num_pairs_test,
                                        test_split=None,
                                        num_tx_train=None,
                                        val_split=0.2,
                                        verbosity=0):
        """

        - t_power is num_tx x num_rx*num_ch x num_samples

        - `test_split` indicates the fraction of transmitters used for testing.
          `num_tx_train` indicates the number of transmitters for training. Only
          one of these two must be provided.

        - `num_pairs_train` training and `num_pairs_val` validation pairs are
          constructed with pairs of transmitters selected for training. 

        - `num_pairs_test` testing pairs are constructed with pairs of
          transmitters selected for training. 
            
        Returns: 

            `l_accuracy`: list where the n-th entry contains the accuracy of
            classifier `l_classifiers[n]`.

        """

        def lprint(*args, level=1):
            if verbosity >= level:
                print(*args)

        if test_split is not None and num_tx_train is not None:
            raise ValueError

        v_ind = gsim.rs.permutation(len(t_power))
        if test_split is not None:
            num_tx_test = int(np.floor(test_split * len(t_power)))
        else:
            assert num_tx_train is not None
            assert num_tx_train < len(v_ind)
            num_tx_test = len(v_ind) - num_tx_train

        lprint(
            f'Using {len(v_ind)-num_tx_test} tx locations for training and {num_tx_test} for testing.'
        )
        v_ind_test = v_ind[:num_tx_test]
        v_ind_train = v_ind[num_tx_test:]

        t_power_train = t_power[v_ind_train]
        t_power_test = t_power[v_ind_test]

        t_feat_pairs_test, v_same_test = SameOrNotClassifier.build_dataset(
            t_power_test, num_pairs_test)

        l_accuracy = []
        for classifier in l_classifiers:

            lprint(f"----- Analyzing classifier {classifier} ---------")

            # Accuracy before training
            v_same_predicted = classifier.are_the_same(t_feat_pairs_test)
            accuracy = np.mean(v_same_test == v_same_predicted)
            lprint(f'Test accuracy before training = {accuracy}')

            # Training
            classifier.train(t_power_train,
                             num_pairs_train=num_pairs_train,
                             num_pairs_val=num_pairs_val,
                             val_split=val_split)

            # Accuracy after training
            v_same_predicted = classifier.are_the_same(t_feat_pairs_test)
            accuracy = np.mean(v_same_test == v_same_predicted)
            lprint(
                f'Test accuracy after training ({num_tx_test} test tx.) = {accuracy}'
            )

            l_accuracy.append(accuracy)

        return l_accuracy

    @staticmethod
    def analyze_same_or_not_classifiers_montecarlo(t_power,
                                                   lf_classifiers,
                                                   *args,
                                                   num_mc_iter=1,
                                                   **kwargs):
        """ lf_classifiers is a list whose n-th entry is a function that returns
        a classifier. This is used to instantiate new classifiers for each MC
        iteration."""

        v_sum_accuracy = np.zeros((len(lf_classifiers), ))

        for _ in range(num_mc_iter):
            l_classifiers = [f_classifier() for f_classifier in lf_classifiers]
            l_accuracy = Simulator.analyze_same_or_not_classifiers(
                t_power, l_classifiers, *args, **kwargs)
            v_sum_accuracy += np.array(l_accuracy)

        return v_sum_accuracy / num_mc_iter
