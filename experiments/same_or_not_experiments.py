import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import scipy.signal
import os
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from generators.power_measurement_generator import PowerMeasurementGenerator
from inference_blocks.inference_blocks import DnnSameOrNotClassifier, DistSameOrNotClassifier, InferenceBlock, KMeansSameOrNotClassifier
from simulators.simulator import Simulator

import gsim
from gsim.gfigure import GFigure

gsim.rs = RandomState(MT19937(SeedSequence(123456789)))


def get_icassp_data():
    t_power, m_loc_tx, m_loc_rx, v_inds_non_testable = PowerMeasurementGenerator.get_data_220805e_shortened(
    )
    t_power = t_power[:, :, 80:]
    t_power = PowerMeasurementGenerator.filter_time_series(t_power, 16)
    return t_power, m_loc_tx, m_loc_rx, v_inds_non_testable


class ExperimentSet(gsim.AbstractExperimentSet):

    def experiment_1001(l_args):

        t_power, m_loc_tx, m_loc_rx, v_inds_non_testable = get_icassp_data()

        lf_classifiers = [
            lambda: DnnSameOrNotClassifier(
                num_epochs=350, verbosity=0, run_eagerly=False),
            lambda: DistSameOrNotClassifier(verbosity=0),
            lambda: DistSameOrNotClassifier(norm="l1"),
            lambda: KMeansSameOrNotClassifier(num_clusters=15, verbosity=0),
        ]

        l_num_tx_train = [10, 15, 20, 25, 35, 45]

        # One row per entry of `l_num_tx_train`
        m_accuracy = np.array([
            Simulator.analyze_same_or_not_classifiers_montecarlo(
                t_power,
                lf_classifiers=lf_classifiers,
                num_pairs_train=500 * 5,
                num_pairs_val=300,
                num_pairs_test=300,
                num_tx_train=num_tx_train,
                num_mc_iter=100) for num_tx_train in l_num_tx_train
        ])

        print(f'm_accuracy = {m_accuracy}')

        G = GFigure(xaxis=l_num_tx_train,
                    yaxis=m_accuracy.T,
                    xlabel="Number of training positions",
                    ylabel="Accuracy",
                    title="",
                    legend=[cf().__str__() for cf in lf_classifiers])

        return G

    def experiment_1002(l_args):

        t_power, m_loc_tx, m_loc_rx, v_inds_non_testable = get_icassp_data()

        lf_classifiers = [
            lambda: DnnSameOrNotClassifier(
                num_epochs=350, verbosity=0, run_eagerly=False),
            lambda: DistSameOrNotClassifier(verbosity=0),
            lambda: DistSameOrNotClassifier(norm="l1"),
            lambda: KMeansSameOrNotClassifier(num_clusters=15, verbosity=0),
        ]

        l_num_feat = [1, 2, 4, 7, 9]

        lv_accuracy = []
        ind_start = 0

        for ind_iter in range(ind_start, len(l_num_feat)):
            num_feat = l_num_feat[ind_iter]
            print(f"-------- num_feat = {num_feat} --------")
            v_accuracy = Simulator.analyze_same_or_not_classifiers_montecarlo(
                t_power[:, :num_feat, :],
                lf_classifiers=lf_classifiers,
                num_pairs_train=500 * 5,
                num_pairs_val=300,
                num_pairs_test=300,
                num_tx_train=40,
                num_mc_iter=10)
            lv_accuracy.append(v_accuracy)

        m_accuracy = np.array(lv_accuracy)

        print(f'm_accuracy = {m_accuracy}')

        G = GFigure(xaxis=l_num_feat,
                    yaxis=m_accuracy.T,
                    xlabel="Number of features",
                    ylabel="Accuracy",
                    title="",
                    legend=[cf().__str__() for cf in lf_classifiers])

        return G
