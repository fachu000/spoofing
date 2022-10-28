import pickle
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import scipy.signal
import os
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

data_folder = 'generators/data/'


class PowerMeasurementGenerator:

    @staticmethod
    def get_data_220805e_shortened():
        """See get_features for the format of `t_power`. 
        
        This data has no averaging. By shortened, it means that only the first
        *5000 samples are kept from each time series."""

        def set_tx_positions():
            Npoint = 52
            points = np.zeros([Npoint, 2])
            points[0, :] = np.array([3.97, 2.04])
            points[1, :] = np.array([3.67, 2.07])
            points[2, :] = np.array([3.37, 2.11])
            points[3, :] = np.array([3.07, 2.14])
            points[4, :] = np.array([2.78, 2.17])
            points[5, :] = np.array([2.48, 2.21])
            points[6, :] = np.array([2.18, 2.24])
            points[7, :] = np.array([1.88, 2.27])
            points[8, :] = np.array([1.58, 2.31])
            points[9, :] = np.array([1.28, 2.34])
            points[10, :] = np.array([0.99, 2.37])
            points[11, :] = np.array([0.69, 2.40])
            points[12, :] = np.array([0.39, 2.44])
            points[13, :] = np.array([4.38, 4.54])
            points[14, :] = np.array([4.08, 4.57])
            points[15, :] = np.array([3.78, 4.60])
            points[16, :] = np.array([3.48, 4.63])
            points[17, :] = np.array([3.19, 4.66])
            points[18, :] = np.array([2.89, 4.70])
            points[19, :] = np.array([2.59, 4.73])
            points[20, :] = np.array([2.29, 4.76])
            points[21, :] = np.array([1.99, 4.79])
            points[22, :] = np.array([1.69, 4.82])
            points[23, :] = np.array([1.40, 4.85])
            points[24, :] = np.array([1.10, 4.88])
            points[25, :] = np.array([0.80, 4.91])
            points[26, :] = np.array([4.05, 2.53])
            points[27, :] = np.array([3.75, 2.56])
            points[28, :] = np.array([3.45, 2.60])
            points[29, :] = np.array([3.15, 2.63])
            points[30, :] = np.array([2.86, 2.67])
            points[31, :] = np.array([2.56, 2.70])
            points[32, :] = np.array([2.26, 2.73])
            points[33, :] = np.array([1.96, 2.77])
            points[34, :] = np.array([1.66, 2.80])
            points[35, :] = np.array([1.36, 2.84])
            points[36, :] = np.array([1.07, 2.87])
            points[37, :] = np.array([0.77, 2.90])
            points[38, :] = np.array([0.47, 2.94])
            points[39, :] = np.array([3.98, 4.06])
            points[40, :] = np.array([3.68, 4.09])
            points[41, :] = np.array([3.38, 4.12])
            points[42, :] = np.array([3.08, 4.15])
            points[43, :] = np.array([2.78, 4.18])
            points[44, :] = np.array([2.48, 4.21])
            points[45, :] = np.array([2.18, 4.24])
            points[46, :] = np.array([1.87, 4.27])
            points[47, :] = np.array([1.57, 4.31])
            points[48, :] = np.array([1.27, 4.34])
            points[49, :] = np.array([0.97, 4.37])
            points[50, :] = np.array([0.67, 4.40])
            points[51, :] = np.array([0.37, 4.43])
            return points

        def set_rec_positions():
            Nrec = 4
            rec = np.zeros([Nrec, 3])
            rec[0, :] = ([0.627, 2.019, 4])
            rec[1, :] = ([-2.849, 5.545, 4])
            rec[2, :] = ([0.767, 6.43, 4])
            rec[3, :] = ([8.343, 4.842, 4])
            return rec

        fi_name = data_folder + 'signal_powers_220805e-shortened.pickle'

        with open(fi_name, 'rb') as file:
            t_power = pickle.load(file)

        # Replace infinities with outliers
        t_power[abs(t_power) == np.inf] = 1

        m_loc_tx = set_tx_positions()
        m_loc_rx = set_rec_positions()
        v_inds_non_testable = []
        return t_power, m_loc_tx, m_loc_rx, v_inds_non_testable

    @staticmethod
    def filter_time_series(t_power, filter_lengths):
        """
        `t_power` is num_tx x num_feat x num_samples

        `filter_lengths` is a vector with num_feat entries or integer. 
        """
        num_tx, num_feat, _ = t_power.shape
        t_power_natural = 10**(t_power / 10)
        if type(filter_lengths) in [int, np.int64]:
            filter_lengths = np.tile(filter_lengths, (num_feat, ))

        ll_power = [[
            scipy.signal.convolve(t_power_natural[ind_tx, ind_feat, :],
                                  (1 / filter_lengths[ind_feat]) * np.ones(
                                      (filter_lengths[ind_feat], )),
                                  mode="same") for ind_feat in range(num_feat)
        ] for ind_tx in range(num_tx)]
        t_power = 10 * np.log10(np.array(ll_power))

        # Remove beginning and end
        max_filter_length = np.max(filter_lengths, )
        t_power = t_power[..., max_filter_length:-max_filter_length]

        return t_power

        # True if tx point location v_loc is not excluded from testable points
        if m_loc_tx_non_testable.size == 0:
            return True
        return not (0 in np.linalg.norm(
            v_loc[0:2] - m_loc_tx_non_testable[:, 0:2], axis=1))

    @staticmethod
    def plot_room(m_loc_tx, m_loc_rx, axis=None):
        # MARKER_LIST = [
        #     "$0$", "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$9$",
        #     "$10$", "$11$"
        # ]
        MARKER_LIST = [f"${ind}$" for ind in range(len(m_loc_tx))]
        MARKER_LIST_Rec = ["$R1$", "$R2$", "$R3$", "$R4$", "$R5$"]
        import matplotlib as mpl
        tableMain = mpl.patches.Rectangle([0.622, 2.285],
                                          3,
                                          2.2,
                                          -6,
                                          alpha=0.3,
                                          color='0.7')  # based on point 2
        tableKevin = mpl.patches.Rectangle([7.133, 1.731],
                                           4.5,
                                           2.2,
                                           90,
                                           alpha=0.3,
                                           color='0.7')  # based on point 0
        tableWall = mpl.patches.Rectangle([1.034, 7.614],
                                          2,
                                          1,
                                          0,
                                          alpha=0.3,
                                          color='0.7')  # based on point 5
        roomwall = np.array([[0, 0], [11, 0], [11, 6], [5., 8.5], [-4, 8.5],
                             [-4, 6], [-2., 6], [-2, 0.5], [0, 0.5], [0, 0]])

        if axis is None:
            plt.figure()
            axis = plt.subplot(111)
            plt.axis('scaled')
        axis.add_patch(tableMain)
        axis.add_patch(tableKevin)
        axis.add_patch(tableWall)
        plt.plot(roomwall[:, 0], roomwall[:, 1], 'b')
        plt.plot(m_loc_tx[:, 0], m_loc_tx[:, 1], 'b*')
        plt.plot(m_loc_rx[:, 0], m_loc_rx[:, 1], 'rx')
        plt.ylabel('y [m]'), plt.xlabel('x [m]')
        plt.axis('equal')
        for kk in range(0, len(m_loc_rx)):
            plt.scatter(m_loc_rx[kk, 0] + 0.4,
                        m_loc_rx[kk, 1],
                        marker=MARKER_LIST_Rec[kk],
                        s=200,
                        c='r')
        # for kk in range(0, len(m_loc_tx)):
        #     plt.scatter(m_loc_tx[kk, 0] + 0.4,
        #                 m_loc_tx[kk, 1],
        #                 marker=MARKER_LIST[kk],
        #                 s=100,
        #                 c='b')
        plt.xlim([-5, 13])
        plt.ylim([-5, 13])