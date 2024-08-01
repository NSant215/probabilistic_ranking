import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.linalg.basic import pinv2


def sorted_barplot(P, W):
    """
    Function for making a sorted bar plot based on values in P, and labelling the plot with the
    corresponding names
    :param P: An array of length num_players (107)
    :param W: Array containing names of each player
    :return: None
    """
    M = len(P)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(20, 20))
    sorted_indices = np.argsort(P)
    sorted_names = W[sorted_indices]
    plt.barh(xx, P[sorted_indices])
    plt.yticks(np.linspace(0, M, M), labels=sorted_names[:, 0])
    plt.ylim([-2, 109])
    plt.show()

def barplots(P1, P2, P3, W):
    M = len(P1)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(20, 20))
    sorted_indices1 = np.argsort(P1)
    sorted_indices2 = np.argsort(P2)
    sorted_indices3 = np.argsort(P3)
    sorted_names1 = W[sorted_indices1]
    sorted_names2 = W[sorted_indices2]
    sorted_names3 = W[sorted_indices3]

    gs = gridspec.GridSpec(1, 3)

    fig = plt.figure(figsize=(10, 30), dpi=80)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.barh(xx, P1[sorted_indices1], color = 'r')
    ax1.set_yticks(np.linspace(0, M, M), labels=sorted_names1[:, 0])
    ax1.ylim([-2, 109])
    ax1.set_title('(a) Empirical Game Outcomes')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.barh(xx, P2[sorted_indices2], color = 'g')
    ax2.yticks(np.linspace(0, M, M), labels=sorted_names2[:, 0])
    ax2.ylim([-2, 109])
    ax2.set_title('(b) Gibbs Sampling')

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.barh(xx, P2[sorted_indices3], color = 'b')
    ax3.yticks(np.linspace(0, M, M), labels=sorted_names3[:, 0])
    ax3.ylim([-2, 109])
    ax3.set_title('(c) Message Passing)')

    plt.show()