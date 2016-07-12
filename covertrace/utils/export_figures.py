import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def save_subplots_in_pdf(values, row=6, col=6):
    """values could be either array or list of array"""
    with PdfPages('multipage.pdf') as pp:
        for num, plt_obj in enumerate(values):
            res = num % (row * col)
            if res == 0:
                plt.figure()
            if res == (row * col) - 1:
                pp.savefig()
            plt.subplot(row, col, res+1)
            _plot_subplots(plt_obj)


def _plot_subplots(plt_obj):
    if isinstance(plt_obj, np.ndarray):
        plt.plot(plt_obj)
    if isinstance(plt_obj, list) or isinstance(plt_obj, tuple):
        for i in plt_obj:
            plt.plot(i)
            plt.hold(True)
