import os
import numpy as np
import matplotlib.pyplot as plt

def load_all_corrmats():
    """load all correlation matrices and return them in dictionary with timestamps as keys."""
    # get list of all correlation matrices
    corr_path = f"{os.getcwd()}/data/corr_matrices/"
    file_list = os.listdir(corr_path)

    # load all correlation matrices
    corrmats = {}  # dictionary of correlation matrices with timestamp as key
    for i in file_list:
        name = i[:-4]
        corrmats[name] = np.genfromtxt(corr_path+i, delimiter=",")

    return corrmats

def load_corrmat_tickers():
    """load the ticker symbols of the full correlation matrices. returned as list."""
    with open(f"{os.getcwd()}/data/corr_matrix_tickers.txt") as f:
        tickers = f.read()
    tickers = tickers.split("\n")[:-1]
    return tickers

def notnull_corrmat_and_tickers(corrmat, tickers):
    """for a correlation matrix return the correlation matrix without NaN values and return the list of tickers in order of the columns. return (corrmat, tickers)"""
    # check which values are NaN and remove those
    notnan_mask = ~np.isnan(corrmat)

    # look at first row of the mask to subset tickers
    tickers = np.array(tickers)
    tickers_notnan = tickers[notnan_mask[0,:]]
    dim = len(tickers_notnan)

    # get corrmat for values that are not nan
    corrmat_notnan = corrmat[notnan_mask]
    corrmat_notnan = np.reshape(corrmat_notnan, (dim, dim))

    return corrmat_notnan, tickers_notnan

def print_corrmats(corrmats, n_cols=4):
    T = len(corrmats.keys())

    # dimensions of figure/plot grid
    n_cols = 4
    n_rows = int(np.ceil(T/n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3), constrained_layout=True)
    for cnt,i in enumerate(sorted(corrmats.keys())):
        # determine axis
        col = cnt % n_cols
        row = cnt // n_cols
        ax = axs[row, col]

        # plot correlation matrix
        ax.imshow(corrmats[i])
        ax.set_title(i)
        fig.suptitle("Correlation matrices over time")