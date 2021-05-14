import numpy as np
import matplotlib.pyplot as plt

def fig_frame33(data:dict, function:callable)->plt.Figure:
    """Generated integrated figure.

    Args:
        data (dict): dict of dict, each nested dict containing data for ploting single ax.
        function (callable): plot funciton for single ax.

    Returns:
        plt.Figure: matplotlib.figure.Figure
    """
    # create figure canvas
    fig = plt.figure(figsize=(15,15), dpi=100)
    gs = fig.add_gridspec(nrows=3, ncols=3, 
                          left=0.05, right=0.95, top=0.96, bottom=0.05, 
                          wspace=0.25, hspace=0.35)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for i, key in enumerate(data.keys()):
        ax[i] = function(ax = ax[i], **data[key])
    xlabel = ax[0].get_xlabel()
    ylabel = ax[0].get_ylabel()
    [axi.set_xlabel('') for axi in ax]
    [axi.set_ylabel('') for axi in ax]
    [ax[i].set_xlabel(xlabel) for i in (6,7,8)]
    [ax[i].set_ylabel(ylabel) for i in (0,3,6)]
    return fig

def fig_frame25(data:dict, function:callable)->plt.Figure:
    """Generated integrated figure.

    Args:
        data (dict): dict of dict, each nested dict containing data for ploting single ax.
        function (callable): plot funciton for single ax.

    Returns:
        plt.Figure: matplotlib.figure.Figure
    """
    # create figure canvas
    fig = plt.figure(figsize=(20,9), dpi=100)
    gs = fig.add_gridspec(nrows=2, ncols=5, 
                          left=0.05, right=0.95, top=0.90, bottom=0.10, 
                          wspace=0.25, hspace=0.30)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for i, key in enumerate(data.keys()):
        ax[i] = function(ax = ax[i], **data[key])
    for i in np.arange(len(data.keys()), len(ax)):
        ax[i].axis('off')
    xlabel = ax[0].get_xlabel()
    ylabel = ax[0].get_ylabel()
    [axi.set_xlabel('') for axi in ax]
    [axi.set_ylabel('') for axi in ax]
    [ax[i].set_xlabel(xlabel) for i in (5,6,7,8,9)]
    [ax[i].set_ylabel(ylabel) for i in (0,5)]
    return fig
    
def fig_frame52(data:dict, function:callable)->plt.Figure:
    """Generated integrated figure.

    Args:
        data (dict): dict of dict, each nested dict containing data for ploting single ax.
        function (callable): plot funciton for single ax.

    Returns:
        plt.Figure: matplotlib.figure.Figure
    """
    # create figure canvas
    fig = plt.figure(figsize=(8,20), dpi=100)
    gs = fig.add_gridspec(nrows=5, ncols=2, 
                          left=0.10, right=0.95, top=0.95, bottom=0.05, 
                          wspace=0.20, hspace=0.25)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for i, key in enumerate(data.keys()):
        ax[i] = function(ax = ax[i], **data[key])
    for i in np.arange(len(data.keys()), len(ax)):
        ax[i].axis('off')
    xlabel = ax[0].get_xlabel()
    ylabel = ax[0].get_ylabel()
    [axi.set_xlabel('') for axi in ax]
    [axi.set_ylabel('') for axi in ax]
    [ax[i].set_xlabel(xlabel) for i in (8,9)]
    [ax[i].set_ylabel(ylabel) for i in (0,2,4,6,8)]
    return fig

def fig_frame24(data:dict, function:callable)->plt.Figure:
    """Generated integrated figure.

    Args:
        data (dict): dict of dict, each nested dict containing data for ploting single ax.
        function (callable): plot funciton for single ax.

    Returns:
        plt.Figure: matplotlib.figure.Figure
    """
    # create figure canvas
    fig = plt.figure(figsize=(18,9), dpi=100)
    gs = fig.add_gridspec(nrows=2, ncols=4, 
                          left=0.05, right=0.95, top=0.90, bottom=0.10, 
                          wspace=0.25, hspace=0.30)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for i, key in enumerate(data.keys()):
        ax[i] = function(ax = ax[i], **data[key])
    xlabel = ax[0].get_xlabel()
    ylabel = ax[0].get_ylabel()
    [axi.set_xlabel('') for axi in ax]
    [axi.set_ylabel('') for axi in ax]
    [ax[i].set_xlabel(xlabel) for i in (4,5,6,7)]
    [ax[i].set_ylabel(ylabel) for i in (0,4)]
    return fig

def fig_frame42(data:dict, function:callable)->plt.Figure:
    """Generated integrated figure.

    Args:
        data (dict): dict of dict, each nested dict containing data for ploting single ax.
        function (callable): plot funciton for single ax.

    Returns:
        plt.Figure: matplotlib.figure.Figure
    """
    # create figure canvas
    fig = plt.figure(figsize=(9,18), dpi=100)
    gs = fig.add_gridspec(nrows=4, ncols=2, 
                          left=0.10, right=0.95, top=0.95, bottom=0.05, 
                          wspace=0.20, hspace=0.30)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for i, key in enumerate(data.keys()):
        ax[i] = function(ax = ax[i], **data[key])
    xlabel = ax[0].get_xlabel()
    ylabel = ax[0].get_ylabel()
    [axi.set_xlabel('') for axi in ax]
    [axi.set_ylabel('') for axi in ax]
    [ax[i].set_xlabel(xlabel) for i in (6,7)]
    [ax[i].set_ylabel(ylabel) for i in (0,2,4,6)]
    return fig

def plot_union(data:dict, function:callable):
    filter_pool = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma', 'sub_delta', 'above_delta']
    fig = plt.figure(figsize=(18,6), dpi=200)
    # plot banded
    gs = fig.add_gridspec(nrows=2, ncols=4, 
                        left=0.28, right=0.98, top=0.92, bottom=0.08, 
                        wspace=0.25, hspace=0.25)
    ax = np.array([fig.add_subplot(i) for i in gs])
    for idx, band in enumerate(filter_pool):
        ax[idx] = function(ax[idx], **data[band])
        ax[idx].set_title(band)
    [ax[i].set_xlabel('') for i in (0,1,2,3)]

    # plot raw
    gs_origin = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.24,
                                top=0.69, bottom=0.31) 
    ax = fig.add_subplot(gs_origin[0])
    ax = function(ax, **data['raw'])
    ax.set_title('raw')

    # handle common legends
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        if len(handles) <= 4:
            ncol = 1
        else:
            ncol = 2
        ax.legend(handles, labels, loc='center', bbox_to_anchor=(-0.1, -1.0, 1., 1.), fontsize=9, ncol=ncol)
    return fig
