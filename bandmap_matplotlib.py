import gc
from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QVBoxLayout, QWidget
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams["axes.axisbelow"] = True
matplotlib.rcParams['font.size'] = 8.0
matplotlib.rcParams['axes.titlesize'] = 8.0
import numpy as np


class bandmap_new_plotwindow(QMainWindow):
    """
    THIS CLASS IS INHERITED FROM QMAINWINDOW THEREFORE THE INSTNACES CREATED BY THIS CLASS MANGAGE THEMSELVES, BUT NOT BY THE MOZI MAINWINDOW
    make plot in pop-up window via matplotlib
    minorTicksNumber = 1~10
    """

    def __init__(self,
                 data_2D,
                 k_array,
                 KE_array,
                 E_F,
                 level_min,
                 level_max,
                 figSize,
                 axisLimit,
                 XtickStep=1.0,
                 YtickStep=1.0,
                 XminorTicks=4,
                 YminorTicks=4,
                 showColorBar=False,
                 userLUT=None,
                 plotMode='BE only',
                 reverseSign_k=False,
                 meV=False,
                 parent=None):
        super().__init__()
        self.parent=parent
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setWindowTitle('Plot...')
        _main = QWidget()
        self.setCentralWidget(_main)
        sizeInInch = [x / 25.4 for x in figSize]

        fig = FigureCanvas(Figure(figsize=sizeInInch, dpi=100))
        fig.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QVBoxLayout(_main)
        layout.addWidget(fig)
        self.addToolBar(NavigationToolbar2QT(fig, self))
        x_pixel = len(k_array)
        if reverseSign_k:
            # reverse k sign because often forward emission is on the negative k side
            x_min = k_array[-1]
            x_max = k_array[0]
        else:
            x_min = k_array[0]
            x_max = k_array[-1]
        y_pixel = len(KE_array)
        if plotMode == 'KE only':
            y_min = KE_array[0]
            y_max = KE_array[-1]
        elif plotMode == 'BE only' or plotMode == 'BE & KE':
            y_min = E_F - KE_array[0]
            y_max = E_F - KE_array[-1]
        xi, yi = np.mgrid[x_min:x_max:x_pixel * 1j, y_min:y_max:y_pixel * 1j]
        zi = data_2D

        ax = fig.figure.subplots()
        ax.xaxis.set_major_locator(MultipleLocator(XtickStep))
        ax.xaxis.set_minor_locator(AutoMinorLocator(XminorTicks))
        ax.yaxis.set_major_locator(MultipleLocator(YtickStep))
        ax.yaxis.set_minor_locator(AutoMinorLocator(YminorTicks))
        ax.tick_params(axis='x', which='major', top=True, direction='in')
        ax.tick_params(axis='x', which='minor', top=True, direction='in')
        ax.tick_params(axis='y', which='major', right=True, direction='in')
        ax.tick_params(axis='y', which='minor', right=True, direction='in')
        ax.set_xlabel('$k (\AA^{-1})$')

        if reverseSign_k:
            # reverse k sign because often forward emission is on the negative k side
            # zi = np.flip(data_2D, axis=1)
            ax.set_xlim(left=-axisLimit[0][1], right=-axisLimit[0][0])
        else:
            ax.set_xlim(left=axisLimit[0][0], right=axisLimit[0][1])

        if plotMode == 'KE only':
            ax.set_ylabel('KE (eV)')
            ax.set_ylim(bottom=E_F - axisLimit[1][1], top=E_F - axisLimit[1][0])
        elif plotMode == 'BE only':
            ax.set_ylabel('BE (eV)')
            ax.set_ylim(bottom=axisLimit[1][1], top=axisLimit[1][0])
        else:  #'BE & KE': left BE, right KE
            ax.set_ylabel('BE (eV)')
            ax.set_ylim(bottom=axisLimit[1][1], top=axisLimit[1][0])
            ax.tick_params(axis='y', which='major', right=False, direction='in')
            ax.tick_params(axis='y', which='minor', right=False, direction='in')
            # secondary axis does not work here, because it is not a line plot
            ax2 = ax.twinx()
            ax2.set_ylabel('KE (eV)')
            trans_E = lambda x: E_F - x
            ymin, ymax = ax.get_ylim()
            ax2.set_ylim((trans_E(ymin), trans_E(ymax)))
            ax2.plot([], [])
            ax2.tick_params(axis='y', which='major', direction='in')
            ax2.tick_params(axis='y', which='minor', direction='in')

        if userLUT is None:
            cmap = cmap_bipolar
        else:
            cmap = LinearSegmentedColormap.from_list('cm_user', userLUT, N=100)

        pc = ax.pcolormesh(
            xi,
            yi,
            zi,
            rasterized=True,
            shading='auto',
            cmap=cmap,
            vmin=level_min,
            vmax=level_max)
        if showColorBar:
            fig.figure.colorbar(pc, ax=ax)
        self.show()

    ##### window container management #####
    def saveId(self, Id):
        # save the python id of this class instance
        self.Id = Id

    def closeEvent(self, event):
        # use saved python id to remove object from the dict windowContainer
        self.close()
        del self.parent.windowContainer[self.Id]
        # garbage collect to free memory
        gc.collect()