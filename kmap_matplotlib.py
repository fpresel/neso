import gc
from PySide6.QtWidgets import QMainWindow, QSizePolicy, QVBoxLayout, QWidget
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams["axes.axisbelow"] = True
matplotlib.rcParams['font.size'] = 8.0
matplotlib.rcParams['axes.titlesize'] = 8.0
import numpy as np


#  Class modified from Mozi.py. Manages matplotlib-windows for export of plots.
class kmap_new_plotwindow(QMainWindow):

    def __init__(
            self,
            data_2D,
            kx_array,
            level_min,
            level_max,
            figSize,
            axisLimit,
            BZ_pos_size,
            minorTicksNumber=4,
            gridDensity=1,
            showColorBar=False,
            showBZ=False,
            userLUT=None,
            ky_array=np.array([]),
            # if ky_array is not specified, it means ky_array is identical as kx_array.
            parent=None):
        super().__init__()
        self.parent=parent
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setWindowTitle('Plot...')
        _main = QWidget()
        self.setCentralWidget(_main)
        sizeInInch = figSize / 25.4
        fig = FigureCanvas(Figure(figsize=(sizeInInch, sizeInInch), dpi=100))
        fig.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QVBoxLayout(_main)
        layout.addWidget(fig)
        self.addToolBar(NavigationToolbar2QT(fig, self))
        if len(ky_array) == 0:
            xi, yi = np.meshgrid(kx_array, kx_array, indexing='ij')
        else:
            xi, yi = np.meshgrid(kx_array, ky_array, indexing='ij')
        zi = data_2D
        ax = fig.figure.subplots()
        majorLocator = MultipleLocator(1)
        minorLocator = AutoMinorLocator(minorTicksNumber)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.tick_params(axis='x', which='major', top=True, direction='in')
        ax.tick_params(axis='x', which='minor', top=True, direction='in')
        ax.tick_params(axis='y', which='major', right=True, direction='in')
        ax.tick_params(axis='y', which='minor', right=True, direction='in')
        if gridDensity == 0:
            pass
        elif gridDensity == 1:
            ax.grid(b=True, which='major')
        elif gridDensity == 2:
            ax.grid(b=True, which='both')
        ax.set_aspect('equal')
        ax.set_xlim(left=-1 * axisLimit, right=axisLimit)
        ax.set_ylim(bottom=-1 * axisLimit, top=axisLimit)
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
        if gridDensity == 0:
            pass
        elif gridDensity == 1:
            ax.grid(b=True, which='major')
        elif gridDensity == 2:
            ax.grid(b=True, which='both')
        if showColorBar:
            fig.figure.colorbar(pc, ax=ax)
        if showBZ:
            ax.add_patch(
                patches.Rectangle((BZ_pos_size[0][0], BZ_pos_size[0][1]),
                                  BZ_pos_size[1][0],
                                  BZ_pos_size[1][1],
                                  fill=False,
                                  ec='k'))
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