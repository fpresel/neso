import gc
import os
import sys

import h5py
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, AutoLocator
import matplotlib.path as path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, optimize, stats
from skimage import measure
from tifffile import imread

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QCheckBox, QDialog, QFileDialog, QLabel, QMainWindow, QMessageBox, QSizePolicy, QVBoxLayout, QWidget
import PyQt5.uic
import pyqtgraph as pg

from kmap_matplotlib import kmap_new_plotwindow
from bandmap_matplotlib import bandmap_new_plotwindow

# matplotlib related settings
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams["axes.axisbelow"] = True
colorlist_bipolar = [(0, 1, 1), (0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 1, 0)]
cmap_bipolar = LinearSegmentedColormap.from_list('cm_bipolar',
                                                 colorlist_bipolar,
                                                 N=100)
colorlist_aschoell = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (0, 0, 1)]
cmap_aschoell = LinearSegmentedColormap.from_list('cm_aschoell',
                                                  colorlist_aschoell,
                                                  N=100)
colorlist_cdad = [(0, 0, 1), (0, 0, 1), (1, 1, 1), (0, 0, 0), (0, 0, 0)]
cmap_cdad = LinearSegmentedColormap.from_list('cm_cdad', colorlist_cdad, N=100)
# pyqtgraph related settings
pg.setConfigOption('background', None)
pg.setConfigOption('foreground', 'k')
labelOption = {'position': 0.1, 'color': 'w', 'fill': 'k', 'movable': True}
labelStyle = {'color': 'k', 'font-size': '10pt'}
pos_bw = np.array([0.0, 1.0])
color_bw = np.array([[255, 255, 255, 255], [0, 0, 0, 255]], dtype=np.ubyte)
colorMap_bw = pg.ColorMap(pos_bw, color_bw)
pos_bipolar = np.array([0.0, 0.0001, 0.25, 0.5, 0.75, 1.0])
color_bipolar = np.array(
    [[0, 255, 255, 0], [0, 255, 255, 255], [0, 0, 255, 255], [0, 0, 0, 255],
     [255, 0, 0, 255], [255, 255, 0, 255]],
    dtype=np.ubyte)
colorMap_bipolar = pg.ColorMap(pos_bipolar, color_bipolar)
pos_viridis = np.array([0.0, 0.0001, 0.25, 0.5, 0.75, 1.0])
color_virdis = np.array(
    [[68, 1, 84, 0], [68, 1, 84, 255], [58, 82, 139, 255], [32, 144, 140, 255],
     [94, 201, 97, 255], [253, 231, 36, 255]],
    dtype=np.ubyte)
colorMap_virdis = pg.ColorMap(pos_viridis, color_virdis)
pos_aschoell = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
color_aschoell = np.array(
    [[0, 0, 0, 255], [255, 0, 0, 255], [255, 255, 0, 255],
     [255, 255, 255, 255], [0, 0, 255, 255]],
    dtype=np.ubyte)
colorMap_aschoell = pg.ColorMap(pos_aschoell, color_aschoell)
pos_cdad = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
color_cdad = np.array([[0, 0, 255, 255], [0, 0, 255, 255],
                       [255, 255, 255, 255], [0, 0, 0, 255], [0, 0, 0, 255]],
                      dtype=np.ubyte)
colorMap_cdad = pg.ColorMap(pos_cdad, color_cdad)

ui_mainwindow, bc_mainwindow = PyQt5.uic.loadUiType('GUI_Neso.ui')
ui_kmapExportDialog, bc_kmapExportDialog = PyQt5.uic.loadUiType(
    'GUI_kmapExportDialog.ui')
ui_bandmapExportDialog, bc_bandmapExportDialog = PyQt5.uic.loadUiType(
    'GUI_bandmapExportDialog.ui')
ui_CDADViewer, bc_CDADViewer = PyQt5.uic.loadUiType('GUI_CDADViewer.ui')
ui_textImageViewer, bc_textImageViewer = PyQt5.uic.loadUiType(
    'GUI_textImageViewer.ui')
ui_calibrationDialog, bc_calibrationDialog = PyQt5.uic.loadUiType(
    'GUI_calibrationDialog.ui')


# Use the two classes
class MainWindow(bc_mainwindow, ui_mainwindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # dict to hold all opened windows (important!)
        self.windowContainer = {}

        self.data_KEkxky = np.array([])
        self.E_F = 0
        self.KE_axis = np.array([])
        self.latticeconstant = 3.597  # Copper
        self.rotation = 0.0
        self.sliceNum = 0
        self.tiffilepaths = []

        self.KEkxky_plotmode = "original"  # "corrected"
        self.edc_plotted = False
        self.bandmap_plotted = False
        self.data_KEkxky_corrected_exist = False

        _tooltip_dispersion = "b, ky<sub>0</sub> & a are parameters for the parabola KE = b*(ky-ky<sub>0</sub>)<sup>2</sup> + a in the (ky,KE) map"
        self.doubleSpinBox_bParabola.setToolTip(_tooltip_dispersion)
        self.doubleSpinBox_ky0Parabola.setToolTip(_tooltip_dispersion)
        self.doubleSpinBox_aParabola.setToolTip(_tooltip_dispersion)

        self.imv_kmapStack = pg.ImageView(view=pg.PlotItem())
        self.imv_kmapStack.setPredefinedGradient('bipolar')
        self.imv_kmapStack.view.showAxis('top', show=True)
        self.imv_kmapStack.view.showAxis('right', show=True)
        self.imv_kmapStack.view.getAxis('top').setStyle(tickLength=5)
        self.imv_kmapStack.view.getAxis('bottom').setStyle(tickLength=5)
        self.imv_kmapStack.view.getAxis('left').setStyle(tickLength=5)
        self.imv_kmapStack.view.getAxis('right').setStyle(tickLength=5)
        self.imv_kmapStack.view.setLabel('bottom',
                                         text='k<sub>x</sub>',
                                         units='\u212B\u207B\u00B9',
                                         **labelStyle)
        self.imv_kmapStack.view.setLabel('left',
                                         text='k<sub>y</sub>',
                                         units='\u212B\u207B\u00B9',
                                         **labelStyle)
        self.imv_kmapStack.view.invertY(False)
        self.imv_kmapStack.view.setAspectLocked(lock=True)
        self.layout_kmapStack.addWidget(self.imv_kmapStack)
        # line for profile plot
        self.pline = pg.InfiniteLine(pos=QtCore.QPointF(0., 0.),
                                     movable=False,
                                     angle=90)
        self.imv_kmapStack.addItem(self.pline)
        self.pcircle = pg.CircleROI([-2.0, -2.0], [4.0, 4.0],
                                    movable=False,
                                    pen='y')
        self.imv_kmapStack.addItem(self.pcircle)
        for handle in self.pcircle.getHandles():
            self.pcircle.removeHandle(handle)
        pi_over_a = np.pi / self.latticeconstant
        self.rect_BZ = pg.RectROI([-pi_over_a * 2**0.5, -pi_over_a],
                                  [2 * pi_over_a * 2**0.5, 2 * pi_over_a],
                                  movable=False,
                                  pen="g")
        self.imv_kmapStack.addItem(self.rect_BZ)
        for handle in self.rect_BZ.getHandles():
            self.rect_BZ.removeHandle(handle)
        self.pline.hide()
        self.pcircle.hide()
        self.rect_BZ.hide()

        # edcPlot
        self.pw_edc = pg.PlotWidget()
        self.layout_edcPlot.addWidget(self.pw_edc)
        self.pdi_edc = self.pw_edc.plot()
        self.pdi_edc.setPen(pg.mkPen('k'))
        self.pdi_fitFermi = self.pw_edc.plot()
        self.pdi_fitFermi.setPen(pg.mkPen('r'))
        self.fitFermi_range = [27, 29]
        self.vregion_fitFermi = pg.LinearRegionItem(
            values=self.fitFermi_range,
            orientation=pg.LinearRegionItem.Vertical,
            movable=True,
            bounds=[0, 999])
        self.pw_edc.addItem(self.vregion_fitFermi)

        # profilePlot
        self.pw_profile = pg.PlotWidget()
        self.layout_profilePlot.addWidget(self.pw_profile)
        # bandmap
        self.imv_bandmap = pg.ImageView(view=pg.PlotItem())
        self.imv_bandmap.setColorMap(colorMap_bipolar)
        self.imv_bandmap.view.showAxis('top', show=True)
        self.imv_bandmap.view.showAxis('right', show=True)
        self.imv_bandmap.view.setAspectLocked(lock=False)
        self.imv_bandmap.view.invertY(False)
        self.pdi_parabola = self.imv_bandmap.view.plot()
        # self.pdi_parabola.setPen(pg.mkPen('k'))
        self.layout_bandmap.addWidget(self.imv_bandmap)

        self.setup_all_connections()

    def setup_all_connections(self):
        self.action_Load_TIF.triggered.connect(self.load_from_tif)
        self.action_Load_HDF.triggered.connect(self.load_from_hdf)
        self.action_Save_as_HDF.triggered.connect(self.save_as_hdf)
        self.action_Save_for_kMap.triggered.connect(
            self.save_GrazData_for_kMap)
        self.action_Export_k_map.triggered.connect(self.export_kmap)
        self.action_Export_bandmap.triggered.connect(self.export_bandmap)
        self.action_textImageViewer.triggered.connect(
            self.newwindow_textImageViewer)
        self.action_CDADViewer.triggered.connect(self.newwindow_CDADViewer)
        self.checkBox_showProfileTool.stateChanged.connect(
            self.show_profile_tool)
        self.checkBox_showCircle.stateChanged.connect(self.show_circle)
        self.checkBox_showBZ.stateChanged.connect(self.show_BZ)
        self.checkBox_rotate90BZ.stateChanged.connect(self.BZ_valueChanged)
        self.checkBox_useCroppedData.stateChanged.connect(self.toggle_crop)
        self.checkBox_useCorrectedData.stateChanged.connect(
            self.toggle_correction)
        self.doubleSpinBox_EFforPlot.valueChanged.connect(
            self.E_F_valueChanged)
        self.doubleSpinBox_kRange.valueChanged.connect(self.plot_KEkxky)
        self.doubleSpinBox_kxShift.valueChanged.connect(self.plot_KEkxky)
        self.doubleSpinBox_kyShift.valueChanged.connect(self.plot_KEkxky)
        self.doubleSpinBox_plineXpos.valueChanged.connect(
            self.plineX_valueChanged)
        self.doubleSpinBox_plineYpos.valueChanged.connect(
            self.plineY_valueChanged)
        self.doubleSpinBox_circleRadius.valueChanged.connect(
            self.pcircle_valueChanged)
        self.doubleSpinBox_latticeconstant.valueChanged.connect(
            self.BZ_valueChanged)
        self.horizontalSlider_kmapSliceNum.valueChanged.connect(
            self.KEkxky_sliderMoved)
        self.pushButton_calibrate.clicked.connect(self.open_calibration)
        self.pushButton_getKEvalues.clicked.connect(self.get_KE_values)
        self.pushButton_setKEvalues.clicked.connect(self.set_KE_values)
        self.pushButton_plotLineScan.clicked.connect(self.plot_linescan)
        self.pushButton_plotBandMap.clicked.connect(self.plot_bandmap)
        self.pushButton_plotEDC_allK.clicked.connect(self.plot_edc_all_k)
        self.pushButton_fitFermi.clicked.connect(self.fitFermi)
        self.pushButton_plot_kyKE.clicked.connect(self.plot_kyKE)
        self.pushButton_refreshParabola.clicked.connect(self.refresh_parabola)
        self.pushButton_refreshRotation.clicked.connect(self.refresh_rotation)
        self.radioButton_plineX.toggled.connect(self.toggleSpinBox_profile)
        self.radioButton_plineY.toggled.connect(self.toggleSpinBox_profile)
        self.radioButton_KE.toggled.connect(self.KE_BE_toggled)

    ###########################
    def BZ_valueChanged(self):
        self.checkBox_showBZ.setChecked(True)
        self.latticeconstant = self.doubleSpinBox_latticeconstant.value()
        pi_over_a = np.pi / self.latticeconstant
        # To be extended if other fcc surfaces other than (110) are used
        if self.checkBox_rotate90BZ.isChecked():
            pos_Y, pos_X = -pi_over_a * 2**0.5, -pi_over_a
            size_Y, size_X = 2 * pi_over_a * 2**0.5, 2 * pi_over_a
        else:
            pos_X, pos_Y = -pi_over_a * 2**0.5, -pi_over_a
            size_X, size_Y = 2 * pi_over_a * 2**0.5, 2 * pi_over_a
        self.rect_BZ.setPos([pos_X, pos_Y])
        self.rect_BZ.setSize([size_X, size_Y])

    def current_used_data(self):
        if self.checkBox_useCroppedData.isChecked():
            if self.KEkxky_plotmode == "original":
                print(
                    "Now using the original, cropped data of shape {}.".format(
                        self.data_KEkxky_crop.shape))
            else:
                print("Now using the corrected, cropped data of shape {}.".
                      format(self.data_KEkxky_crop.shape))
        else:
            if self.KEkxky_plotmode == "original":
                print("Now using the original data of shape {}.".format(
                    self.data_KEkxky.shape))
            else:
                print("Now using the corrected data of shape {}.".format(
                    self.data_KEkxky_corrected.shape))

    def E_F_valueChanged(self):
        self.E_F = self.doubleSpinBox_EFforPlot.value()

    def export_bandmap(self):
        if self.E_F == 0:
            QMessageBox.warning(self, "Warning",
                                "The Fermi level has not been calibrated!")
        else:
            histo = self.imv_bandmap.getHistogramWidget()
            level_min, level_max = histo.getLevels()
            userLUT = histo.gradient.colorMap().getLookupTable(mode='float',
                                                               alpha=False,
                                                               nPts=20)
            if self.radioButton_plineX.isChecked() == True:
                k_array = self.kx_axis
            else:
                k_array = self.ky_axis
            if self.KEkxky_plotmode == "original":
                KE_array = self.KE_axis
            else:
                KE_array = self.KE_axis_corrected
            self.dialog_bandmapExport = bandmapExportDialog(
                self.data_2D_bandmap,
                k_array,
                KE_array,
                self.E_F,
                level_min,
                level_max,
                userLUT,
                parent=self)
            self.dialog_bandmapExport.open()

    def export_kmap(self):
        histo = self.imv_kmapStack.getHistogramWidget()
        level_min, level_max = histo.getLevels()
        userLUT = histo.gradient.colorMap().getLookupTable(mode='float',
                                                           alpha=False,
                                                           nPts=20)
        pi_over_a = np.pi / self.latticeconstant
        if self.checkBox_rotate90BZ.isChecked():
            pos_Y, pos_X = -pi_over_a * 2**0.5, -pi_over_a
            size_Y, size_X = 2 * pi_over_a * 2**0.5, 2 * pi_over_a
        else:
            pos_X, pos_Y = -pi_over_a * 2**0.5, -pi_over_a
            size_X, size_Y = 2 * pi_over_a * 2**0.5, 2 * pi_over_a
        BZ_pos_size = [[pos_X, pos_Y], [size_X, size_Y]]
        self.dialog_kmapExport = kmapExportDialog(self.data_currentkmap,
                                                  self.kx_axis,
                                                  self.ky_axis,
                                                  level_min,
                                                  level_max,
                                                  userLUT,
                                                  BZ_pos_size,
                                                  parent=self)
        self.dialog_kmapExport.open()

    def Fermi_edge_residuals(self, params, only_residuals=True):
        Fermi_pos = params[0]  # in eV
        Temp = params[1]  # in K
        Norm = params[2]
        Bgd = params[3]
        k_B = 8.617332e-5  # eV/K

        if only_residuals:
            return self.y_edc[self.i_min:self.i_max] - Norm / (np.exp(
                (self.x_edc[self.i_min:self.i_max] - Fermi_pos) /
                (k_B * Temp)) + 1) - Bgd
        else:
            return Norm / (np.exp(
                (self.x_edc[self.i_min:self.i_max] - Fermi_pos) /
                (k_B * Temp)) + 1) + Bgd

    def fitFermi(self):
        # Fermi_pos in eV / Temp in K / Norm / Bgd
        initial_paramater_list = [
            self.doubleSpinBox_EFermi_initial.value(),
            self.doubleSpinBox_temperature_initial.value(), 50000, 0
        ]

        fit_KE_min, fit_KE_max = self.vregion_fitFermi.getRegion()

        if self.doubleSpinBox_EFermi_initial.value(
        ) > fit_KE_max or self.doubleSpinBox_EFermi_initial.value(
        ) < fit_KE_min:
            QMessageBox.warning(
                self, "Warning",
                "Please choose an initial EF value within the vertical region!"
            )
        else:
            self.i_min = (np.abs(self.x_edc - fit_KE_min)).argmin()
            self.i_max = (np.abs(self.x_edc - fit_KE_max)).argmin()

            if self.i_max != self.i_min:
                bestFit_param, cov_x, info, msg, ierr = optimize.leastsq(
                    self.Fermi_edge_residuals,
                    initial_paramater_list,
                    full_output=1)
                self.doubleSpinBox_EFermi_fit.setValue(bestFit_param[0])
                self.doubleSpinBox_EFforPlot.setValue(bestFit_param[0])
                self.doubleSpinBox_EFermi_initial.setValue(bestFit_param[0])
                self.doubleSpinBox_temperature_fit.setValue(bestFit_param[1])
                self.label_BgdNorm.setText(
                    'Bgd/Norm: {:.2f} / {:.2f} (a.u.)'.format(
                        bestFit_param[3], bestFit_param[2]))

            self.pdi_fitFermi.setData(x=self.x_edc[self.i_min:self.i_max],
                                      y=self.Fermi_edge_residuals(
                                          bestFit_param, False))

        # update BE min/max spinboxes for output plot
        # self.doubleSpinBox_BEmin.setValue(self.E_F - self.KE_array[-1])
        # self.doubleSpinBox_BEmax.setValue(self.E_F - self.KE_array[0])

    def get_bandmap_slice_from_interpolation(self,
                                             f_interp,
                                             gridData_flatten,
                                             gridShape=[0, 0]):
        a = gridShape[0]
        b = gridShape[1]
        if self.radioButton_plineX.isChecked() == True:
            if a == 1:
                bandmap_slice = f_interp(gridData_flatten)
            else:
                data = f_interp(gridData_flatten).reshape(a, b)
                bandmap_slice = np.average(data, axis=0)
        else:
            if b == 1:
                bandmap_slice = f_interp(gridData_flatten)
            else:
                data = f_interp(gridData_flatten).reshape(a, b)
                bandmap_slice = np.average(data, axis=1)
        return bandmap_slice

    def get_KE_values(self):
        self.filename_kmapStack = []
        self.KE_axis = np.array([])
        KEpos = self.spinBox_KEposInFileName.value()
        try:
            for _, filepath in enumerate(self.tiffilepaths):
                filename = filepath.split('/')[-1]
                self.filename_kmapStack.append(filename)
                KE = filename[:-4].split('_')[-KEpos]
                self.KE_axis = np.append(self.KE_axis, float(KE))
            self.initialize_data_loading()
            self.KEkxky_sliderMoved()
        except (ValueError, IndexError):
            QMessageBox.warning(
                self, "Warning",
                "Could not find the KE values from file names at position {} (backwards), please check!"
                .format(KEpos))

    def get_KX_KY_meshgrid_for_profileplot(self):
        w = self.doubleSpinBox_plineWidth.value()
        if self.radioButton_plineX.isChecked() == True:
            y0 = self.doubleSpinBox_plineXpos.value()
            if w == 0.:
                KX, KY = np.meshgrid(self.kx_axis, y0, sparse=False)
            else:
                y = np.linspace(y0 - w,
                                y0 + w,
                                num=int(1 + w * 2 / self.k_stepsize))
                KX, KY = np.meshgrid(self.kx_axis, y, sparse=False)
        else:
            x0 = self.doubleSpinBox_plineYpos.value()
            if w == 0.:
                KX, KY = np.meshgrid(x0, self.ky_axis, sparse=False)
            else:
                x = np.linspace(x0 - w,
                                x0 + w,
                                num=int(1 + w * 2 / self.k_stepsize))
                KX, KY = np.meshgrid(x, self.ky_axis, sparse=False)
        return KX, KY

    def initialize_data_loading(self):
        # self.spinBox_kmapSliceNum.setValue(1)
        # self.doubleSpinBox_kmapSliceKE.setValue(self.KE_axis[0])
        self.data_KEkxky_original = self.data_KEkxky
        self.horizontalSlider_kmapSliceNum.setValue(1)
        self.horizontalSlider_kmapSliceNum.setMinimum(1)
        self.horizontalSlider_kmapSliceNum.setMaximum(len(self.KE_axis))
        self.pushButton_plotLineScan.setEnabled(True)
        self.pushButton_plotBandMap.setEnabled(True)
        self.pushButton_plotEDC_allK.setEnabled(True)
        self.radioButton_KE.setChecked(True)
        self.action_Export_k_map.setEnabled(True)
        self.action_Export_bandmap.setEnabled(False)
        self.checkBox_useCroppedData.setEnabled(True)
        self.checkBox_useCroppedData.setChecked(False)
        self.checkBox_useCorrectedData.setEnabled(True)
        self.checkBox_useCorrectedData.setChecked(False)
        self.KE_stepsize = (self.KE_axis[-1] - self.KE_axis[0]) / len(
            self.KE_axis)
        self.pdi_edc.clear()
        self.pw_profile.clear()
        self.imv_bandmap.clear()
        self.pdi_parabola.clear()
        self.KEkxky_plotmode = "original"
        self.edc_plotted = False
        self.bandmap_plotted = False
        self.data_KEkxky_corrected_exist = False
        self.rotation = 0.0
        self.doubleSpinBox_rotationAngle.setValue(0.0)
        self.current_used_data()

    def KE_BE_toggled(self):
        if self.edc_plotted == True:
            self.update_edc()
        if self.bandmap_plotted == True:
            self.update_bandmap()

    def load_from_hdf(self):
        hdfpath, _ = QFileDialog.getOpenFileName(None, "Load data from HDF",
                                                 "", "NanoESCA data (*.hdf5)")
        if hdfpath != '':
            try:
                with h5py.File(hdfpath, 'r') as f:
                    self.filename_kmapStack = f["tiffilenames"][()]
                    self.KE_axis = f["KEs"][()]
                    self.data_KEkxky = f["data"][()]
                    self.doubleSpinBox_kRange.setValue(f["paras/kRange"][()])
                    self.doubleSpinBox_kxShift.setValue(f["paras/kxShift"][()])
                    self.doubleSpinBox_kyShift.setValue(f["paras/kyShift"][()])
                    self.doubleSpinBox_aParabola.setValue(f["paras/a"][()])
                    self.doubleSpinBox_bParabola.setValue(f["paras/b"][()])
                    self.doubleSpinBox_ky0Parabola.setValue(f["paras/ky0"][()])
                    self.doubleSpinBox_circleRadius.setValue(
                        f["paras/circleRadius"][()])
                    self.doubleSpinBox_EFforPlot.setValue(f["EF"][()])
                self.initialize_data_loading()
                self.KEkxky_sliderMoved()
            except:
                QMessageBox.warning(self, "Warning",
                                    "Error during loading data from hdf file!")

    def load_from_tif(self):
        self.KE_axis = np.array([])
        self.tiffilepaths, _ = QFileDialog.getOpenFileNames(
            None, "Open file", "", "NanoESCA image files (*.tif)")
        if self.tiffilepaths != []:
            # tif files are read as int using imread.
            # it should be changed to float, otherwise cannot assign NaN values to int.
            self.data_KEkxky = np.rot90(imread(self.tiffilepaths),
                                        axes=(2, 1)).astype(np.float32)
            # data cube is (Z, X, Y)
            # rotate in X/Y plane so that light comes from -Y direction
            QMessageBox.information(
                self, "Data loaded",
                "{} TIF files are loaded! Now you can get the KE values from their filenames (button 'Get KE values') or set the KE values yourself (button 'Set KE values')."
                .format(len(self.tiffilepaths)))
            self.pushButton_plotLineScan.setEnabled(False)
            self.pushButton_plotBandMap.setEnabled(False)
            self.pushButton_getKEvalues.setEnabled(True)
            self.pushButton_setKEvalues.setEnabled(True)

    def newwindow_CDADViewer(self):
        self.window_into_container(CDADViewer(parent=self))

    def newwindow_textImageViewer(self):
        self.window_into_container(textImageViewer(parent=self))

    def open_calibration(self):
        self.dialog_calibrationDialog = calibrationDialog()
        self.dialog_calibrationDialog.open()

    def pcircle_valueChanged(self):
        self.checkBox_showCircle.setChecked(True)
        r = self.doubleSpinBox_circleRadius.value()
        d = 2 * r
        self.pcircle.setPos([-r, -r])
        self.pcircle.setSize([d, d])

    def plineX_valueChanged(self):
        self.radioButton_plineX.setChecked(True)
        self.pline.setValue(self.doubleSpinBox_plineXpos.value())

    def plineY_valueChanged(self):
        self.radioButton_plineY.setChecked(True)
        self.pline.setValue(self.doubleSpinBox_plineYpos.value())

    def plot_bandmap(self):
        self.imv_bandmap.clear()
        self.pdi_parabola.clear()

        KX, KY = self.get_KX_KY_meshgrid_for_profileplot()
        points = np.dstack([KX, KY]).reshape(-1, 2)

        if self.checkBox_useCroppedData.isChecked():
            data = self.data_KEkxky_crop
            if self.KEkxky_plotmode == "original":
                KEaxis = self.KE_axis
            else:
                KEaxis = self.KE_axis_corrected
        else:
            if self.KEkxky_plotmode == "original":
                data = self.data_KEkxky
                KEaxis = self.KE_axis
            else:
                data = self.data_KEkxky_corrected
                KEaxis = self.KE_axis_corrected

        if self.radioButton_plineX.isChecked() == True:
            self.data_2D_bandmap = np.zeros([len(self.kx_axis), len(KEaxis)])
            self.imv_bandmap.view.setLabel('bottom',
                                           'k<sub>x</sub>',
                                           units='\u212B\u207B\u00B9',
                                           **labelStyle)
        else:
            self.data_2D_bandmap = np.zeros([len(self.ky_axis), len(KEaxis)])
            self.imv_bandmap.view.setLabel('bottom',
                                           'k<sub>y</sub>',
                                           units='\u212B\u207B\u00B9',
                                           **labelStyle)

        for i, _ in enumerate(KEaxis):
            f_interp = interpolate.RegularGridInterpolator(
                (self.kx_axis, self.ky_axis),
                data[i],
                bounds_error=False,
                fill_value=np.nan)

            self.data_2D_bandmap[:,
                                 i] = self.get_bandmap_slice_from_interpolation(
                                     f_interp, points, KX.shape)
        self.update_bandmap()
        self.bandmap_plotted = True
        self.action_Export_bandmap.setEnabled(True)

    def plot_edc_all_k(self):
        if self.checkBox_useCroppedData.isChecked():
            data = self.data_KEkxky_crop
            if self.KEkxky_plotmode == "original":
                self.x_edc = self.KE_axis
            else:
                self.x_edc = self.KE_axis_corrected
            self.y_edc = np.nanmean(data, axis=(1, 2))
        else:
            if self.KEkxky_plotmode == "original":
                data = self.data_KEkxky
                self.x_edc = self.KE_axis
            else:
                data = self.data_KEkxky_corrected
                self.x_edc = self.KE_axis_corrected
            self.y_edc = np.average(data, axis=(1, 2))

        self.pdi_edc.setData(y=self.y_edc, x=self.x_edc)
        self.pw_edc.autoRange()
        self.pw_edc.setLabel('left', 'Intensity', units=None, **labelStyle)
        self.pw_edc.setLabel('bottom',
                             'Kinetic Energy',
                             units='eV',
                             **labelStyle)
        self.pw_edc.enableAutoRange(x=True, y=True)
        self.vregion_fitFermi.setBounds([self.x_edc[0], self.x_edc[-1]])
        self.pushButton_fitFermi.setEnabled(True)
        self.edc_plotted = True

    def plot_kyKE(self):
        # plot averaged I(ky,KE) from I(KE,kx,ky) for correcting dispersion of analyzer
        # use self.imv_bandmap

        self.bandmap_plotted = False
        self.action_Export_bandmap.setEnabled(False)
        self.radioButton_KE.setChecked(True)

        self.imv_bandmap.clear()

        if self.KEkxky_plotmode == "original":
            data = np.mean(self.data_KEkxky, axis=1)
            posY = self.KE_axis[0]
        else:
            data = np.mean(self.data_KEkxky_corrected, axis=1)
            posY = self.KE_axis_corrected[0]
        data = np.flipud(np.rot90(data))

        scaleY = self.KE_stepsize
        self.imv_bandmap.view.setLabel('bottom',
                                       'k<sub>y</sub>',
                                       units='\u212B\u207B\u00B9',
                                       **labelStyle)
        self.imv_bandmap.view.setLabel('left',
                                       'Kinetic Energy',
                                       units='eV',
                                       **labelStyle)
        self.imv_bandmap.view.invertY(False)
        self.imv_bandmap.setImage(data,
                                  pos=[self.ky_axis[0], posY],
                                  scale=[self.k_stepsize, scaleY],
                                  autoRange=True)

    def plot_linescan(self):
        if self.KEkxky_plotmode == "original":
            data = self.data_KEkxky
        else:
            data = self.data_KEkxky_corrected
        f_interp = interpolate.RegularGridInterpolator(
            (self.kx_axis, self.ky_axis),
            data[self.sliceNum],
            bounds_error=False,
            fill_value=np.nan)
        self.pw_profile.clear()

        if self.radioButton_plineX.isChecked() == True:
            y0 = self.doubleSpinBox_plineXpos.value()
            linePoints = [(x, y0) for x in self.kx_axis]
            x_data = self.kx_axis
            y_data = f_interp(linePoints)
        else:
            x0 = self.doubleSpinBox_plineYpos.value()
            linePoints = [(x0, y) for y in self.ky_axis]
            x_data = self.ky_axis
            y_data = f_interp(linePoints)
        self.pw_profile.plot(x_data, y_data, pen=pg.mkPen('k'))
        self.pw_profile.setLabel('left', 'Intensity', units=None, **labelStyle)
        self.pw_profile.setLabel('bottom',
                                 'k\u2225',
                                 units='\u212B\u207B\u00B9',
                                 **labelStyle)
        self.pw_profile.enableAutoRange(x=True, y=True)

    def plot_KEkxky(self):
        kmax = self.doubleSpinBox_kRange.value()
        self.kxShift = self.doubleSpinBox_kxShift.value()
        self.kyShift = self.doubleSpinBox_kyShift.value()
        self.kxsize = self.data_KEkxky.shape[1]
        self.kysize = self.data_KEkxky.shape[2]
        self.kx_axis = np.linspace(-kmax + self.kxShift,
                                   kmax + self.kxShift,
                                   num=self.kxsize)
        self.ky_axis = np.linspace(-kmax + self.kyShift,
                                   kmax + self.kyShift,
                                   num=self.kysize)
        self.k_stepsize = 2 * kmax / self.kxsize
        if self.checkBox_useCroppedData.isChecked():
            self.data_currentkmap = self.data_KEkxky_crop[self.sliceNum]
        else:
            if self.KEkxky_plotmode == "original":
                self.data_currentkmap = self.data_KEkxky[self.sliceNum]
            else:
                self.data_currentkmap = self.data_KEkxky_corrected[
                    self.sliceNum]

        self.imv_kmapStack.setImage(
            self.data_currentkmap,
            pos=[-kmax + self.kxShift, -kmax + self.kyShift],
            scale=[self.k_stepsize, self.k_stepsize],
            autoRange=True)
        if self.KEkxky_plotmode == "original":
            self.imv_kmapStack.view.setTitle("<font size=4>#{}</font>".format(
                self.filename_kmapStack[self.sliceNum]))
        else:
            self.imv_kmapStack.view.setTitle(
                "<font size=4>#{}, {:.2f} eV (dispersion corrected)</font>".
                format(self.sliceNum + 1,
                       self.doubleSpinBox_kmapSliceKE.value()))

    def refresh_parabola(self):
        xdata = self.ky_axis
        a = self.doubleSpinBox_aParabola.value()
        b = self.doubleSpinBox_bParabola.value()
        ky0 = self.doubleSpinBox_ky0Parabola.value()
        ydata = [b * (x - ky0)**2 + a for x in xdata]
        self.pdi_parabola.clear()
        self.pdi_parabola.setData(y=ydata, x=xdata)
        self.data_KEkxky_corrected_exist = False

    def refresh_rotation(self):
        self.rotation = self.doubleSpinBox_rotationAngle.value()

        data_zeros = np.zeros(self.data_KEkxky_original.shape)
        data_rotated = ndimage.rotate(self.data_KEkxky_original,
                                      self.rotation,
                                      axes=(1, 2),
                                      reshape=False)
        self.data_KEkxky = data_zeros + data_rotated
        # correction data needs to be recalculated
        self.data_KEkxky_corrected_exist = False
        self.checkBox_useCorrectedData.setChecked(False)
        self.checkBox_useCroppedData.setChecked(False)
        QMessageBox.information(
            self, "Data rotated",
            "Raw data cube was rotated by {:.1f}\u00B0! You need to re-crop and/or re-correct based on the new rotated data cube."
            .format(self.rotation))
        self.plot_KEkxky()

    def save_as_hdf(self):
        outputpath, _ = QFileDialog.getSaveFileName(
            None, "Save current data to HDF",
            "_KE{}-{}".format(self.KE_axis[0],
                              self.KE_axis[-1]), "NanoESCA data (*.hdf5)")
        if outputpath != '':
            try:
                with h5py.File(outputpath, 'w') as f:
                    f["KEs"] = self.KE_axis
                    f["EF"] = self.E_F

                    f.create_dataset("tiffilenames",
                                     data=np.array(self.filename_kmapStack,
                                                   dtype='S'))
                    # f["data"] = self.data_KEkxky
                    f.create_dataset("data",
                                     data=self.data_KEkxky,
                                     compression="gzip")

                    f.create_group("paras")
                    f["paras/kRange"] = self.doubleSpinBox_kRange.value()
                    f["paras/kxShift"] = self.doubleSpinBox_kxShift.value()
                    f["paras/kyShift"] = self.doubleSpinBox_kyShift.value()
                    f["paras/a"] = self.doubleSpinBox_aParabola.value()
                    f["paras/b"] = self.doubleSpinBox_bParabola.value()
                    f["paras/ky0"] = self.doubleSpinBox_ky0Parabola.value()
                    f["paras/circleRadius"] = self.doubleSpinBox_circleRadius.value(
                    )
                QMessageBox.information(
                    self, "Data saved",
                    "Data cube are saved to {}!".format(outputpath))
            except:
                QMessageBox.warning(self, "Warning",
                                    "Error during saving data to hdf file!")

    def save_GrazData_for_kMap(self):
        # shifting and cropping data
        print('converting data')
        # BUGFIX
        if self.KEkxky_plotmode == "original":
            data = self.data_KEkxky
            ke = self.KE_axis / 27.211386  #convert to hartree
        else:
            data = self.data_KEkxky_corrected
            ke = self.KE_axis_corrected / 27.211386  #convert to hartree
        # RESUME; replaced "self.data_KEkxky" with "data" in all occurrences
 #       ke = self.KE_axis / 27.211386  #convert to hartree
        data_shift = np.copy(data)
        data_zeros = np.copy(data)
        xshift = int(self.doubleSpinBox_kxShift.value() / self.k_stepsize)
        yshift = int(self.doubleSpinBox_kyShift.value() / self.k_stepsize)
        a = int(self.kxsize / 2)
        b = int(self.kysize / 2)
        print('entering data set')
        for i in range(data.shape[0]):
            data_zeros[i] = np.zeros(data[i].shape)
            data_shift[i] = ndimage.shift(data[i],
                                          (xshift, yshift))
            data_shift[i] = data_shift[i] + data_zeros[i]
            radius = (np.sqrt(
                ke[i] *
                2)) / 0.529177249  # radius of circle according to E_kin in A
            r = int(radius / self.k_stepsize)
            y_1, x_1 = np.ogrid[-a:self.kxsize - a, -b:self.kysize - b]
            mask = x_1 * x_1 + y_1 * y_1 >= r * r
            data_shift[i, mask] = np.nan
        print('conversion done')

        outputpath, _ = QFileDialog.getSaveFileName(
            None, "Save current data to process in kMap.py",
            "_KE{}-{}".format(self.KE_axis[0],
                              self.KE_axis[-1]), "NanoESCA data (*.hdf5)")
        if outputpath != '':
            try:
                with h5py.File(outputpath, 'w') as f:
                    f["name"] = 'hdf5 data'
                    f["axis_1_label"] = 'E_kin'
                    f["axis_2_label"] = 'kx'
                    f["axis_3_label"] = 'ky'
                    f["axis_1_units"] = 'eV'
                    f["axis_2_units"] = '1/A'
                    f["axis_3_units"] = '1/A'
                    kRange = self.doubleSpinBox_kRange.value()
                    kEmin = self.KE_axis[0]
                    kEmax = self.KE_axis[-1]
                    f["axis_1_range"] = [kEmin, kEmax]
                    f["axis_2_range"] = [-kRange, kRange]
                    f["axis_3_range"] = [-kRange, kRange]

                    f.create_dataset("data",
                                     data=data_shift,
                                     dtype='f8',
                                     compression="gzip")

                QMessageBox.information(
                    self, "Data saved",
                    "Data cube are saved to {}!".format(outputpath))
            except:
                QMessageBox.warning(self, "Warning",
                                    "Error during saving data to hdf file!")

    def set_KE_values(self):
        self.filename_kmapStack = []
        KEmin = self.doubleSpinBox_KEmin.value()
        KEmax = self.doubleSpinBox_KEmax.value()
        if KEmin > KEmax:
            KEmin, KEmax = KEmax, KEmin
            self.doubleSpinBox_KEmin.setValue(KEmin)
            self.doubleSpinBox_KEmax.setValue(KEmax)
        try:
            for _, filepath in enumerate(self.tiffilepaths):
                filename = filepath.split('/')[-1]
                self.filename_kmapStack.append(filename)
            self.KE_axis = np.linspace(KEmin,
                                       KEmax,
                                       num=len(self.tiffilepaths))
            self.initialize_data_loading()
            self.KEkxky_sliderMoved()
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Warning",
                                "Could not set the KE values, please check!")

    def show_BZ(self):
        if self.checkBox_showBZ.isChecked():
            self.rect_BZ.show()
        else:
            self.rect_BZ.hide()

    def show_circle(self):
        if self.checkBox_showCircle.isChecked():
            self.pcircle.show()
        else:
            self.pcircle.hide()

    def show_profile_tool(self):
        if self.checkBox_showProfileTool.isChecked():
            self.pline.show()
        else:
            self.pline.hide()

    def KEkxky_sliderMoved(self):
        self.sliceNum = self.horizontalSlider_kmapSliceNum.value() - 1
        self.spinBox_kmapSliceNum.setValue(self.sliceNum + 1)
        if self.KEkxky_plotmode == "original":
            self.doubleSpinBox_kmapSliceKE.setValue(
                self.KE_axis[self.sliceNum])
        else:
            self.doubleSpinBox_kmapSliceKE.setValue(
                self.KE_axis_corrected[self.sliceNum])
        self.plot_KEkxky()

    def toggle_correction(self):
        if self.checkBox_useCorrectedData.isChecked() == False:
            self.KEkxky_plotmode = "original"
            self.horizontalSlider_kmapSliceNum.setMinimum(1)
            self.horizontalSlider_kmapSliceNum.setMaximum(len(self.KE_axis))
            self.horizontalSlider_kmapSliceNum.setValue(1)
            self.sliceNum = 0
        else:
            self.KEkxky_plotmode = "corrected"
            if self.data_KEkxky_corrected_exist == False:
                # find the new KE range due the parabolic correction (it gets shorter)
                b = self.doubleSpinBox_bParabola.value()
                ky0 = self.doubleSpinBox_ky0Parabola.value()
                KEleft = abs(b * (self.ky_axis[0] - ky0)**2)
                KEright = abs(b * (self.ky_axis[-1] - ky0)**2)
                KEchange = KEleft if KEleft > KEright else KEright
                if b < 0:
                    KEmin = self.KE_axis[0] + KEchange
                    KEmax = self.KE_axis[-1]
                else:
                    KEmin = self.KE_axis[0]
                    KEmax = self.KE_axis[-1] - KEchange
                self.KE_axis_corrected = np.linspace(
                    KEmin,
                    KEmax,
                    num=int(1 + (KEmax - KEmin) / self.KE_stepsize))
                # interpolation [KE,kx,ky]
                points = []
                for _, KE in enumerate(self.KE_axis_corrected):
                    points.append([(b * (ky - ky0)**2 + KE, ky)
                                   for ky in self.ky_axis])
                points = np.array(points).reshape([-1, 2])
                self.data_KEkxky_corrected = np.zeros([
                    len(self.KE_axis_corrected),
                    len(self.kx_axis),
                    len(self.ky_axis)
                ])
                # from time import perf_counter
                # t_start = perf_counter()
                for i in range(len(self.kx_axis)):
                    f_interp = interpolate.RegularGridInterpolator(
                        (self.KE_axis, self.ky_axis),
                        self.data_KEkxky[:, i, :],
                        bounds_error=False,
                        fill_value=np.nan)
                    self.data_KEkxky_corrected[:, i, :] = f_interp(
                        points).reshape(
                            [len(self.KE_axis_corrected),
                             len(self.ky_axis)])
                self.data_KEkxky_corrected_exist = True
                # t_stop = perf_counter()
                # print("Elapsed time in seconds:", t_stop - t_start)
            else:
                pass
            self.horizontalSlider_kmapSliceNum.setMinimum(1)
            self.horizontalSlider_kmapSliceNum.setMaximum(
                len(self.KE_axis_corrected))
            self.horizontalSlider_kmapSliceNum.setValue(1)
            self.sliceNum = 0

        self.toggle_crop()  # make sure self.data_KEkxky_crop is updated

    def toggle_crop(self):
        # a,b : index of center
        a = int(self.kxsize / 2 - self.kxShift / self.k_stepsize)
        print(self.kxsize, self.kxShift, self.k_stepsize)
        b = int(self.kysize / 2 - self.kyShift / self.k_stepsize)
        r = int(self.doubleSpinBox_circleRadius.value() / self.k_stepsize)
        print(self.doubleSpinBox_circleRadius.value())
        y, x = np.ogrid[-a:self.kxsize - a, -b:self.kysize - b]
        mask = x * x + y * y >= r * r
        if self.KEkxky_plotmode == "original":
            self.data_KEkxky_crop = np.copy(self.data_KEkxky)
            for i in range(self.data_KEkxky.shape[0]):
                self.data_KEkxky_crop[i, mask] = np.nan
        else:
            self.data_KEkxky_crop = np.copy(self.data_KEkxky_corrected)
            for i in range(self.data_KEkxky_corrected.shape[0]):
                self.data_KEkxky_crop[i, mask] = np.nan
        self.current_used_data()
        self.plot_KEkxky()

    def toggleSpinBox_profile(self):
        if self.radioButton_plineX.isChecked() == True:
            self.pline.setAngle(0.)
            self.pline.setValue(self.doubleSpinBox_plineXpos.value())
            self.pline.show()
        else:
            self.pline.setAngle(90.)
            self.pline.setValue(self.doubleSpinBox_plineYpos.value())
            self.pline.show()
        self.checkBox_showProfileTool.setChecked(True)

    def update_bandmap(self):
        if self.KEkxky_plotmode == "original":
            KEaxis = self.KE_axis
        else:
            KEaxis = self.KE_axis_corrected
        if self.radioButton_KE.isChecked():
            posY = KEaxis[0]
            scaleY = self.KE_stepsize
            self.imv_bandmap.view.setLabel('left',
                                           'Kinetic Energy',
                                           units='eV',
                                           **labelStyle)
            self.imv_bandmap.view.invertY(False)
        else:
            posY = self.E_F - KEaxis[0]
            scaleY = -self.KE_stepsize
            self.imv_bandmap.view.setLabel('left',
                                           'Binding Energy',
                                           units='eV',
                                           **labelStyle)
            self.imv_bandmap.view.invertY(True)

        if self.radioButton_plineX.isChecked() == True:
            self.imv_bandmap.setImage(self.data_2D_bandmap,
                                      pos=[self.kx_axis[0], posY],
                                      scale=[self.k_stepsize, scaleY],
                                      autoRange=True)
        else:
            self.imv_bandmap.setImage(self.data_2D_bandmap,
                                      pos=[self.ky_axis[0], posY],
                                      scale=[self.k_stepsize, scaleY],
                                      autoRange=True)

    def update_edc(self):
        if self.KEkxky_plotmode == "original":
            KEaxis = self.KE_axis
        else:
            KEaxis = self.KE_axis_corrected
        if self.radioButton_KE.isChecked():
            self.x_edc = KEaxis
            self.pw_edc.setLabel('bottom',
                                 'Kinetic Energy',
                                 units='eV',
                                 **labelStyle)
            self.pw_edc.invertX(False)
            self.pushButton_fitFermi.setEnabled(True)
            self.vregion_fitFermi.show()
            self.pdi_fitFermi.show()
        else:
            self.x_edc = self.E_F - KEaxis
            self.pw_edc.setLabel('bottom',
                                 'Binding Energy',
                                 units='eV',
                                 **labelStyle)
            self.pw_edc.invertX(True)
            self.pushButton_fitFermi.setEnabled(False)
            self.vregion_fitFermi.hide()
            self.pdi_fitFermi.hide()
        self.pdi_edc.setData(y=self.y_edc, x=self.x_edc)
        self.pw_edc.autoRange()

    ###### window management ######
    def window_into_container(self, newWindow):
        """ save the newwindow into the dict windowContainer with its python id as the item key, so that the new window is owned by mainwindow and does not get lost """
        newId = id(newWindow)
        self.windowContainer[newId] = newWindow
        newWindow.saveId(newId)

    def closeEvent(self, event):
        # clear the windowContainer dict will remove the references and the windows will be garbage collected
        self.windowContainer.clear()
        self.close()


class newwindow_4plots(QMainWindow):
    """
    make plot in pop-up window via matplotlib
    """
    def __init__(self,
                 data_plus,
                 data_minus,
                 data_diff,
                 data_theo,
                 x_theo,
                 y_theo,
                 figsize_x,
                 figsize_y,
                 axis_limit,
                 k_limit,
                 level_min=0,
                 level_max=1,
                 parent=None):
        super().__init__()
        self.parent = parent
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setWindowTitle('Plot...')
        _main = QWidget()
        self.setCentralWidget(_main)

        inch = 1.0 / 25.4
        fig = FigureCanvas(
            Figure(figsize=(figsize_x * inch, figsize_y * inch), dpi=100))
        fig.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        ax0 = fig.figure.add_subplot(141)
        ax1 = fig.figure.add_subplot(142, sharex=ax0, sharey=ax0)
        ax2 = fig.figure.add_subplot(143, sharex=ax0, sharey=ax0)
        ax3 = fig.figure.add_subplot(144, sharex=ax0, sharey=ax0)

        layout = QVBoxLayout(_main)
        layout.addWidget(fig)
        self.addToolBar(NavigationToolbar2QT(fig, self))

        axis_min = -2.42  #k_array[0]
        axis_max = 2.42  #k_array[-1]
        axis_pixel = 600  #len(k_array)
        x_exp, y_exp = np.mgrid[axis_min:axis_max:axis_pixel * 1j,
                                axis_min:axis_max:axis_pixel * 1j]

        circle = path.Path.circle(center=(0.0, 0.0), radius=k_limit)
        # fig, ax = plt.subplots()
        # ax.pcolormesh(X, Y, Z, clip_path=(circle, ax.transData), clip_on=True)

        ax0.pcolormesh(x_exp,
                       y_exp,
                       data_plus,
                       rasterized=True,
                       shading='auto',
                       cmap=cmap_aschoell,
                       clip_path=(circle, ax0.transData),
                       clip_on=True)
        ax0.set_aspect('equal')
        ax0.set_xticks([-2, -1, 0, 1, 2])
        ax0.set_yticks([-2, -1, 0, 1, 2])
        ax0.xaxis.set_tick_params(top=True, direction='in')
        ax0.yaxis.set_tick_params(right=True, direction='in')
        ax0.set_xlim(left=-1 * axis_limit, right=axis_limit)
        ax0.set_ylim(bottom=-1 * axis_limit, top=axis_limit)

        ax1.pcolormesh(x_exp,
                       y_exp,
                       data_minus,
                       rasterized=True,
                       shading='auto',
                       cmap=cmap_aschoell,
                       clip_path=(circle, ax1.transData),
                       clip_on=True)
        ax1.set_aspect('equal')
        ax1.set_xticks([-2, -1, 0, 1, 2])
        ax1.set_yticks([-2, -1, 0, 1, 2])
        ax1.xaxis.set_tick_params(top=True, direction='in')
        ax1.yaxis.set_tick_params(right=True, direction='in')

        ax2.pcolormesh(x_exp,
                       y_exp,
                       data_diff,
                       rasterized=True,
                       shading='auto',
                       cmap=cmap_cdad,
                       clip_path=(circle, ax2.transData),
                       clip_on=True)
        ax2.set_aspect('equal')
        ax2.set_xticks([-2, -1, 0, 1, 2])
        ax2.set_yticks([-2, -1, 0, 1, 2])
        ax2.xaxis.set_tick_params(top=True, direction='in')
        ax2.yaxis.set_tick_params(right=True, direction='in')

        # transfer ogrid to mgrid
        x_theo = np.tile(x_theo, (1, y_theo.shape[1]))
        y_theo = np.tile(y_theo, (x_theo.shape[0], 1))
        cs = ax3.contour(x_theo, y_theo, data_theo, levels=[1.0])
        contourpaths = cs.collections[0].get_paths()
        # make a mask from it with the dimensions of Z
        points = np.append(x_exp.reshape(-1, 1), y_exp.reshape(-1, 1), axis=1)
        mask = np.empty(x_exp.shape)
        for j in range(len(contourpaths)):
            vert = contourpaths[j]
            mask += vert.contains_points(points).reshape(x_exp.shape)
        mask = np.logical_not(mask)
        data_diff = np.ma.array(data_diff, mask=mask)
        # finally, plot
        ax3.clear()
        ax3.pcolormesh(x_exp,
                       y_exp,
                       data_diff,
                       rasterized=True,
                       shading='auto',
                       cmap=cmap_cdad,
                       clip_path=(circle, ax3.transData),
                       clip_on=True)
        ax3.set_aspect('equal')
        ax3.set_xticks([-2, -1, 0, 1, 2])
        ax3.set_yticks([-2, -1, 0, 1, 2])
        ax3.xaxis.set_tick_params(top=True, direction='in')
        ax3.yaxis.set_tick_params(right=True, direction='in')

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


class newwindow_1plot(QMainWindow):
    """
    make plot in pop-up window via matplotlib
    """
    def __init__(self,
                 data,
                 figsize_x,
                 figsize_y,
                 axis_limit,
                 k_limit,
                 level_min=0,
                 level_max=1,
                 parent=None):
        super().__init__()
        self.parent = parent
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setWindowTitle('Plot...')
        _main = QWidget()
        self.setCentralWidget(_main)

        inch = 1.0 / 25.4
        fig = FigureCanvas(
            Figure(figsize=(figsize_x * inch, figsize_y * inch), dpi=100))
        fig.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        ax = fig.figure.add_subplot(111)
        # ax1 = fig.figure.add_subplot(142, sharex=ax0, sharey=ax0)
        # ax2 = fig.figure.add_subplot(143, sharex=ax0, sharey=ax0)
        # ax3 = fig.figure.add_subplot(144, sharex=ax0, sharey=ax0)

        layout = QVBoxLayout(_main)
        layout.addWidget(fig)
        self.addToolBar(NavigationToolbar2QT(fig, self))

        axis_min = -2.42  #k_array[0]
        axis_max = 2.42  #k_array[-1]
        axis_pixel = 600  #len(k_array)
        x_exp, y_exp = np.mgrid[axis_min:axis_max:axis_pixel * 1j,
                                axis_min:axis_max:axis_pixel * 1j]

        circle = path.Path.circle(center=(0.0, 0.0), radius=k_limit)
        # fig, ax = plt.subplots()
        # ax.pcolormesh(X, Y, Z, clip_path=(circle, ax.transData), clip_on=True)

        ax.pcolormesh(x_exp,
                      y_exp,
                      data,
                      rasterized=True,
                      shading='auto',
                      cmap=cmap_aschoell,
                      clip_path=(circle, ax.transData),
                      clip_on=True)
        ax.set_aspect('equal')
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.xaxis.set_tick_params(top=True, direction='in')
        ax.yaxis.set_tick_params(right=True, direction='in')
        ax.set_xlim(left=-1 * axis_limit, right=axis_limit)
        ax.set_ylim(bottom=-1 * axis_limit, top=axis_limit)

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


class kmapExportDialog(bc_kmapExportDialog, ui_kmapExportDialog):
    def __init__(self,
                 data_kmap,
                 kx_axis,
                 ky_axis,
                 level_min,
                 level_max,
                 userLUT,
                 BZ_pos_size,
                 parent=None):
        super().__init__()
        self.setupUi(self)
        self.data_kmap = data_kmap
        self.kx_axis = kx_axis
        self.ky_axis = ky_axis
        self.level_min = level_min
        self.level_max = level_max
        self.userLUT = userLUT
        self.BZ_pos_size = BZ_pos_size
        self.parent = parent
        self.pushButton_OK.clicked.connect(self.plot_kmap_matplotlib)
        self.pushButton_saveHDF.clicked.connect(self.save_kmap_as_hdf)

    def plot_kmap_matplotlib(self):
        figSize = self.spinBox_figSize.value()
        axisLimit = self.doubleSpinBox_axisLimit.value()
        minorTicksNumber = self.spinBox_minorTicks.value()
        gridDensity = self.spinBox_gridDensity.value()
        if self.checkBox_showColorBar.isChecked():
            showColorBar = True
        else:
            showColorBar = False
        if self.checkBox_showBZ.isChecked():
            showBZ = True
        else:
            showBZ = False

        self.parent.window_into_container(
            kmap_new_plotwindow(self.data_kmap,
                                self.kx_axis,
                                self.level_min,
                                self.level_max,
                                figSize,
                                axisLimit,
                                self.BZ_pos_size,
                                minorTicksNumber=minorTicksNumber,
                                gridDensity=gridDensity,
                                showColorBar=showColorBar,
                                showBZ=showBZ,
                                userLUT=self.userLUT,
                                ky_array=self.ky_axis,
                                parent=self.parent))

    def save_kmap_as_hdf(self):
        outputpath, _ = QFileDialog.getSaveFileName(
            None, "Save current k-map to HDF", "kmap_", "k-map data (*.hdf5)")
        if outputpath != '':
            try:
                with h5py.File(outputpath, 'w') as f:
                    f.create_dataset('data_kmap',
                                     data=self.data_kmap,
                                     dtype='f8',
                                     compression='gzip',
                                     compression_opts=9)
                    f["kx_array"] = self.kx_axis
                    f["ky_array"] = self.ky_axis
                QMessageBox.information(
                    self, "Data saved",
                    "Data cube are saved to {}!".format(outputpath))
            except:
                QMessageBox.warning(self, "Warning",
                                    "Error during saving data to hdf file!")


class bandmapExportDialog(bc_bandmapExportDialog, ui_bandmapExportDialog):
    def __init__(self,
                 data_bandmap,
                 k_array,
                 KE_array,
                 E_F,
                 level_min,
                 level_max,
                 userLUT,
                 parent=None):
        super().__init__()
        self.setupUi(self)
        self.data_bandmap = data_bandmap
        self.k_array = k_array
        self.KE_array = KE_array
        self.E_F = E_F
        self.level_min = level_min
        self.level_max = level_max
        self.userLUT = userLUT
        self.parent = parent
        self.pushButton_OK.clicked.connect(self.plot_bandmap_matplotlib)
        self.pushButton_saveHDF.clicked.connect(self.save_bandmap_as_hdf)

    def plot_bandmap_matplotlib(self):
        figSize = [
            self.spinBox_figSizeX.value(),
            self.spinBox_figSizeY.value()
        ]
        axisLimit = [[
            self.doubleSpinBox_kmin.value(),
            self.doubleSpinBox_kmax.value()
        ], [
            self.doubleSpinBox_BEmin.value(),
            self.doubleSpinBox_BEmax.value()
        ]]
        XtickStep = self.doubleSpinBox_XtickStep.value()
        YtickStep = self.doubleSpinBox_YtickStep.value()
        XminorTicks = self.spinBox_XminorTicks.value()
        YminorTicks = self.spinBox_YminorTicks.value()
        modeIndex = self.comboBox_plotMode.currentIndex()

        if modeIndex == 0:
            plotMode = 'BE only'
        elif modeIndex == 1:
            plotMode = 'KE only'
        else:
            plotMode = 'BE & KE'
        if self.checkBox_showColorBar.isChecked():
            showColorBar = True
        else:
            showColorBar = False
        if self.checkBox_reverseSign_k.isChecked():
            reverseSign_k = True
        else:
            reverseSign_k = False

        self.parent.window_into_container(
            bandmap_new_plotwindow(self.data_bandmap,
                                   self.k_array,
                                   self.KE_array,
                                   self.E_F,
                                   self.level_min,
                                   self.level_max,
                                   figSize,
                                   axisLimit,
                                   XtickStep=XtickStep,
                                   YtickStep=YtickStep,
                                   XminorTicks=XminorTicks,
                                   YminorTicks=YminorTicks,
                                   showColorBar=showColorBar,
                                   userLUT=self.userLUT,
                                   plotMode=plotMode,
                                   reverseSign_k=reverseSign_k,
                                   parent=self.parent))

    def save_bandmap_as_hdf(self):
        outputpath, _ = QFileDialog.getSaveFileName(
            None, "Save current bandmap to HDF", "bandmap_",
            "Bandmap data (*.hdf5)")
        if outputpath != '':
            try:
                with h5py.File(outputpath, 'w') as f:
                    f.create_dataset('data_bandmap',
                                     data=self.data_bandmap,
                                     dtype='f8',
                                     compression='gzip',
                                     compression_opts=9)
                    f["k_array"] = self.k_array
                    f["KE_array"] = self.KE_array
                    f["E_F"] = self.E_F
                QMessageBox.information(
                    self, "Data saved",
                    "Data cube are saved to {}!".format(outputpath))
            except:
                QMessageBox.warning(self, "Warning",
                                    "Error during saving data to hdf file!")


class calibrationDialog(ui_calibrationDialog, bc_calibrationDialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_calculateHorizon.clicked.connect(
            self.calculate_horizon)

    def calculate_horizon(self):
        self.tiffilepaths = []
        Ekin = self.doubleSpinBox_calibrationEkin.value()
        workf = self.doubleSpinBox_calibrationWF.value()
        conversion = (2.0 * (9.1093837 * 10**-31)) / (
            (1.0545718 * 10**-34)**2)  # (2*me)/(h-bar**2), units = J * m**2
        conversion = conversion * (1.6021766 * 10**-19) * (
            (10**-10)**2)  # converting to eV, A**2
        horizon = np.sqrt(conversion) * np.sqrt(Ekin - workf)
        print(horizon)
        QMessageBox.information(
            self, "NanoESCA Graz calibration",
            "PE horizon was calculated to be " + str(round(horizon, 2)) +
            ". Please calibrate k||range accordingly.".format(
                len(self.tiffilepaths)))


class CDADViewer(bc_CDADViewer, ui_CDADViewer):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        self.parent = parent
        self.data_cplus = None
        self.data_cminus = None
        self.rotated90 = False

        self.imv_cplus = pg.ImageView(view=pg.PlotItem())
        # self.imv_cplus.setPredefinedGradient('bipolar')
        self.imv_cplus.setColorMap(colorMap_aschoell)
        self.imv_cplus.view.showAxis('top', show=True)
        self.imv_cplus.view.showAxis('right', show=True)
        self.imv_cplus.view.getAxis('top').setStyle(tickLength=5)
        self.imv_cplus.view.getAxis('bottom').setStyle(tickLength=5)
        self.imv_cplus.view.getAxis('left').setStyle(tickLength=5)
        self.imv_cplus.view.getAxis('right').setStyle(tickLength=5)
        self.imv_cplus.view.setLabel('bottom',
                                     text='k<sub>x</sub>',
                                     units='\u212B\u207B\u00B9',
                                     **labelStyle)
        self.imv_cplus.view.setLabel('left',
                                     text='k<sub>y</sub>',
                                     units='\u212B\u207B\u00B9',
                                     **labelStyle)
        self.imv_cplus.view.invertY(False)
        self.imv_cplus.view.setAspectLocked(lock=True)
        self.layout_cPlus.addWidget(self.imv_cplus)

        self.imv_cminus = pg.ImageView(view=pg.PlotItem())
        # self.imv_cminus.setPredefinedGradient('bipolar')
        self.imv_cminus.setColorMap(colorMap_aschoell)
        self.imv_cminus.view.showAxis('top', show=True)
        self.imv_cminus.view.showAxis('right', show=True)
        self.imv_cminus.view.getAxis('top').setStyle(tickLength=5)
        self.imv_cminus.view.getAxis('bottom').setStyle(tickLength=5)
        self.imv_cminus.view.getAxis('left').setStyle(tickLength=5)
        self.imv_cminus.view.getAxis('right').setStyle(tickLength=5)
        self.imv_cminus.view.setLabel('bottom',
                                      text='k<sub>x</sub>',
                                      units='\u212B\u207B\u00B9',
                                      **labelStyle)
        self.imv_cminus.view.setLabel('left',
                                      text='k<sub>y</sub>',
                                      units='\u212B\u207B\u00B9',
                                      **labelStyle)
        self.imv_cminus.view.invertY(False)
        self.imv_cminus.view.setAspectLocked(lock=True)
        self.layout_cMinus.addWidget(self.imv_cminus)

        self.imv_diff = pg.ImageView(view=pg.PlotItem())
        # self.imv_diff.setPredefinedGradient('bipolar')
        self.imv_diff.setColorMap(colorMap_cdad)
        self.imv_diff.view.showAxis('top', show=True)
        self.imv_diff.view.showAxis('right', show=True)
        self.imv_diff.view.getAxis('top').setStyle(tickLength=5)
        self.imv_diff.view.getAxis('bottom').setStyle(tickLength=5)
        self.imv_diff.view.getAxis('left').setStyle(tickLength=5)
        self.imv_diff.view.getAxis('right').setStyle(tickLength=5)
        self.imv_diff.view.setLabel('bottom',
                                    text='k<sub>x</sub>',
                                    units='\u212B\u207B\u00B9',
                                    **labelStyle)
        self.imv_diff.view.setLabel('left',
                                    text='k<sub>y</sub>',
                                    units='\u212B\u207B\u00B9',
                                    **labelStyle)
        self.imv_diff.view.invertY(False)
        self.imv_diff.view.setAspectLocked(lock=True)
        self.layout_diff.addWidget(self.imv_diff)

        self.pw_theo = pg.PlotWidget()
        self.pw_theo.setXRange(-2.5, 2.5)
        self.pw_theo.setYRange(-2.5, 2.5)
        self.pw_theo.setAspectLocked(lock=True)
        self.layout_theoContour.addWidget(self.pw_theo)

        self.setup_all_connections()
        self.show()

    def setup_all_connections(self):
        self.pushButton_loadCPlus.clicked.connect(self.load_cplus_data)
        self.pushButton_loadCMinus.clicked.connect(self.load_cminus_data)
        self.pushButton_loadDiff.clicked.connect(self.load_diff_data)
        self.pushButton_normalize.clicked.connect(self.normalize)
        self.pushButton_plotNormDiff.clicked.connect(self.plot_norm_diff)
        self.pushButton_loadTheo.clicked.connect(self.load_theo_data)
        self.pushButton_refreshTheo.clicked.connect(self.plot_theo_data)
        self.pushButton_plotAll.clicked.connect(self.plot_all_newWindow)
        self.pushButton_plotOne.clicked.connect(self.plot_one_newWindow)
        self.checkBox_rot90.stateChanged.connect(self.checkBox_rot90_changed)

    #######################################
    def checkBox_rot90_changed(self):
        if self.checkBox_rot90.isChecked():
            self.rotated90 = True
        else:
            self.rotated90 = False

    def load_cminus_data(self):
        filepath, _ = QFileDialog.getOpenFileName(None, "Open file", "",
                                                  "NanoESCA data (*.txt)")
        if filepath != '':
            self.data_cminus = self.load_from_txt(filepath)
            self.imv_cminus.clear()
            self.imv_cminus.setImage(self.data_cminus,
                                     pos=[-2.42, -2.42],
                                     scale=[4.84 / 600, 4.84 / 600],
                                     autoRange=True)
            self.imv_cminus.view.setTitle("{}".format(filepath.split("/")[-1]))

    def load_cplus_data(self):
        filepath, _ = QFileDialog.getOpenFileName(None, "Open file", "",
                                                  "NanoESCA data (*.txt)")
        if filepath != '':
            self.data_cplus = self.load_from_txt(filepath)
            self.imv_cplus.clear()
            self.imv_cplus.setImage(self.data_cplus,
                                    pos=[-2.42, -2.42],
                                    scale=[4.84 / 600, 4.84 / 600],
                                    autoRange=True)
            self.imv_cplus.view.setTitle("{}".format(filepath.split("/")[-1]))

    def load_diff_data(self):
        filepath, _ = QFileDialog.getOpenFileName(None, "Open file", "",
                                                  "NanoESCA data (*.txt)")
        if filepath != '':
            data = self.load_from_txt(filepath)
            self.imv_diff.clear()
            self.imv_diff.setImage(data,
                                   pos=[-2.42, -2.42],
                                   scale=[4.84 / 600, 4.84 / 600],
                                   autoRange=True)
            self.imv_diff.view.setTitle("{}".format(filepath.split("/")[-1]))

    def load_theo_data(self):
        filepath, _ = QFileDialog.getOpenFileName(None, "Open file", "",
                                                  "NanoESCA data (*.hdf5)")
        if filepath != '':
            with h5py.File(filepath, 'r') as f:
                self.data_theo = f["data_2D"][()]
                krange = f["krange"][()]
                kStepSize = f["kStepSize"][()]
                intensity_max = np.nanmax(self.data_theo)
                intensity_min = np.nanmin(self.data_theo)
                print(intensity_min, intensity_max)
                self.label_theoRange.setText(
                    "Theory values range: {:.2f}~{:.2f}".format(
                        intensity_min, intensity_max))
            kmin = krange[0][0]
            kmax = krange[0][-1]
            axis_pixel = len(krange[0])
            self.x_theo, self.y_theo = np.ogrid[
                kmin:kmax:self.data_theo.shape[0] * 1j,
                kmin:kmax:self.data_theo.shape[1] * 1j]

            self.plot_theo_data()

    def load_from_txt(self, filepath):
        try:
            reader = pd.read_csv(
                filepath,
                encoding='cp1252',
                delimiter='\t',
                engine='c',
                dtype=float,
                float_precision=None,
                header=None,
            )
            data = reader.values
        except:
            raise
        data[data == 0.0] = np.nan
        data = np.rot90(data, axes=(1, 0))
        return data

    def normalize(self):
        if self.data_cplus is not None and self.data_cminus is not None:
            self.data_cplus_norm = self.normalize_np_array(self.data_cplus)
            self.data_cminus_norm = self.normalize_np_array(self.data_cminus)
            self.imv_cplus.clear()
            self.imv_cplus.setImage(self.data_cplus_norm,
                                    pos=[-2.42, -2.42],
                                    scale=[4.84 / 600, 4.84 / 600],
                                    autoRange=True)
            self.imv_cminus.clear()
            self.imv_cminus.setImage(self.data_cminus_norm,
                                     pos=[-2.42, -2.42],
                                     scale=[4.84 / 600, 4.84 / 600],
                                     autoRange=True)
            self.pushButton_plotNormDiff.setEnabled(True)

    def normalize_np_array(self, nparray):
        min_cplus = np.nanmin(nparray)
        max_cplus = np.nanmax(nparray)
        nparray_norm = (nparray - min_cplus) / (max_cplus - min_cplus)
        return nparray_norm

    def plot_norm_diff(self):
        self.data_diff_norm = self.data_cplus_norm - self.data_cminus_norm
        self.imv_diff.clear()
        self.imv_diff.setImage(self.data_diff_norm,
                               pos=[-2.42, -2.42],
                               scale=[4.84 / 600, 4.84 / 600],
                               autoRange=True)

    def plot_all_newWindow(self):
        # histo = self.imv_diff.getHistogramWidget()
        # level_min, level_max = histo.getLevels()
        figsizeX = self.spinBox_figsizeX.value()
        figsizeY = self.spinBox_figsizeY.value()
        axisLimit = self.doubleSpinBox_axisLimit.value()
        kLimit = self.doubleSpinBox_cropK.value()
        if self.rotated90:
            data_theo = np.rot90(self.data_theo, axes=(0, 1))
        else:
            data_theo = self.data_theo
        self.parent.window_into_container(
            newwindow_4plots(self.data_cplus_norm,
                             self.data_cminus_norm,
                             self.data_diff_norm,
                             data_theo,
                             self.x_theo,
                             self.y_theo,
                             figsizeX,
                             figsizeY,
                             axisLimit,
                             kLimit,
                             parent=self.parent))

    def plot_one_newWindow(self):
        figsizeX = self.spinBox_figsizeX.value()
        figsizeY = self.spinBox_figsizeX.value()
        axisLimit = self.doubleSpinBox_axisLimit.value()
        kLimit = self.doubleSpinBox_cropK.value()
        # if self.rotated90:
        #     data_theo = np.rot90(self.data_theo, axes=(0, 1))
        # else:
        #     data_theo = self.data_theo
        self.parent.window_into_container(
            newwindow_1plot(self.data_cplus,
                            figsizeX,
                            figsizeY,
                            axisLimit,
                            kLimit,
                            parent=self.parent))

    def plot_theo_data(self):
        if self.rotated90:
            data_theo = np.rot90(self.data_theo, axes=(0, 1))
        else:
            data_theo = self.data_theo
        level = self.doubleSpinBox_contourLevel.value()
        contours = measure.find_contours(data_theo, level)
        fx = interpolate.interp1d(np.arange(0, self.x_theo.shape[0]),
                                  self.x_theo.flatten())
        fy = interpolate.interp1d(np.arange(0, self.y_theo.shape[1]),
                                  self.y_theo.flatten())
        self.pw_theo.clear()

        for contour in contours:
            contour[:, 0] = fx(contour[:, 0])
            contour[:, 1] = fy(contour[:, 1])
            self.pw_theo.plot(contour[:, 0],
                              contour[:, 1],
                              pen='k',
                              symbol='d')

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


class textImageViewer(bc_textImageViewer, ui_textImageViewer):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        self.parent = parent
        self.latticeconstant = 3.597  # Copper

        self.imv = pg.ImageView(view=pg.PlotItem())
        self.imv.setPredefinedGradient('bipolar')
        # self.imv.setColorMap(colorMap_aschoell)
        self.imv.view.showAxis('top', show=True)
        self.imv.view.showAxis('right', show=True)
        self.imv.view.getAxis('top').setStyle(tickLength=5)
        self.imv.view.getAxis('bottom').setStyle(tickLength=5)
        self.imv.view.getAxis('left').setStyle(tickLength=5)
        self.imv.view.getAxis('right').setStyle(tickLength=5)
        self.imv.view.setLabel('bottom',
                               text='k<sub>x</sub>',
                               units='\u212B\u207B\u00B9',
                               **labelStyle)
        self.imv.view.setLabel('left',
                               text='k<sub>y</sub>',
                               units='\u212B\u207B\u00B9',
                               **labelStyle)
        self.imv.view.invertY(False)
        self.imv.view.setAspectLocked(lock=True)
        self.layout_imv.addWidget(self.imv)

        # line for profile plot
        self.pline = pg.InfiniteLine(pos=QtCore.QPointF(0., 0.),
                                     movable=False,
                                     angle=90)
        self.imv.addItem(self.pline)
        self.pcircle = pg.CircleROI([-2.0, -2.0], [4.0, 4.0],
                                    movable=False,
                                    pen='y')
        self.imv.addItem(self.pcircle)
        for handle in self.pcircle.getHandles():
            self.pcircle.removeHandle(handle)
        pi_over_a = np.pi / self.latticeconstant
        self.rect_BZ = pg.RectROI([-pi_over_a * 2**0.5, -pi_over_a],
                                  [2 * pi_over_a * 2**0.5, 2 * pi_over_a],
                                  movable=False,
                                  pen="g")
        self.imv.addItem(self.rect_BZ)
        for handle in self.rect_BZ.getHandles():
            self.rect_BZ.removeHandle(handle)
        self.pline.hide()
        self.pcircle.hide()
        self.rect_BZ.hide()

        self.setup_all_connections()
        self.show()

    def setup_all_connections(self):
        self.action_Load_TXT.triggered.connect(self.load_from_txt)
        self.action_Export_k_map.triggered.connect(self.export_kmap)
        self.checkBox_rotate90BZ.stateChanged.connect(self.BZ_valueChanged)
        self.checkBox_showBZ.stateChanged.connect(self.show_BZ)
        self.checkBox_showCircle.stateChanged.connect(self.show_circle)
        self.checkBox_showProfileTool.stateChanged.connect(
            self.show_profile_tool)
        self.checkBox_useCroppedData.stateChanged.connect(self.toggle_crop)
        self.doubleSpinBox_circleRadius.valueChanged.connect(
            self.pcircle_valueChanged)
        self.doubleSpinBox_latticeconstant.valueChanged.connect(
            self.BZ_valueChanged)
        self.doubleSpinBox_kRange.valueChanged.connect(self.refresh_plot)
        self.doubleSpinBox_kxShift.valueChanged.connect(self.refresh_plot)
        self.doubleSpinBox_kyShift.valueChanged.connect(self.refresh_plot)
        self.doubleSpinBox_plineXpos.valueChanged.connect(
            self.plineX_valueChanged)
        self.doubleSpinBox_plineYpos.valueChanged.connect(
            self.plineY_valueChanged)
        self.radioButton_plineX.toggled.connect(self.toggleSpinBox_profile)
        self.radioButton_plineY.toggled.connect(self.toggleSpinBox_profile)

    #######################################
    def BZ_valueChanged(self):
        self.checkBox_showBZ.setChecked(True)
        self.latticeconstant = self.doubleSpinBox_latticeconstant.value()
        pi_over_a = np.pi / self.latticeconstant
        # To be extended if other fcc surfaces other than (110) are used
        if self.checkBox_rotate90BZ.isChecked():
            pos_Y, pos_X = -pi_over_a * 2**0.5, -pi_over_a
            size_Y, size_X = 2 * pi_over_a * 2**0.5, 2 * pi_over_a
        else:
            pos_X, pos_Y = -pi_over_a * 2**0.5, -pi_over_a
            size_X, size_Y = 2 * pi_over_a * 2**0.5, 2 * pi_over_a
        self.rect_BZ.setPos([pos_X, pos_Y])
        self.rect_BZ.setSize([size_X, size_Y])

    def export_kmap(self):
        histo = self.imv.getHistogramWidget()
        level_min, level_max = histo.getLevels()
        userLUT = histo.gradient.colorMap().getLookupTable(mode='float',
                                                           alpha=False,
                                                           nPts=20)
        pi_over_a = np.pi / self.latticeconstant
        if self.checkBox_rotate90BZ.isChecked():
            pos_Y, pos_X = -pi_over_a * 2**0.5, -pi_over_a
            size_Y, size_X = 2 * pi_over_a * 2**0.5, 2 * pi_over_a
        else:
            pos_X, pos_Y = -pi_over_a * 2**0.5, -pi_over_a
            size_X, size_Y = 2 * pi_over_a * 2**0.5, 2 * pi_over_a
        BZ_pos_size = [[pos_X, pos_Y], [size_X, size_Y]]
        if self.checkBox_useCroppedData.isChecked():
            img = self.data_crop
        else:
            img = self.data
        self.dialog_kmapExport = kmapExportDialog(img,
                                                  self.kx_axis,
                                                  self.ky_axis,
                                                  level_min,
                                                  level_max,
                                                  userLUT,
                                                  BZ_pos_size,
                                                  parent=self.parent)
        self.dialog_kmapExport.open()

    def initialize_data_loading(self):
        # self.horizontalSlider_kmapSliceNum.setValue(1)
        # self.horizontalSlider_kmapSliceNum.setMinimum(1)
        # self.horizontalSlider_kmapSliceNum.setMaximum(len(self.KE_axis))
        # self.pushButton_plotLineScan.setEnabled(True)
        # self.pushButton_plotBandMap.setEnabled(True)
        # self.pushButton_plotEDC_allK.setEnabled(True)
        self.action_Export_k_map.setEnabled(True)
        # self.action_Export_bandmap.setEnabled(False)
        self.checkBox_useCroppedData.setEnabled(True)
        self.checkBox_useCroppedData.setChecked(False)
        # self.checkBox_useCorrectedData.setEnabled(True)
        # self.checkBox_useCorrectedData.setChecked(False)
        # self.KE_stepsize = (self.KE_axis[-1] - self.KE_axis[0]) / len(
        #     self.KE_axis)
        # self.pdi_edc.clear()
        # self.pw_profile.clear()
        # self.imv_bandmap.clear()
        # self.pdi_parabola.clear()
        # self.KEkxky_plotmode = "original"
        # self.edc_plotted = False
        # self.bandmap_plotted = False
        # self.data_KEkxky_corrected_exist = False
        # self.rotation = 0.0
        # self.doubleSpinBox_rotationAngle.setValue(0.0)
        # self.current_used_data()

    def load_from_txt(self):
        filepath, _ = QFileDialog.getOpenFileName(
            None, "Open file", "",
            "NanoESCA data exported from ImageJ (*.txt)")
        if filepath != '':
            try:
                reader = pd.read_csv(
                    filepath,
                    encoding='cp1252',
                    delimiter='\t',
                    engine='c',
                    dtype=float,
                    float_precision=None,
                    header=None,
                )
                self.data = reader.values
                if self.data.shape[0] == self.data.shape[1]:
                    self.label_datashape.setText("Data in shape {}".format(
                        self.data.shape))
                    self.data[self.data == 0.0] = np.nan
                    self.data = np.rot90(self.data, axes=(1, 0))
                    self.imv.clear()
                    self.imv.view.setTitle("{}".format(
                        filepath.split("/")[-1]))
                    self.refresh_plot()
                    self.initialize_data_loading()
                else:
                    QMessageBox.warning(
                        self, "Warning",
                        "The imported data has different x and y dimension, which is not supported now!"
                    )
            except:
                QMessageBox.warning(self, "Warning",
                                    "Error during saving data to txt file!")

    def pcircle_valueChanged(self):
        self.checkBox_showCircle.setChecked(True)
        r = self.doubleSpinBox_circleRadius.value()
        d = 2 * r
        self.pcircle.setPos([-r, -r])
        self.pcircle.setSize([d, d])

    def plineX_valueChanged(self):
        self.radioButton_plineX.setChecked(True)
        self.pline.setValue(self.doubleSpinBox_plineXpos.value())

    def plineY_valueChanged(self):
        self.radioButton_plineY.setChecked(True)
        self.pline.setValue(self.doubleSpinBox_plineYpos.value())

    def refresh_plot(self):
        if hasattr(self, "data"):
            kmax = self.doubleSpinBox_kRange.value()
            self.kxShift = self.doubleSpinBox_kxShift.value()
            self.kyShift = self.doubleSpinBox_kyShift.value()
            self.kxsize = self.data.shape[0]
            self.kysize = self.data.shape[1]
            self.k_stepsize = 2 * kmax / self.kxsize
            self.kx_axis = np.linspace(-1 * kmax + self.kxShift,
                                       kmax + self.kxShift,
                                       num=self.kxsize)
            self.ky_axis = np.linspace(-1 * kmax + self.kyShift,
                                       kmax + self.kyShift,
                                       num=self.kysize)
            pos = [-1 * kmax + self.kxShift, -1 * kmax + self.kyShift]
            scale = [2 * kmax / self.kxsize, 2 * kmax / self.kysize]
            if self.checkBox_useCroppedData.isChecked():
                img = self.data_crop
            else:
                img = self.data
            self.imv.setImage(img, pos=pos, scale=scale, autoRange=True)

    def show_BZ(self):
        if self.checkBox_showBZ.isChecked():
            self.rect_BZ.show()
        else:
            self.rect_BZ.hide()

    def show_circle(self):
        if self.checkBox_showCircle.isChecked():
            self.pcircle.show()
        else:
            self.pcircle.hide()

    def show_profile_tool(self):
        if self.checkBox_showProfileTool.isChecked():
            self.pline.show()
        else:
            self.pline.hide()

    def toggle_crop(self):
        # a,b : index of center
        a = int(self.kxsize / 2 - self.kxShift / self.k_stepsize)
        b = int(self.kysize / 2 - self.kyShift / self.k_stepsize)
        r = int(self.doubleSpinBox_circleRadius.value() / self.k_stepsize)
        y, x = np.ogrid[-a:self.kxsize - a, -b:self.kysize - b]
        mask = x * x + y * y >= r * r
        self.data_crop = np.copy(self.data)
        self.data_crop[mask] = np.nan
        self.refresh_plot()

    def toggleSpinBox_profile(self):
        if self.radioButton_plineX.isChecked() == True:
            self.pline.setAngle(0.)
            self.pline.setValue(self.doubleSpinBox_plineXpos.value())
            self.pline.show()
        else:
            self.pline.setAngle(90.)
            self.pline.setValue(self.doubleSpinBox_plineYpos.value())
            self.pline.show()
        self.checkBox_showProfileTool.setChecked(True)

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
