# Dual.py

# region imports
from Air import *
from matplotlib import pyplot as plt
from PyQt5 import QtWidgets as qtw
import numpy as np
# endregion

# region Model
class dualCycleModel:
    def __init__(self, p_initial=1e5, v_cylinder=3e-3, t_initial=300, pressure_ratio=1.5, cutoff=1.2, ratio=18.0, name='Air Standard Dual Cycle'):
        self.units = units()
        self.units.SI = True
        self.air = air()

        self.p_initial = p_initial
        self.T_initial = t_initial
        self.PressureRatio = pressure_ratio  # P3/P2
        self.Cutoff = cutoff                # rc = V4/V3
        self.Ratio = ratio                  # r = V1/V2
        self.V_Cylinder = v_cylinder

        self.air.set(P=self.p_initial, T=self.T_initial)
        self.air.n = self.V_Cylinder / self.air.State.v
        self.air.m = self.air.n * self.air.MW

        # Set thermodynamic states
        self.State1 = self.air.set(P=self.p_initial, T=self.T_initial, name="State 1 - BDC")
        self.State2 = self.air.set(v=self.State1.v / self.Ratio, s=self.State1.s, name="State 2 - TDC")

        P3 = self.State2.P * self.PressureRatio
        self.State3 = self.air.set(P=P3, v=self.State2.v, name="State 3")

        self.State4 = self.air.set(P=P3, v=self.State3.v * self.Cutoff, name="State 4")

        self.State5 = self.air.set(v=self.State1.v, s=self.State4.s, name="State 5 - BDC")

        self.W_Compression = self.State2.u - self.State1.u
        self.W_Expansion = self.State4.u - self.State5.u
        self.W_PressureStroke = P3 * (self.State4.v - self.State3.v)

        self.Q_In = self.State3.u - self.State2.u + self.State4.h - self.State3.h
        self.Q_Out = self.State5.u - self.State1.u

        self.W_Cycle = self.W_Expansion + self.W_PressureStroke - self.W_Compression
        self.Eff = 100 * self.W_Cycle / self.Q_In

        self.upperCurve = StateDataForPlotting()
        self.lowerCurve = StateDataForPlotting()
        self.calculated = True
        self.cycleType = "dual"
# endregion

# region Controller
class dualCycleController:
    def __init__(self, model=None, ax=None):
        self.model = dualCycleModel() if model is None else model
        self.view = dualCycleView()
        self.view.ax = ax

    def calc(self):
        T0 = float(self.view.le_TLow.text())
        P0 = float(self.view.le_P0.text())
        V0 = float(self.view.le_V0.text())
        PR = float(self.view.le_THigh.text())  # Using High Temp input for Pressure Ratio
        CR = float(self.view.le_CR.text())     # CR is cutoff ratio
        ratio = self.model.Ratio               # compression ratio
        metric = self.view.rdo_Metric.isChecked()
        self.set(T_0=T0, P_0=P0, V_0=V0, pressure_ratio=PR, cutoff=CR, ratio=ratio, SI=metric)

    def set(self, T_0=300.0, P_0=1e5, V_0=3e-3, pressure_ratio=1.5, cutoff=1.2, ratio=18.0, SI=True):
        self.model.units.set(SI=SI)
        self.model.T_initial = T_0 if SI else T_0 / self.model.units.CF_T
        self.model.p_initial = P_0 if SI else P_0 / self.model.units.CF_P
        self.model.V_Cylinder = V_0 if SI else V_0 / self.model.units.CF_V
        self.model.PressureRatio = pressure_ratio
        self.model.Cutoff = cutoff
        self.model.Ratio = ratio

        # Reconstruct model with updated parameters
        self.model = dualCycleModel(
            p_initial=self.model.p_initial,
            v_cylinder=self.model.V_Cylinder,
            t_initial=self.model.T_initial,
            pressure_ratio=self.model.PressureRatio,
            cutoff=self.model.Cutoff,
            ratio=self.model.Ratio,
        )

        self.buildDataForPlotting()
        self.updateView()

    def buildDataForPlotting(self):
        self.model.upperCurve.clear()
        self.model.lowerCurve.clear()
        a = air()

        # 2-3 (v = const, P rising)
        DeltaP = np.linspace(self.model.State2.P, self.model.State3.P, 30)
        for P in DeltaP:
            state = a.set(P=P, v=self.model.State2.v)
            self.model.upperCurve.add((state.T, state.P, state.u, state.h, state.s, state.v))

        # 3-4 (P = const, v increasing)
        DeltaV = np.linspace(self.model.State3.v, self.model.State4.v, 30)
        for v in DeltaV:
            state = a.set(P=self.model.State3.P, v=v)
            self.model.upperCurve.add((state.T, state.P, state.u, state.h, state.s, state.v))

        # 4-5 (s = const, v increasing)
        DeltaV = np.linspace(self.model.State4.v, self.model.State5.v, 30)
        for v in DeltaV:
            state = a.set(v=v, s=self.model.State4.s)
            self.model.upperCurve.add((state.T, state.P, state.u, state.h, state.s, state.v))

        # 1-2 (s = const, v decreasing)
        DeltaV = np.linspace(self.model.State1.v, self.model.State2.v, 30)
        for v in DeltaV:
            state = a.set(v=v, s=self.model.State1.s)
            self.model.lowerCurve.add((state.T, state.P, state.u, state.h, state.s, state.v))

    def plot_cycle_XY(self, X='s', Y='T', logx=False, logy=False, mass=False, total=False):
        self.view.plot_cycle_XY(self.model, X, Y, logx, logy, mass, total)

    def print_summary(self):
        self.view.print_summary(self.model)

    def get_summary(self):
        return self.view.get_summary(self.model)

    def setWidgets(self, w=None):
        [self.view.lbl_THigh, self.view.lbl_TLow, self.view.lbl_P0, self.view.lbl_V0, self.view.lbl_CR,
         self.view.le_THigh, self.view.le_TLow, self.view.le_P0, self.view.le_V0, self.view.le_CR,
         self.view.le_T1, self.view.le_T2, self.view.le_T3, self.view.le_T4,
         self.view.lbl_T1Units, self.view.lbl_T2Units, self.view.lbl_T3Units, self.view.lbl_T4Units,
         self.view.le_PowerStroke, self.view.le_CompressionStroke, self.view.le_HeatAdded, self.view.le_Efficiency,
         self.view.lbl_PowerStrokeUnits, self.view.lbl_CompressionStrokeUnits, self.view.lbl_HeatInUnits,
         self.view.rdo_Metric, self.view.cmb_Abcissa, self.view.cmb_Ordinate,
         self.view.chk_LogAbcissa, self.view.chk_LogOrdinate, self.view.ax, self.view.canvas] = w

    def updateView(self):
        self.view.updateView(self.model)
# endregion

# region View
class dualCycleView:
    def __init__(self):
        self.lbl_THigh = qtw.QLabel()
        self.lbl_TLow = qtw.QLabel()
        self.lbl_P0 = qtw.QLabel()
        self.lbl_V0 = qtw.QLabel()
        self.lbl_CR = qtw.QLabel()
        self.le_THigh = qtw.QLineEdit()
        self.le_TLow = qtw.QLineEdit()
        self.le_P0 = qtw.QLineEdit()
        self.le_V0 = qtw.QLineEdit()
        self.le_CR = qtw.QLineEdit()
        self.le_T1 = qtw.QLineEdit()
        self.le_T2 = qtw.QLineEdit()
        self.le_T3 = qtw.QLineEdit()
        self.le_T4 = qtw.QLineEdit()
        self.lbl_T1Units = qtw.QLabel()
        self.lbl_T2Units = qtw.QLabel()
        self.lbl_T3Units = qtw.QLabel()
        self.lbl_T4Units = qtw.QLabel()
        self.le_Efficiency = qtw.QLineEdit()
        self.le_PowerStroke = qtw.QLineEdit()
        self.le_CompressionStroke = qtw.QLineEdit()
        self.le_HeatAdded = qtw.QLineEdit()
        self.lbl_PowerStrokeUnits = qtw.QLabel()
        self.lbl_CompressionStrokeUnits = qtw.QLabel()
        self.lbl_HeatInUnits = qtw.QLabel()
        self.rdo_Metric = qtw.QRadioButton()
        self.cmb_Abcissa = qtw.QComboBox()
        self.cmb_Ordinate = qtw.QComboBox()
        self.chk_LogAbcissa = qtw.QCheckBox()
        self.chk_LogOrdinate = qtw.QCheckBox()
        self.canvas = None
        self.ax = None

    def updateView(self, cycle):
        cycle.units.set(SI=self.rdo_Metric.isChecked())
        logx = self.chk_LogAbcissa.isChecked()
        logy = self.chk_LogOrdinate.isChecked()
        xvar = self.cmb_Abcissa.currentText()
        yvar = self.cmb_Ordinate.currentText()
        if cycle.calculated:
            self.plot_cycle_XY(cycle, X=xvar, Y=yvar, logx=logx, logy=logy, mass=False, total=True)

        self.le_Efficiency.setText(f"{cycle.Eff:.2f}")
        self.le_PowerStroke.setText(f"{cycle.W_Expansion:.2f}")
        self.le_CompressionStroke.setText(f"{cycle.W_Compression:.2f}")
        self.le_HeatAdded.setText(f"{cycle.Q_In:.2f}")

        self.le_T1.setText(f"{cycle.State1.T:.1f}")
        self.le_T2.setText(f"{cycle.State2.T:.1f}")
        self.le_T3.setText(f"{cycle.State3.T:.1f}")
        self.le_T4.setText(f"{cycle.State4.T:.1f}")

    def plot_cycle_XY(self, model, X='s', Y='T', logx=False, logy=False, mass=False, total=False):
        self.ax.clear()
        XData = model.lowerCurve.getDataCol(X) + model.upperCurve.getDataCol(X)
        YData = model.lowerCurve.getDataCol(Y) + model.upperCurve.getDataCol(Y)
        self.ax.plot(XData, YData, '-ok')
        self.ax.set_xlabel(model.lowerCurve.getAxisLabel(X))
        self.ax.set_ylabel(model.lowerCurve.getAxisLabel(Y))
        self.ax.set_yscale('log' if logy else 'linear')
        self.ax.set_xscale('log' if logx else 'linear')
        self.canvas.draw()
# endregion