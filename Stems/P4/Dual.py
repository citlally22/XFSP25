from Air import *
import numpy as np
from PyQt5 import QtWidgets as qtw
import matplotlib.pyplot as plt

# ---------------------
# Dual Cycle Model (no .copy())
# ---------------------
class dualCycleModel():
    def __init__(self, p_initial=1E5, v_cylinder=3E-3, t_initial=300, pressure_ratio=1.5, cutoff=1.2, ratio=18.0, name='Air Standard Dual Cycle'):
        self.units = units()
        self.units.SI = False
        self.air = air()
        self.air.set(P=p_initial, T=t_initial)
        self.p_initial = p_initial
        self.T_initial = t_initial

        self.P_ratio = min(max(1.1, pressure_ratio), 5.0)
        self.Cutoff = min(max(1.1, cutoff), 4.0)
        self.Ratio = min(max(4.0, ratio), 25.0)
        self.V_Cylinder = v_cylinder

        self.air.n = self.V_Cylinder / self.air.State.v
        self.air.m = self.air.n * self.air.MW

        self.State1 = self.air.set(P=self.p_initial, T=self.T_initial)
        self.State1.name = "State 1"

        self.State2 = self.air.set(v=self.State1.v / self.Ratio, s=self.State1.s)
        self.State2.name = "State 2"

        self.State3 = self.air.set(P=self.State2.P * self.P_ratio, v=self.State2.v)
        self.State3.name = "State 3"

        self.State4 = self.air.set(P=self.State3.P, v=self.State3.v * self.Cutoff)
        self.State4.name = "State 4"

        self.State5 = self.air.set(v=self.State1.v, s=self.State4.s)
        self.State5.name = "State 5"

        self.W_Compression = self.State2.u - self.State1.u
        self.W_Power = self.State4.u - self.State5.u
        self.Q_In = (self.State3.u - self.State2.u) + (self.State4.h - self.State3.h)
        self.Q_Out = self.State5.u - self.State1.u
        self.W_Cycle = self.W_Power - self.W_Compression
        self.Eff = max(0.0, min(100.0, 100.0 * self.W_Cycle / self.Q_In)) if self.Q_In != 0 else 0.0

        self.upperCurve = StateDataForPlotting()
        self.lowerCurve = StateDataForPlotting()
        self.calculated = True
        self.cycleType = 'dual'

    def getSI(self):
        return self.units.SI


# ---------------------
# Dual Cycle Controller
# ---------------------
class dualCycleController():
    def __init__(self, model=None, ax=None):
        self.model = dualCycleModel() if model is None else model
        self.view = dualCycleView()
        self.view.ax = ax

    def calc(self):
        T0 = float(self.view.le_TLow.text())
        P0 = float(self.view.le_P0.text())
        V0 = float(self.view.le_V0.text())
        CR = float(self.view.le_CR.text())
        rc = float(self.view.le_THigh.text())  # THigh used as cutoff
        P3overP2 = 1.5
        metric = self.view.rdo_Metric.isChecked()
        self.set(T_0=T0, P_0=P0, V_0=V0, rc=rc, P3_P2=P3overP2, ratio=CR, SI=metric)

    def set(self, T_0=300.0, P_0=100000.0, V_0=0.002, rc=1.2, P3_P2=1.5, ratio=18.0, SI=True):
        self.model.units.set(SI=SI)
        self.model.T_initial = T_0 if SI else T_0 / self.model.units.CF_T
        self.model.p_initial = P_0 if SI else P_0 / self.model.units.CF_P
        self.model.Cutoff = rc
        self.model.P_ratio = P3_P2
        self.model.V_Cylinder = V_0 if SI else V_0 / self.model.units.CF_V
        self.model.Ratio = ratio

        self.model.__init__(p_initial=self.model.p_initial,
                            v_cylinder=self.model.V_Cylinder,
                            t_initial=self.model.T_initial,
                            pressure_ratio=self.model.P_ratio,
                            cutoff=self.model.Cutoff,
                            ratio=self.model.Ratio)
        self.model.calculated = True
        self.buildDataForPlotting()
        self.updateView()

    def buildDataForPlotting(self):
        self.model.upperCurve.clear()
        self.model.lowerCurve.clear()
        a = air()

        for T in np.linspace(self.model.State2.T, self.model.State3.T, 30):
            s = a.set(T=T, v=self.model.State2.v)
            self.model.upperCurve.add((s.T, s.P, s.u, s.h, s.s, s.v))
        for v in np.linspace(self.model.State3.v, self.model.State4.v, 30):
            s = a.set(P=self.model.State3.P, v=v)
            self.model.upperCurve.add((s.T, s.P, s.u, s.h, s.s, s.v))
        for v in np.linspace(self.model.State4.v, self.model.State5.v, 30):
            s = a.set(v=v, s=self.model.State4.s)
            self.model.upperCurve.add((s.T, s.P, s.u, s.h, s.s, s.v))
        for T in np.linspace(self.model.State5.T, self.model.State1.T, 30):
            s = a.set(T=T, v=self.model.State5.v)
            self.model.upperCurve.add((s.T, s.P, s.u, s.h, s.s, s.v))
        for v in np.linspace(self.model.State1.v, self.model.State2.v, 30):
            s = a.set(v=v, s=self.model.State1.s)
            self.model.lowerCurve.add((s.T, s.P, s.u, s.h, s.s, s.v))

    def updateView(self):
        self.view.updateView(cycle=self.model)

    def setWidgets(self, w=None):
        [self.view.lbl_THigh, self.view.lbl_TLow, self.view.lbl_P0, self.view.lbl_V0, self.view.lbl_CR,
         self.view.le_THigh, self.view.le_TLow, self.view.le_P0, self.view.le_V0, self.view.le_CR,
         self.view.le_T1, self.view.le_T2, self.view.le_T3, self.view.le_T4,
         self.view.lbl_T1Units, self.view.lbl_T2Units, self.view.lbl_T3Units, self.view.lbl_T4Units,
         self.view.le_PowerStroke, self.view.le_CompressionStroke, self.view.le_HeatAdded, self.view.le_Efficiency,
         self.view.lbl_PowerStrokeUnits, self.view.lbl_CompressionStrokeUnits, self.view.lbl_HeatInUnits,
         self.view.rdo_Metric, self.view.cmb_Abcissa, self.view.cmb_Ordinate,
         self.view.chk_LogAbcissa, self.view.chk_LogOrdinate, self.view.ax, self.view.canvas] = w


# ---------------------
# Dual Cycle View
# ---------------------
class dualCycleView():
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
        if cycle.calculated:
            self.plot_cycle_XY(cycle, X=self.cmb_Abcissa.currentText(), Y=self.cmb_Ordinate.currentText(),
                               logx=self.chk_LogAbcissa.isChecked(), logy=self.chk_LogOrdinate.isChecked(),
                               mass=False, total=True)

    def plot_cycle_XY(self, cycle, X='v', Y='P', logx=False, logy=False, mass=False, total=False):
        if X == Y:
            return
        ax = self.ax
        ax.clear()
        ax.set_xscale('log' if logx else 'linear')
        ax.set_yscale('log' if logy else 'linear')

        def get(col, curve): return [s.getVal(col) for s in curve.data]

        XdataLC = get(X, cycle.lowerCurve)
        YdataLC = get(Y, cycle.lowerCurve)
        XdataUC = get(X, cycle.upperCurve)
        YdataUC = get(Y, cycle.upperCurve)

        ax.plot(XdataLC, YdataLC, color='k')
        ax.plot(XdataUC, YdataUC, color='g')
        ax.set_xlabel(X)
        ax.set_ylabel(Y)
        ax.set_title("Dual Cycle")
        for state in [cycle.State1, cycle.State2, cycle.State3, cycle.State4, cycle.State5]:
            ax.plot(state.getVal(X), state.getVal(Y), marker='o', markerfacecolor='w', markeredgecolor='k')
        self.canvas.draw()
