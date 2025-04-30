import sys
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from Car_GUI import Ui_Form
from QuarterCarModel import CarController

class MainWindow(qtw.QWidget, Ui_Form):
    def __init__(self):
        """
        Main window constructor.
        """
        super().__init__()
        self.setupUi(self)

        # Setup car controller
        input_widgets = (self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang,
                         self.le_tmax, self.chk_IncludeAccel)
        display_widgets = (self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel,
                           self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_Plot)

        self.controller = CarController((input_widgets, display_widgets))

        # Connect signals to slots
        self.btn_calculate.clicked.connect(self.controller.calculate)
        self.pb_Optimize.clicked.connect(self.doOptimize)
        self.chk_LogX.stateChanged.connect(self.controller.doPlot)
        self.chk_LogY.stateChanged.connect(self.controller.doPlot)
        self.chk_LogAccel.stateChanged.connect(self.controller.doPlot)
        self.chk_ShowAccel.stateChanged.connect(self.controller.doPlot)
        self.controller.setupEventFilter(self)
        self.controller.setupCanvasMoveEvent(self)
        self.show()

    def eventFilter(self, obj, event):
        """
        Overrides the default eventFilter of the widget.

        Args:
            obj: The object on which the event happened
            event: The event itself

        Returns:
            bool: Result from the parent widget
        """
        if obj == self.gv_Schematic.scene():
            et = event.type()
            if et == qtc.QEvent.GraphicsSceneMouseMove:
                scenePos = event.scenePos()
                strScene = "Mouse Position: x = {}, y = {}".format(round(scenePos.x(), 2), round(-scenePos.y(), 2))
                self.setWindowTitle(strScene)
            if event.type() == qtc.QEvent.GraphicsSceneWheel:
                zm = self.controller.getZoom()
                if event.delta() > 0:
                    zm += 0.1
                else:
                    zm -= 0.1
                zm = max(0.1, zm)
                self.controller.setZoom(zm)
        self.controller.updateSchematic()
        return super(MainWindow, self).eventFilter(obj, event)

    def mouseMoveEvent_Canvas(self, event):
        """
        Handle mouse move events on the canvas to update the display.

        Args:
            event: The mouse event
        """
        if event.inaxes:
            t_val = event.xdata
            if t_val is not None and self.controller.model is not None and self.controller.model.results is not None:
                self.controller.animate(t_val)
                ywheel, ybody, yroad, accel = self.controller.getPoints(t_val)
                self.setWindowTitle(
                    't={:0.2f}(ms), y-road:{:0.3f}(mm), y-wheel:{:0.2f}(mm), y-car:{:0.2f}(mm), accel={:0.2f}(g)'.format(
                        t_val * 1000, yroad * 1000, ywheel * 1000, ybody * 1000, accel
                    )
                )

    def doOptimize(self):
        """Optimize the suspension parameters."""
        app.setOverrideCursor(qtc.Qt.WaitCursor)
        self.controller.OptimizeSuspension()
        app.restoreOverrideCursor()

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    mw.setWindowTitle('Quarter Car Model')
    sys.exit(app.exec())