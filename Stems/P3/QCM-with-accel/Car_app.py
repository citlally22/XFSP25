"""
Main application file for the Quarter Car Model GUI.

This module sets up the main window, integrates the UI form, and connects the
CarController for simulation, visualization, and optimization of a quarter car model.
It handles user interactions such as mouse movements, zooming, and button clicks.
"""

import sys
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from Car_GUI import Ui_Form  # Generated UI form from Qt Designer
from QuarterCarModel import CarController  # Controller for the quarter car model


class MainWindow(qtw.QWidget, Ui_Form):
    """
    Main window class for the Quarter Car Model application.

    Inherits from QWidget and the generated Ui_Form to create the GUI.
    Manages user interactions, connects signals to slots, and integrates the
    CarController for simulation and optimization.

    Attributes:
        controller (CarController): Controller instance managing the model and view
    """

    def __init__(self):
        """
        Initialize the main window and set up the UI.

        Sets up the UI using the generated form, initializes the CarController,
        connects signals to slots for user interactions, and displays the window.
        """
        super().__init__()
        self.setupUi(self)  # Initialize the UI from Ui_Form

        # Setup CarController with input and display widgets
        input_widgets = (
            self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang,
            self.le_tmax, self.chk_IncludeAccel
        )
        display_widgets = (
            self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel,
            self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_Plot
        )

        self.controller = CarController((input_widgets, display_widgets))  # Initialize the controller

        # Connect signals to slots for user interactions
        self.btn_calculate.clicked.connect(self.controller.calculate)  # Connect calculate button
        self.pb_Optimize.clicked.connect(self.doOptimize)  # Connect optimize button
        self.chk_LogX.stateChanged.connect(self.controller.doPlot)  # Update plot on log X change
        self.chk_LogY.stateChanged.connect(self.controller.doPlot)  # Update plot on log Y change
        self.chk_LogAccel.stateChanged.connect(self.controller.doPlot)  # Update plot on log accel change
        self.chk_ShowAccel.stateChanged.connect(self.controller.doPlot)  # Update plot on show accel change
        self.controller.setupEventFilter(self)  # Set up event filter for schematic interactions
        self.controller.setupCanvasMoveEvent(self)  # Set up canvas mouse move events
        self.show()  # Display the window

    def eventFilter(self, obj, event):
        """
        Handle events for the graphics scene, such as mouse movements and wheel events.

        Updates the window title with the mouse position in the schematic and handles
        zooming with the mouse wheel.

        Args:
            obj: The object on which the event occurred
            event: The event object

        Returns:
            bool: Result from the parent widget's event filter
        """
        if obj == self.gv_Schematic.scene():
            et = event.type()
            # Handle mouse movement in the schematic
            if et == qtc.QEvent.GraphicsSceneMouseMove:
                scenePos = event.scenePos()
                # Update window title with mouse position (inverted y for display)
                strScene = "Mouse Position: x = {}, y = {}".format(
                    round(scenePos.x(), 2), round(-scenePos.y(), 2)
                )
                self.setWindowTitle(strScene)
            # Handle mouse wheel for zooming
            if event.type() == qtc.QEvent.GraphicsSceneWheel:
                zm = self.controller.getZoom()  # Get current zoom level
                # Adjust zoom based on wheel direction
                if event.delta() > 0:
                    zm += 0.1  # Zoom in
                else:
                    zm -= 0.1  # Zoom out
                zm = max(0.1, zm)  # Ensure zoom doesn't go below 0.1
                self.controller.setZoom(zm)  # Apply new zoom level
        self.controller.updateSchematic()  # Update the schematic display
        return super(MainWindow, self).eventFilter(obj, event)

    def mouseMoveEvent_Canvas(self, event):
        """
        Handle mouse move events on the matplotlib canvas.

        Updates the schematic animation and window title based on the mouse position
        over the plot, showing the corresponding time, positions, and acceleration.

        Args:
            event: The mouse event from matplotlib
        """
        if event.inaxes:  # Check if the mouse is over the plot axes
            t_val = event.xdata  # Get the time value from the x-coordinate
            # Ensure the time value and model data are available
            if t_val is not None and self.controller.model is not None and self.controller.model.results is not None:
                self.controller.animate(t_val)  # Animate the schematic at this time
                # Get positions and acceleration at this time
                ywheel, ybody, yroad, accel = self.controller.getPoints(t_val)
                # Update window title with simulation data (converted to mm and ms for display)
                self.setWindowTitle(
                    't={:0.2f}(ms), y-road:{:0.3f}(mm), y-wheel:{:0.2f}(mm), y-car:{:0.2f}(mm), accel={:0.2f}(g)'.format(
                        t_val * 1000,  # Convert seconds to milliseconds
                        yroad * 1000,  # Convert meters to millimeters
                        ywheel * 1000,  # Convert meters to millimeters
                        ybody * 1000,   # Convert meters to millimeters
                        accel
                    )
                )

    def doOptimize(self):
        """
        Trigger the optimization of suspension parameters.

        Changes the cursor to a wait cursor during optimization and restores it afterward.
        Calls the controller's OptimizeSuspension method to perform the optimization.
        """
        app.setOverrideCursor(qtc.Qt.WaitCursor)  # Show wait cursor during optimization
        self.controller.OptimizeSuspension()  # Perform optimization
        app.restoreOverrideCursor()  # Restore normal cursor


if __name__ == '__main__':
    """Entry point for the application."""
    app = qtw.QApplication(sys.argv)  # Create the QApplication instance
    mw = MainWindow()  # Create the main window
    mw.setWindowTitle('Quarter Car Model')  # Set the initial window title
    sys.exit(app.exec())  # Start the application event loop