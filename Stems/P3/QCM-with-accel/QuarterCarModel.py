"""
Quarter Car Model GUI implementation with detailed schematic.

This module provides a simulation of a quarter car model, including a GUI for input,
simulation, optimization, and visualization of results. It uses PyQt5 for the GUI,
scipy for numerical integration and optimization, and matplotlib for plotting.
The schematic includes custom graphical items for the car body, wheel, springs, dashpot, and road.
"""

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import math
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
import logging

# These imports are necessary for drawing a matplotlib graph on the GUI
# No simple widget for this exists in QT Designer, so the widget is added in code
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

# Configure matplotlib to use a specific font and reduce font-related logging
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Set up logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#region class definitions
#region specialized graphic items
class MassBlock(qtw.QGraphicsItem):
    """
    Represents a rectangular mass block in the schematic (e.g., car body).

    Attributes:
        x (float): Center x-coordinate of the block
        y (float): Center y-coordinate of the block
        y0 (float): Initial y-coordinate
        pen (QPen): Pen for drawing the block outline
        brush (QBrush): Brush for filling the block
        width (float): Width of the block
        height (float): Height of the block
        rect (QRectF): Bounding rectangle of the block
        name (str): Name of the block (e.g., "CarBody")
        label (str): Optional label to display next to the block
        mass (float): Mass of the block (kg)
        transformation (QTransform): Transformation for positioning
    """

    def __init__(self, CenterX, CenterY, width=30, height=10, parent=None, pen=None, brush=None, name='CarBody', label=None, mass=10):
        """
        Initialize the mass block.

        Args:
            CenterX (float): Center x-coordinate
            CenterY (float): Center y-coordinate
            width (float): Width of the block
            height (float): Height of the block
            parent (QGraphicsItem, optional): Parent item
            pen (QPen, optional): Pen for drawing the outline
            brush (QBrush, optional): Brush for filling the block
            name (str): Name of the block
            label (str, optional): Label to display next to the block
            mass (float): Mass of the block (kg)
        """
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.y0 = self.y  # Store initial y-coordinate
        self.pen = pen
        self.brush = brush
        self.width = width
        self.height = height
        self.top = self.y - self.height / 2  # Top edge of the block
        self.left = self.x - self.width / 2  # Left edge of the block
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)  # Bounding rectangle
        self.name = name
        self.label = label
        self.mass = mass
        self.transformation = qtg.QTransform()  # Transformation for positioning
        # Set tooltip with position and mass information
        stTT = self.name + "\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)

    def setMass(self, mass=None):
        """
        Update the mass of the block and refresh the tooltip.

        Args:
            mass (float, optional): New mass value (kg)
        """
        if mass is not None:
            self.mass = mass
            stTT = self.name + "\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
            self.setToolTip(stTT)

    def boundingRect(self):
        """
        Return the bounding rectangle of the block after transformation.

        Returns:
            QRectF: Transformed bounding rectangle
        """
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def paint(self, painter, option, widget=None):
        """
        Paint the mass block on the canvas.

        Args:
            painter (QPainter): Painter object for drawing
            option (QStyleOptionGraphicsItem): Style options
            widget (QWidget, optional): Widget being painted on
        """
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)
        if self.brush is not None:
            painter.setBrush(self.brush)
        # Recalculate the rectangle position relative to the center
        self.top = -self.height / 2
        self.left = -self.width / 2
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)
        painter.drawRect(self.rect)
        # Draw mass label inside the block
        font = painter.font()
        font.setPointSize(6)
        painter.setFont(font)
        text = "mass = {:0.1f} kg".format(self.mass)
        fm = qtg.QFontMetrics(painter.font())
        painter.drawText(qtc.QPointF(-fm.width(text) / 2.0, fm.height() / 2.0), text)
        # Draw optional label next to the block
        if self.label is not None:
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(qtc.QPointF((self.width / 2.0) + 10, 0), self.label)
        # Apply transformation to position the block
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        self.transformation.reset()

class Wheel(qtw.QGraphicsItem):
    """
    Represents a wheel in the schematic, including a mass block.

    Attributes:
        x (float): Center x-coordinate of the wheel
        road_y (float): Y-coordinate of the road
        radius (float): Radius of the wheel
        y (float): Center y-coordinate of the wheel
        y0 (float): Initial y-coordinate
        penTire (QPen): Pen for drawing the tire outline
        brushWheel (QBrush): Brush for filling the wheel
        penMass (QPen): Pen for drawing the wheel mass block
        brushMass (QBrush): Brush for filling the wheel mass block
        rect (QRectF): Bounding rectangle of the wheel
        name (str): Name of the wheel
        mass (float): Mass of the wheel (kg)
        transformation (QTransform): Transformation for positioning
        massBlock (MassBlock): Mass block representing the wheel's mass
    """

    def __init__(self, CenterX, radius=10, roady=10, parent=None, penTire=None, penMass=None, brushWheel=None, brushMass=None, name='Wheel', mass=10):
        """
        Initialize the wheel.

        Args:
            CenterX (float): Center x-coordinate
            radius (float): Radius of the wheel
            roady (float): Y-coordinate of the road
            parent (QGraphicsItem, optional): Parent item
            penTire (QPen, optional): Pen for drawing the tire
            penMass (QPen, optional): Pen for drawing the wheel mass
            brushWheel (QBrush, optional): Brush for filling the wheel
            brushMass (QBrush, optional): Brush for filling the wheel mass
            name (str): Name of the wheel
            mass (float): Mass of the wheel (kg)
        """
        super().__init__(parent)
        self.x = CenterX
        self.road_y = roady
        self.radius = radius
        self.y = self.road_y - self.radius  # Position the wheel on the road
        self.y0 = self.y  # Store initial y-coordinate
        self.penTire = penTire
        self.brushWheel = brushWheel
        self.penMass = penMass
        self.brushMass = brushMass
        self.rect = qtc.QRectF(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)  # Bounding rectangle
        self.name = name
        self.mass = mass
        self.transformation = qtg.QTransform()  # Transformation for positioning
        stTT = self.name + "\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)
        # Dimensions for the mass block inside the wheel
        self.massWidth = 2 * self.radius * 0.85
        self.massHeight = self.radius / 3
        # Create the mass block for the wheel
        self.massBlock = MassBlock(CenterX, self.y, width=self.massWidth, height=self.massHeight, pen=penMass, brush=brushMass, name="Wheel Mass", mass=mass)

    def setMass(self, mass=None):
        """
        Update the mass of the wheel and its mass block, and refresh the tooltip.

        Args:
            mass (float, optional): New mass value (kg)
        """
        if mass is not None:
            self.mass = mass
            self.massBlock.setMass(mass)
            stTT = self.name + "\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
            self.setToolTip(stTT)

    def boundingRect(self):
        """
        Return the bounding rectangle of the wheel after transformation.

        Returns:
            QRectF: Transformed bounding rectangle
        """
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def addToScene(self, scene):
        """
        Add the wheel and its mass block to the scene.

        Args:
            scene (QGraphicsScene): Scene to add the items to
        """
        scene.addItem(self)
        scene.addItem(self.massBlock)

    def paint(self, painter, option, widget=None):
        """
        Paint the wheel on the canvas.

        Args:
            painter (QPainter): Painter object for drawing
            option (QStyleOptionGraphicsItem): Style options
            widget (QWidget, optional): Widget being painted on
        """
        height = 2 * (self.road_y - self.y)  # Height of the wheel (distance from road to center)
        width = 2 * self.radius  # Width of the wheel
        if self.penTire is not None:
            painter.setPen(self.penTire)
        if self.brushWheel is not None:
            painter.setBrush(self.brushWheel)
        left = -width / 2.0
        top = self.y - height / 2
        self.rect = qtc.QRectF(left, top, width, height)  # Update bounding rectangle
        painter.drawEllipse(self.rect)  # Draw the wheel as an ellipse
        self.massBlock.y = self.y  # Update the position of the mass block
        point = qtc.QPointF(self.radius * 1.1, 0.0)
        painter.drawText(point, self.name)  # Draw the wheel's name

class LinearSpring(qtw.QGraphicsItem):
    """
    Represents a linear spring in the schematic.

    Attributes:
        stPt (QPointF): Starting point of the spring
        enPt (QPointF): Ending point of the spring
        freeLength (float): Free length of the spring
        DL (float): Change in length from free length
        centerPt (QPointF): Center point of the spring
        pen (QPen): Pen for drawing the spring
        coilsWidth (float): Width of the spring coils
        coilsLength (float): Length of the spring coils
        rect (QRectF): Bounding rectangle of the spring
        name (str): Name of the spring
        label (str, optional): Label to display next to the spring
        k (float): Spring constant (N/m)
        nCoils (int): Number of coils in the spring
        transformation (QTransform): Transformation for positioning
    """

    def __init__(self, ptSt, ptEn, coilsWidth=10, coilsLength=30, parent=None, pen=None, name='Spring', label=None, k=10, nCoils=6):
        """
        Initialize the linear spring.

        Args:
            ptSt (QPointF): Starting point of the spring
            ptEn (QPointF): Ending point of the spring
            coilsWidth (float): Width of the spring coils
            coilsLength (float): Length of the spring coils
            parent (QGraphicsItem, optional): Parent item
            pen (QPen, optional): Pen for drawing the spring
            name (str): Name of the spring
            label (str, optional): Label to display next to the spring
            k (float): Spring constant (N/m)
            nCoils (int): Number of coils in the spring
        """
        super().__init__(parent)
        self.stPt = ptSt
        self.enPt = ptEn
        self.freeLength = self.getLength()  # Calculate initial length
        self.DL = self.length - self.freeLength  # Change in length
        self.centerPt = (self.stPt + self.enPt) / 2.0  # Center point
        self.pen = pen
        self.coilsWidth = coilsWidth
        self.coilsLength = coilsLength
        self.top = -self.coilsLength / 2
        self.left = -self.coilsWidth / 2
        self.rect = qtc.QRectF(self.left, self.top, self.coilsWidth, self.coilsLength)  # Bounding rectangle
        self.name = name
        self.label = label
        self.k = k
        self.nCoils = nCoils
        self.transformation = qtg.QTransform()  # Transformation for positioning
        stTT = self.name + "\nx={:0.1f}, y={:0.1f}\nk = {:0.1f}".format(self.centerPt.x(), self.centerPt.y(), self.k)
        self.setToolTip(stTT)

    def setk(self, k=None):
        """
        Update the spring constant and refresh the tooltip.

        Args:
            k (float, optional): New spring constant (N/m)
        """
        if k is not None:
            self.k = k
            stTT = self.name + "\nx={:0.3f}, y={:0.3f}\nk = {:0.3f}".format(self.stPt.x(), self.stPt.y(), self.k)
            self.setToolTip(stTT)

    def boundingRect(self):
        """
        Return the bounding rectangle of the spring after transformation.

        Returns:
            QRectF: Transformed bounding rectangle
        """
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def getLength(self):
        """
        Calculate the current length of the spring.

        Returns:
            float: Length of the spring
        """
        p = self.enPt - self.stPt
        self.length = math.sqrt(p.x() ** 2 + p.y() ** 2)
        return self.length

    def getDL(self):
        """
        Calculate the change in length from the free length.

        Returns:
            float: Change in length
        """
        self.DL = self.length - self.freeLength
        return self.DL

    def getAngleDeg(self):
        """
        Calculate the angle of the spring in degrees.

        Returns:
            float: Angle in degrees
        """
        p = self.enPt - self.stPt
        self.angleRad = math.atan2(p.y(), p.x())
        self.angleDeg = 180.0 / math.pi * self.angleRad
        return self.angleDeg

    def paint(self, painter, option, widget=None):
        """
        Paint the spring on the canvas.

        Args:
            painter (QPainter): Painter object for drawing
            option (QStyleOptionGraphicsItem): Style options
            widget (QWidget, optional): Widget being painted on
        """
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)
        self.getLength()
        self.getAngleDeg()
        self.getDL()
        ht = self.coilsWidth  # Height of the spring coils
        wd = self.coilsLength + self.DL  # Width adjusted by change in length
        top = -ht / 2
        left = -wd / 2
        right = wd / 2
        self.rect = qtc.QRectF(left, top, wd, ht)  # Update bounding rectangle
        # Draw the spring: straight lines at ends, zigzag in the middle
        painter.drawLine(qtc.QPointF(left, 0), qtc.QPointF(left, ht / 2))
        dX = wd / self.nCoils
        for i in range(self.nCoils):
            painter.drawLine(qtc.QPointF(left + i * dX, ht / 2), qtc.QPointF(left + (i + 0.5) * dX, -ht / 2))
            painter.drawLine(qtc.QPointF(left + (i + 0.5) * dX, -ht / 2), qtc.QPointF(left + (i + 1) * dX, ht / 2))
        painter.drawLine(qtc.QPointF(right, ht / 2), qtc.QPointF(right, 0))
        # Draw connecting lines to the start and end points
        painter.drawLine(qtc.QPointF(-self.length / 2, 0), qtc.QPointF(left, 0))
        painter.drawLine(qtc.QPointF(right, 0), qtc.QPointF(self.length / 2, 0))
        # Draw nodes at the ends
        nodeRad = 2
        stRec = qtc.QRectF(-self.length / 2 - nodeRad, -nodeRad, 2 * nodeRad, 2 * nodeRad)
        enRec = qtc.QRectF(self.length / 2 - nodeRad, -nodeRad, 2 * nodeRad, 2 * nodeRad)
        painter.drawEllipse(stRec)
        painter.drawEllipse(enRec)
        # Apply transformation to position and rotate the spring
        self.transformation.translate(self.stPt.x(), self.stPt.y())
        self.transformation.rotate(self.angleDeg)
        self.transformation.translate(self.length / 2, 0)
        self.setTransform(self.transformation)
        self.transformation.reset()

class DashPot(qtw.QGraphicsItem):
    """
    Represents a dashpot (shock absorber) in the schematic.

    Attributes:
        stPt (QPointF): Starting point of the dashpot
        enPt (QPointF): Ending point of the dashpot
        freeLength (float): Free length of the dashpot
        DL (float): Change in length from free length
        centerPt (QPointF): Center point of the dashpot
        pen (QPen): Pen for drawing the dashpot
        dpWidth (float): Width of the dashpot
        dpLength (float): Length of the dashpot
        rect (QRectF): Bounding rectangle of the dashpot
        name (str): Name of the dashpot
        label (str, optional): Label to display next to the dashpot
        c (float): Damping coefficient (N·s/m)
        transformation (QTransform): Transformation for positioning
    """

    def __init__(self, ptSt, ptEn, dpWidth=10, dpLength=30, parent=None, pen=None, name='Dashpot', label=None, c=10):
        """
        Initialize the dashpot.

        Args:
            ptSt (QPointF): Starting point of the dashpot
            ptEn (QPointF): Ending point of the dashpot
            dpWidth (float): Width of the dashpot
            dpLength (float): Length of the dashpot
            parent (QGraphicsItem, optional): Parent item
            pen (QPen, optional): Pen for drawing the dashpot
            name (str): Name of the dashpot
            label (str, optional): Label to display next to the dashpot
            c (float): Damping coefficient (N·s/m)
        """
        super().__init__(parent)
        self.stPt = ptSt
        self.enPt = ptEn
        self.freeLength = self.getLength()  # Calculate initial length
        self.DL = self.length - self.freeLength  # Change in length
        self.centerPt = (self.stPt + self.enPt) / 2.0  # Center point
        self.pen = pen
        self.dpWidth = dpWidth
        self.dpLength = dpLength
        self.top = -self.dpLength / 2
        self.left = -self.dpWidth / 2
        self.rect = qtc.QRectF(self.left, self.top, self.dpWidth, self.dpLength)  # Bounding rectangle
        self.name = name
        self.label = label
        self.c = c
        self.transformation = qtg.QTransform()  # Transformation for positioning
        stTT = self.name + "\nx={:0.1f}, y={:0.1f}\nc = {:0.1f}".format(self.centerPt.x(), self.centerPt.y(), self.c)
        self.setToolTip(stTT)

    def setc(self, c=None):
        """
        Update the damping coefficient and refresh the tooltip.

        Args:
            c (float, optional): New damping coefficient (N·s/m)
        """
        if c is not None:
            self.c = c
            stTT = self.name + "\nx={:0.3f}, y={:0.3f}\nc = {:0.3f}".format(self.stPt.x(), self.stPt.y(), self.c)
            self.setToolTip(stTT)

    def boundingRect(self):
        """
        Return the bounding rectangle of the dashpot after transformation.

        Returns:
            QRectF: Transformed bounding rectangle
        """
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def getLength(self):
        """
        Calculate the current length of the dashpot.

        Returns:
            float: Length of the dashpot
        """
        p = self.enPt - self.stPt
        self.length = math.sqrt(p.x() ** 2 + p.y() ** 2)
        return self.length

    def getDL(self):
        """
        Calculate the change in length from the free length.

        Returns:
            float: Change in length
        """
        self.DL = self.length - self.freeLength
        return self.DL

    def getAngleDeg(self):
        """
        Calculate the angle of the dashpot in degrees.

        Returns:
            float: Angle in degrees
        """
        p = self.enPt - self.stPt
        self.angleRad = math.atan2(p.y(), p.x())
        self.angleDeg = 180.0 / math.pi * self.angleRad
        return self.angleDeg

    def paint(self, painter, option, widget=None):
        """
        Paint the dashpot on the canvas.

        Args:
            painter (QPainter): Painter object for drawing
            option (QStyleOptionGraphicsItem): Style options
            widget (QWidget, optional): Widget being painted on
        """
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)
        self.getLength()
        self.getAngleDeg()
        self.getDL()
        ht = self.dpWidth  # Height of the dashpot
        wd = self.dpLength  # Width of the dashpot
        top = -ht / 2
        left = -wd / 2
        right = wd / 2
        self.rect = qtc.QRectF(left, top, wd, ht)  # Update bounding rectangle
        # Draw the dashpot: a rectangle with a piston
        painter.drawLine(qtc.QPointF(left, -ht / 2), qtc.QPointF(left, ht / 2))
        painter.drawLine(qtc.QPointF(left, -ht / 2), qtc.QPointF(right, -ht / 2))
        painter.drawLine(qtc.QPointF(left, ht / 2), qtc.QPointF(right, ht / 2))
        painter.drawLine(qtc.QPointF(self.DL, ht / 2 * 0.95), qtc.QPointF(self.DL, -ht / 2 * 0.95))
        # Draw connecting lines to the start and end points
        painter.drawLine(qtc.QPointF(-self.length / 2, 0), qtc.QPointF(left, 0))
        painter.drawLine(qtc.QPointF(self.DL, 0), qtc.QPointF(self.length / 2, 0))
        # Draw nodes at the ends
        nodeRad = 2
        stRec = qtc.QRectF(-self.length / 2 - nodeRad, -nodeRad, 2 * nodeRad, 2 * nodeRad)
        enRec = qtc.QRectF(self.length / 2 - nodeRad, -nodeRad, 2 * nodeRad, 2 * nodeRad)
        painter.drawEllipse(stRec)
        painter.drawEllipse(enRec)
        # Apply transformation to position and rotate the dashpot
        self.transformation.translate(self.stPt.x(), self.stPt.y())
        self.transformation.rotate(self.angleDeg)
        self.transformation.translate(self.length / 2, 0)
        self.setTransform(self.transformation)
        self.transformation.reset()

class Road(qtw.QGraphicsItem):
    """
    Represents the road surface in the schematic.

    Attributes:
        x (float): Center x-coordinate of the road
        y (float): Center y-coordinate of the road
        x0 (float): Initial x-coordinate
        y0 (float): Initial y-coordinate
        pen (QPen): Pen for drawing the road outline
        brush (QBrush): Brush for filling the road
        width (float): Width of the road
        height (float): Height of the road
        rect (QRectF): Bounding rectangle of the road
        name (str): Name of the road
        label (str, optional): Label to display next to the road
        transformation (QTransform): Transformation for positioning
    """

    def __init__(self, x, y, width=30, height=10, parent=None, pen=None, brush=None, name='Road', label=None):
        """
        Initialize the road.

        Args:
            x (float): Center x-coordinate
            y (float): Center y-coordinate
            width (float): Width of the road
            height (float): Height of the road
            parent (QGraphicsItem, optional): Parent item
            pen (QPen, optional): Pen for drawing the road
            brush (QBrush, optional): Brush for filling the road
            name (str): Name of the road
            label (str, optional): Label to display next to the road
        """
        super().__init__(parent)
        self.x = x
        self.y = y
        self.x0 = x  # Store initial x-coordinate
        self.y0 = y  # Store initial y-coordinate
        self.pen = pen
        self.brush = brush
        self.width = width
        self.height = height
        self.top = self.y - self.height / 2  # Top edge of the road
        self.left = self.x - self.width / 2  # Left edge of the road
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)  # Bounding rectangle
        self.name = name
        self.label = label
        self.transformation = qtg.QTransform()  # Transformation for positioning

    def boundingRect(self):
        """
        Return the bounding rectangle of the road after transformation.

        Returns:
            QRectF: Transformed bounding rectangle
        """
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def paint(self, painter, option, widget=None):
        """
        Paint the road on the canvas.

        Args:
            painter (QPainter): Painter object for drawing
            option (QStyleOptionGraphicsItem): Style options
            widget (QWidget, optional): Widget being painted on
        """
        if self.pen is not None:
            painter.setPen(self.pen)
        if self.brush is not None:
            painter.setBrush(self.brush)
        self.top = self.y
        self.left = self.x - self.width / 2
        self.right = self.x + self.width / 2
        # Draw the road as a line
        painter.drawLine(qtc.QPointF(self.left, self.top), qtc.QPointF(self.right, self.top))
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)  # Update bounding rectangle
        # Fill the area below the road
        penOutline = qtg.QPen(qtc.Qt.NoPen)
        painter.setPen(penOutline)
        painter.setBrush(self.brush)
        painter.drawRect(self.rect)

#endregion

#region MVC for quarter car model
class CarModel:
    """
    Represents the quarter car model and its dynamics.

    Attributes:
        results (object): Results from solve_ivp
        roadPosData (ndarray): Road position data over time
        wheelPosData (ndarray): Wheel position data over time
        bodyPosData (ndarray): Car body position data over time
        bodyAccelData (ndarray): Car body acceleration data over time (g)
        springForceData (ndarray): Spring force data over time (N)
        dashpotForceData (ndarray): Dashpot force data over time (N)
        tmax (float): Maximum simulation time (s)
        timeData (ndarray): Time points for simulation
        tramp (float): Time to reach the ramp height (s)
        angrad (float): Ramp angle (radians)
        ymag (float): Ramp height (m)
        yangdeg (float): Ramp angle (degrees)
        m1 (float): Car body mass (kg)
        m2 (float): Wheel mass (kg)
        c1 (float): Damping coefficient (N·s/m)
        k1 (float): Suspension spring constant (N/m)
        k2 (float): Tire spring constant (N/m)
        v (float): Car speed (km/h)
        mink1 (float): Minimum allowable k1 (N/m)
        maxk1 (float): Maximum allowable k1 (N/m)
        mink2 (float): Minimum allowable k2 (N/m)
        maxk2 (float): Maximum allowable k2 (N/m)
        accelMax (float): Maximum body acceleration (g)
        accelLim (float): Acceleration limit for optimization (g)
        SSE (float): Sum of squared errors for optimization
    """

    def __init__(self):
        """Initialize the car model with default parameters."""
        self.results = []  # Placeholder for solve_ivp results
        self.roadPosData = []  # Road position data
        self.wheelPosData = []  # Wheel position data
        self.bodyPosData = []  # Car body position data
        self.bodyAccelData = []  # Car body acceleration data
        self.springForceData = []  # Spring force data
        self.dashpotForceData = []  # Dashpot force data
        self.tmax = 3.0  # Maximum simulation time
        self.timeData = np.linspace(0, self.tmax, 2000)  # Initial time points
        self.tramp = 1.0  # Time to reach ramp height
        self.angrad = 0.1  # Ramp angle in radians
        self.ymag = 6.0 / (12 * 3.3)  # Ramp height (m)
        self.yangdeg = 45.0  # Ramp angle in degrees
        self.results = None  # Will store solve_ivp results
        self.m1 = 450  # Car body mass
        self.m2 = 20  # Wheel mass
        self.c1 = 4500  # Damping coefficient
        self.k1 = 15000  # Suspension spring constant
        self.k2 = 90000  # Tire spring constant
        self.v = 120.0  # Car speed
        # Calculate bounds for k1 and k2 based on static deflection
        self.mink1 = (self.m1 * 9.81) / (6.0 * 25.4 / 1000.0)
        self.maxk1 = (self.m1 * 9.81) / (3.0 * 25.4 / 1000.0)
        self.mink2 = ((self.m1 + self.m2) * 9.81) / (1.5 * 25.4 / 1000.0)
        self.maxk2 = ((self.m1 + self.m2) * 9.81) / (0.75 * 25.4 / 1000.0)
        self.accelBodyData = None  # Will store acceleration data
        self.accelMax = 0  # Maximum acceleration
        self.accelLim = 1.5  # Acceleration limit for optimization
        self.SSE = 0.0  # Sum of squared errors

class CarView:
    """
    The view component of the MVC pattern, handling the GUI display.

    Attributes:
        input_widgets (tuple): Input widgets (line edits, checkbox)
        display_widgets (tuple): Display widgets (graphics view, checkboxes, label, layout)
        le_m1 (QLineEdit): Car body mass input
        le_v (QLineEdit): Car speed input
        le_k1 (QLineEdit): Suspension spring constant input
        le_c1 (QLineEdit): Damping coefficient input
        le_m2 (QLineEdit): Wheel mass input
        le_k2 (QLineEdit): Tire spring constant input
        le_ang (QLineEdit): Ramp angle input
        le_tmax (QLineEdit): Maximum time input
        chk_IncludeAccel (QCheckBox): Checkbox for including acceleration in optimization
        gv_Schematic (QGraphicsView): Graphics view for the schematic
        chk_LogX (QCheckBox): Checkbox for log scale on x-axis
        chk_LogY (QCheckBox): Checkbox for log scale on y-axis (displacement)
        chk_LogAccel (QCheckBox): Checkbox for log scale on acceleration axis
        chk_ShowAccel (QCheckBox): Checkbox for showing acceleration plot
        lbl_MaxMinInfo (QLabel): Label for displaying max/min info
        layout_Plot (QVBoxLayout): Layout for the plot area
        tabs (QTabWidget): Tab widget for displacement and forces plots
        displacement_widget (QWidget): Widget for the displacement tab
        displacement_layout (QVBoxLayout): Layout for the displacement tab
        displacement_figure (Figure): Matplotlib figure for displacement plot
        displacement_canvas (FigureCanvasQTAgg): Canvas for displacement plot
        forces_widget (QWidget): Widget for the forces tab
        forces_layout (QVBoxLayout): Layout for the forces tab
        forces_figure (Figure): Matplotlib figure for forces plot
        forces_canvas (FigureCanvasQTAgg): Canvas for forces plot
        ax (Axes): Axes for the displacement plot
        ax1 (Axes): Secondary axes for acceleration plot
        forces_ax (Axes): Axes for the forces plot
        interpolatorWheel (callable): Interpolator for wheel position
        interpolatorBody (callable): Interpolator for body position
        interpolatorAccel (callable): Interpolator for acceleration
        posWheelTracer (Line2D): Tracer for wheel position on plot
        posBodyTracer (Line2D): Tracer for body position on plot
        posRoadTracer (Line2D): Tracer for road position on plot
        accelTracer (Line2D): Tracer for acceleration on plot
        scene (QGraphicsScene): Scene for the schematic
        Road (Road): Road item in the schematic
        Wheel (Wheel): Wheel item in the schematic
        CarBody (MassBlock): Car body item in the schematic
        spring1 (LinearSpring): Suspension spring item
        spring2 (LinearSpring): Tire spring item
        dashpot (DashPot): Dashpot item
        scale (float): Scaling factor for schematic animation
    """

    def __init__(self, args):
        """
        Initialize the view with input and display widgets.

        Args:
            args (tuple): Tuple containing input_widgets and display_widgets
        """
        self.input_widgets, self.display_widgets = args
        # Unpack input widgets
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
        self.le_tmax, self.chk_IncludeAccel = self.input_widgets
        # Unpack display widgets
        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_Plot = self.display_widgets

        # Create a QTabWidget for displacement and forces plots
        self.tabs = qtw.QTabWidget()
        self.layout_Plot.addWidget(self.tabs)

        # Displacement tab setup
        self.displacement_widget = qtw.QWidget()
        self.displacement_layout = qtw.QVBoxLayout(self.displacement_widget)
        self.displacement_figure = Figure(tight_layout=True, frameon=True, facecolor='none')
        self.displacement_canvas = FigureCanvasQTAgg(self.displacement_figure)
        self.displacement_layout.addWidget(NavigationToolbar2QT(self.displacement_canvas))
        self.displacement_layout.addWidget(self.displacement_canvas)
        self.tabs.addTab(self.displacement_widget, "Displacement vs. Time")

        # Forces tab setup
        self.forces_widget = qtw.QWidget()
        self.forces_layout = qtw.QVBoxLayout(self.forces_widget)
        self.forces_figure = Figure(tight_layout=True, frameon=True, facecolor='none')
        self.forces_canvas = FigureCanvasQTAgg(self.forces_figure)
        self.forces_layout.addWidget(NavigationToolbar2QT(self.forces_canvas))
        self.forces_layout.addWidget(self.forces_canvas)
        self.tabs.addTab(self.forces_widget, "Forces vs. Time")

        # Set up axes for the plots
        self.ax = self.displacement_figure.add_subplot()  # Axes for displacement
        self.ax1 = self.ax.twinx()  # Secondary axes for acceleration
        self.forces_ax = self.forces_figure.add_subplot()  # Axes for forces

        # Initialize interpolators and tracers
        self.interpolatorWheel = None
        self.interpolatorBody = None
        self.interpolatorAccel = None
        self.posWheelTracer = None
        self.posBodyTracer = None
        self.posRoadTracer = None
        self.accelTracer = None
        self.buildScene()  # Build the schematic

    def updateView(self, model=None, doPlot=True):
        """
        Update the GUI with the current model parameters and optionally plot.

        Args:
            model (CarModel, optional): Model instance to update from
            doPlot (bool): Whether to update the plots
        """
        # Update input fields with model parameters
        self.le_m1.setText("{:0.2f}".format(model.m1))
        self.le_k1.setText("{:0.2f}".format(model.k1))
        self.le_c1.setText("{:0.2f}".format(model.c1))
        self.le_m2.setText("{:0.2f}".format(model.m2))
        self.le_k2.setText("{:0.2f}".format(model.k2))
        self.le_ang.setText("{:0.2f}".format(model.yangdeg))
        self.le_tmax.setText("{:0.2f}".format(model.tmax))
        # Update max/min info label
        stTmp = "k1_min = {:0.2f}, k1_max = {:0.2f}\nk2_min = {:0.2f}, k2_max = {:0.2f}\n".format(
            model.mink1, model.maxk1, model.mink2, model.maxk2)
        stTmp += "SSE = {:0.2f}".format(model.SSE)
        self.lbl_MaxMinInfo.setText(stTmp)
        # Update schematic items with new masses and constants
        self.CarBody.setMass(model.m1)
        self.Wheel.setMass(model.m2)
        self.spring1.setk(model.k1)
        self.spring2.setk(model.k2)
        self.dashpot.setc(model.c1)
        if doPlot:
            self.doPlot(model)

    def buildScene(self):
        """Set up the schematic scene with graphical items."""
        self.scene = qtw.QGraphicsScene()
        self.scene.setObjectName("MyScene")
        self.scene.setSceneRect(-200, -200, 400, 400)  # Set scene dimensions
        self.gv_Schematic.setScene(self.scene)
        self.setupPensAndBrushes()  # Initialize pens and brushes
        # Create schematic items
        self.Road = Road(0, 100, 300, 10, pen=self.groundPen, brush=self.groundBrush)
        self.Wheel = Wheel(0, 50, roady=self.Road.y, penTire=self.penTire, brushWheel=self.brushWheel,
                           penMass=self.penMass, brushMass=self.brushMass, name="Wheel")
        self.CarBody = MassBlock(0, -70, 100, 30, pen=self.penMass, brush=self.brushMass, name="Car Body",
                                 label='Car Body', mass=150)
        self.spring1 = LinearSpring(qtc.QPointF(-35.0, self.Wheel.y), qtc.QPointF(-35, self.CarBody.y),
                                    pen=self.penMass, name='k1', label='Spring 1', k=10)
        self.spring2 = LinearSpring(qtc.QPointF(self.Wheel.x, self.Wheel.y), qtc.QPointF(self.Wheel.x, self.Road.y),
                                    coilsWidth=10, coilsLength=20, nCoils=4, pen=self.penMass, name='k2',
                                    label='Spring 2', k=10)
        self.dashpot = DashPot(qtc.QPointF(-self.spring1.stPt.x(), self.spring1.stPt.y()),
                               qtc.QPointF(-self.spring1.enPt.x(), self.spring1.enPt.y()), dpWidth=10, dpLength=30,
                               pen=self.penMass, name='c', label='Dashpot', c=10)
        # Add items to the scene
        self.scene.addItem(self.Road)
        self.Wheel.addToScene(self.scene)
        self.scene.addItem(self.CarBody)
        self.scene.addItem(self.spring1)
        self.scene.addItem(self.spring2)
        self.scene.addItem(self.dashpot)

    def setupCanvasMoveEvent(self, window):
        """
        Connect mouse move events on the canvas to the window's event handler.

        Args:
            window: Window object handling the events
        """
        self.displacement_canvas.mpl_connect("motion_notify_event", window.mouseMoveEvent_Canvas)
        self.forces_canvas.mpl_connect("motion_notify_event", window.mouseMoveEvent_Canvas)

    def setupEventFilter(self, window):
        """
        Install an event filter on the scene for mouse tracking.

        Args:
            window: Window object to handle the events
        """
        self.gv_Schematic.setMouseTracking(True)
        self.gv_Schematic.scene().installEventFilter(window)

    def getZoom(self):
        """
        Get the current zoom level of the schematic.

        Returns:
            float: Zoom level (scale factor)
        """
        return self.gv_Schematic.transform().m11()

    def setZoom(self, val):
        """
        Set the zoom level of the schematic.

        Args:
            val (float): Zoom level (scale factor)
        """
        self.gv_Schematic.resetTransform()
        self.gv_Schematic.scale(val, val)

    def updateSchematic(self):
        """Update the schematic display."""
        self.scene.update()

    def setupPensAndBrushes(self):
        """Initialize pens and brushes for schematic items."""
        self.penTire = qtg.QPen(qtg.QColor(qtc.Qt.black))
        self.penTire.setWidth(3)
        self.penMass = qtg.QPen(qtg.QColor("black"))
        self.penMass.setWidth(1)
        color = qtg.QColor(qtc.Qt.gray)
        color.setAlpha(64)
        self.brushWheel = qtg.QBrush(color)
        self.brushMass = qtg.QBrush(qtg.QColor(200, 200, 200, 64))
        self.groundPen = qtg.QPen(qtg.QColor(qtc.Qt.black))
        self.groundPen.setWidth(1)
        self.groundBrush = qtg.QBrush(qtg.QColor(qtc.Qt.black))
        self.groundBrush.setStyle(qtc.Qt.DiagCrossPattern)

    def doPlot(self, model=None):
        """
        Update both displacement and forces plots.

        Args:
            model (CarModel, optional): Model instance to plot
        """
        if model is None or model.results is None:
            logging.debug("Model or results are None in doPlot, skipping plot")
            return

        if not self.is_valid_data(model):
            logging.error("Invalid data for plotting, skipping plot")
            return

        # Plot both displacement and forces
        self.plotDisplacements(model)
        self.plotForces(model)

    def is_valid_data(self, model):
        """
        Check if the simulation data is valid for plotting.

        Args:
            model (CarModel): Model instance to check

        Returns:
            bool: True if data is valid, False otherwise
        """
        data_arrays = [
            model.timeData,
            model.wheelPosData,
            model.bodyPosData,
            model.roadPosData,
            model.springForceData,
            model.dashpotForceData
        ]
        if model.accelBodyData is not None:
            data_arrays.append(model.accelBodyData)

        for data in data_arrays:
            if not np.all(np.isfinite(data)):
                logging.error(f"Invalid data detected: {data}")
                return False
            if np.any(np.abs(data) > 1e6):
                logging.error(f"Data exceeds reasonable bounds: {data}")
                return False
        return True

    def plotDisplacements(self, model):
        """
        Plot displacement vs. time with optional acceleration.

        Args:
            model (CarModel): Model instance to plot
        """
        try:
            self.ax.clear()
            self.ax1.clear()

            # Plot displacements
            self.ax.plot(model.timeData, model.roadPosData, label='Road', color='black')
            self.ax.plot(model.timeData, model.wheelPosData, label='Wheel', color='blue')
            self.ax.plot(model.timeData, model.bodyPosData, label='Car Body', color='green')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Displacement (m)')
            self.ax.legend(loc='upper left')
            self.ax.grid(True)

            # Apply log scales if checked
            if self.chk_LogX.isChecked():
                self.ax.set_xscale('log')
                self.ax.set_xlim(left=1e-6)  # Ensure positive range for log scale
            else:
                self.ax.set_xscale('linear')
            if self.chk_LogY.isChecked():
                self.ax.set_yscale('log')
                # Adjust limits to avoid log scale issues with negative/zero values
                max_y = max(np.max(model.roadPosData), np.max(model.wheelPosData), np.max(model.bodyPosData))
                min_y = min(np.min(model.roadPosData), np.min(model.wheelPosData), np.min(model.bodyPosData))
                if min_y <= 0:
                    self.ax.set_ylim(bottom=1e-6, top=max_y + 1e-6)
            else:
                self.ax.set_yscale('linear')

            # Plot acceleration if checked
            if self.chk_ShowAccel.isChecked() and model.accelBodyData is not None:
                self.ax1.plot(model.timeData, model.accelBodyData, label='Acceleration', color='red', linestyle='--')
                self.ax1.set_ylabel('Acceleration (g)')
                self.ax1.legend(loc='upper right')
                if self.chk_LogX.isChecked():
                    self.ax1.set_xscale('log')
                if self.chk_LogAccel.isChecked():
                    self.ax1.set_yscale('log')
                    max_accel = np.max(np.abs(model.accelBodyData))
                    min_accel = np.min(np.abs(model.accelBodyData))
                    if min_accel <= 0:
                        self.ax1.set_ylim(custom=1e-6, top=max_accel + 1e-6)
                else:
                    self.ax1.set_yscale('linear')

            self.displacement_canvas.draw()
        except Exception as e:
            logging.error(f"Error in plotDisplacements: {str(e)}", exc_info=True)

    def plotForces(self, model):
        """
        Plot spring and dashpot forces vs. time.

        Args:
            model (CarModel): Model instance to plot
        """
        try:
            self.forces_ax.clear()

            # Plot forces
            self.forces_ax.plot(model.timeData, model.springForceData, label='Spring Force', color='blue')
            self.forces_ax.plot(model.timeData, model.dashpotForceData, label='Dashpot Force', color='red')
            self.forces_ax.set_xlabel('Time (s)')
            self.forces_ax.set_ylabel('Force (N)')
            self.forces_ax.legend(loc='upper left')
            self.forces_ax.grid(True)

            # Apply log scales if checked
            if self.chk_LogX.isChecked():
                self.forces_ax.set_xscale('log')
                self.forces_ax.set_xlim(left=1e-6)
            else:
                self.forces_ax.set_xscale('linear')
            if self.chk_LogY.isChecked():
                self.forces_ax.set_yscale('log')
                max_force = max(np.max(np.abs(model.springForceData)), np.max(np.abs(model.dashpotForceData)))
                min_force = min(np.min(np.abs(model.springForceData)), np.min(np.abs(model.dashpotForceData)))
                if min_force <= 0:
                    self.forces_ax.set_ylim(custom=1e-6, top=max_force + 1e-6)
            else:
                self.forces_ax.set_yscale('linear')

            self.forces_canvas.draw()
        except Exception as e:
            logging.error(f"Error in plotForces: {str(e)}", exc_info=True)

    def getPoints(self, model=None, t=0):
        """
        Get the positions and acceleration at a specific time.

        Args:
            model (CarModel, optional): Model instance
            t (float): Time point (s)

        Returns:
            tuple: (wheel position, body position, road position, acceleration)
        """
        if model is None or model.results is None:
            logging.debug("Model or results are None in getPoints, returning zeros")
            return 0, 0, 0, 0
        ywheel = np.interp(t, model.timeData, model.wheelPosData)
        ybody = np.interp(t, model.timeData, model.bodyPosData)
        yroad = np.interp(t, model.timeData, model.roadPosData)
        accel = np.interp(t, model.timeData, model.accelBodyData) if model.accelBodyData is not None else 0
        return ywheel, ybody, yroad, accel

    def animate(self, model=None, t=0):
        """
        Animate the schematic at a specific time.

        Args:
            model (CarModel, optional): Model instance
            t (float): Time point (s)
        """
        if model is None or model.results is None:
            logging.debug("Model or results are None in animate, skipping")
            return
        ywheel, ybody, yroad, accel = self.getPoints(model, t)
        try:
            # Remove previous tracers if they exist
            if self.posWheelTracer is not None:
                self.posWheelTracer.remove()
            if self.posBodyTracer is not None:
                self.posBodyTracer.remove()
            if self.posRoadTracer is not None:
                self.posRoadTracer.remove()
            if self.accelTracer is not None:
                self.accelTracer.remove()
        except Exception as e:
            logging.warning(f"Error removing tracers in animate: {str(e)}")
        # Calculate scaling factor for schematic animation
        self.scale = 200 * (1000 / 25.4) / self.Wheel.radius
        # Update positions of schematic items
        self.Road.y = self.Road.y0 - yroad * self.scale
        self.Wheel.road_y = self.Road.y
        self.Wheel.y = self.Wheel.y0 - ywheel * self.scale
        self.spring2.enPt.setY(self.Road.y)
        self.spring2.stPt.setY(self.Wheel.y)
        self.CarBody.y = self.CarBody.y0 - ybody * self.scale
        self.spring1.stPt.setY(self.Wheel.y)
        self.spring1.enPt.setY(self.CarBody.y)
        self.dashpot.stPt.setY(self.Wheel.y)
        self.dashpot.enPt.setY(self.CarBody.y)
        self.scene.update()

class CarController:
    """
    The controller component of the MVC pattern, handling simulation and optimization.

    Attributes:
        input_widgets (tuple): Input widgets (line edits, checkbox)
        display_widgets (tuple): Display widgets (graphics view, checkboxes, label, layout)
        le_m1 (QLineEdit): Car body mass input
        le_v (QLineEdit): Car speed input
        le_k1 (QLineEdit): Suspension spring constant input
        le_c1 (QLineEdit): Damping coefficient input
        le_m2 (QLineEdit): Wheel mass input
        le_k2 (QLineEdit): Tire spring constant input
        le_ang (QLineEdit): Ramp angle input
        le_tmax (QLineEdit): Maximum time input
        chk_IncludeAccel (QCheckBox): Checkbox for including acceleration in optimization
        gv_Schematic (QGraphicsView): Graphics view for the schematic
        chk_LogX (QCheckBox): Checkbox for log scale on x-axis
        chk_LogY (QCheckBox): Checkbox for log scale on y-axis (displacement)
        chk_LogAccel (QCheckBox): Checkbox for log scale on acceleration axis
        chk_ShowAccel (QCheckBox): Checkbox for showing acceleration plot
        lbl_MaxMinInfo (QLabel): Label for displaying max/min info
        layout_horizontal_main (QHBoxLayout): Main layout for the GUI
        model (CarModel): Model instance
        view (CarView): View instance
        step (int): Step counter for ODE solver
    """

    def __init__(self, args):
        """
        Initialize the controller with input and display widgets.

        Args:
            args (tuple): Tuple containing input_widgets and display_widgets
        """
        self.input_widgets, self.display_widgets = args
        # Unpack input widgets
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
        self.le_tmax, self.chk_IncludeAccel = self.input_widgets
        # Unpack display widgets
        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        self.model = CarModel()  # Create model instance
        self.view = CarView(args)  # Create view instance

    def ode_system(self, t, X):
        """
        Define the system of ODEs for the quarter car model.

        Args:
            t (float): Time (s)
            X (list): State vector [x1, x1dot, x2, x2dot]

        Returns:
            list: Derivatives [x1dot, x1ddot, x2dot, x2ddot]
        """
        # Calculate road height at time t
        if t < self.model.tramp:
            y = self.model.ymag * (t / self.model.tramp)
        else:
            y = self.model.ymag

        x1, x1dot, x2, x2dot = X  # Unpack state variables

        # Calculate accelerations
        x1ddot = (1 / self.model.m1) * (self.model.c1 * (x2dot - x1dot) + self.model.k1 * (x2 - x1))
        x2ddot = (1 / self.model.m2) * (
            -self.model.c1 * (x2dot - x1dot) - self.model.k1 * (x2 - x1) + self.model.k2 * (y - x2))
        self.step += 1

        # Check for non-finite values
        if not np.all(np.isfinite([x1ddot, x2ddot])):
            logging.error(f"ODE system returned non-finite values at t={t}: x1ddot={x1ddot}, x2ddot={x2ddot}")
            raise ValueError("ODE system produced non-finite values")
        return [x1dot, x1ddot, x2dot, x2ddot]

    def calculate(self, doCalc=True):
        """
        Perform the simulation by updating the model and view.

        Args:
            doCalc (bool): Whether to perform the simulation calculation
        """
        try:
            # Update model parameters from GUI inputs
            self.model.m1 = float(self.le_m1.text())
            self.model.m2 = float(self.le_m2.text())
            self.model.c1 = float(self.le_c1.text())
            self.model.k1 = float(self.le_k1.text())
            self.model.k2 = float(self.le_k2.text())
            self.model.v = float(self.le_v.text())
        except ValueError as e:
            logging.error(f"Error parsing input values: {str(e)}")
            raise

        logging.debug(f"Input parameters: m1={self.model.m1}, m2={self.model.m2}, c1={self.model.c1}, k1={self.model.k1}, k2={self.model.k2}, v={self.model.v}")

        # Recalculate bounds for k1 and k2
        self.model.mink1 = (self.model.m1 * 9.81) / (6.0 * 25.4 / 1000.0)
        self.model.maxk1 = (self.model.m1 * 9.81) / (3.0 * 25.4 / 1000.0)
        self.model.mink2 = ((self.model.m1 + self.model.m2) * 9.81) / (1.5 * 25.4 / 1000.0)
        self.model.maxk2 = ((self.model.m1 + self.model.m2) * 9.81) / (0.75 * 25.4 / 1000.0)

        ymag = 6.0 / (12.0 * 3.3)  # Ramp height
        if ymag is not None:
            self.model.ymag = ymag
        self.model.yangdeg = float(self.le_ang.text())
        self.model.tmax = float(self.le_tmax.text())
        if doCalc:
            self.doCalc()
        self.SSE((self.model.k1, self.model.c1, self.model.k2), optimizing=False)
        self.view.updateView(self.model)
        # Print calculation results
        print("\n=== Calculation Results ===")
        print(f"m1: {self.model.m1:.2f} kg")
        print(f"m2: {self.model.m2:.2f} kg")
        print(f"k1: {self.model.k1:.2f} N/m")
        print(f"c1: {self.model.c1:.2f} N·s/m")
        print(f"k2: {self.model.k2:.2f} N/m")
        print(f"v: {self.model.v:.2f} km/h")
        print(f"yangdeg: {self.model.yangdeg:.2f} degrees")
        print(f"tmax: {self.model.tmax:.2f} s")
        print(f"SSE: {self.model.SSE:.2f}")
        if self.model.results is not None:
            print(f"Max Body Position: {self.model.bodyPosData.max():.4f} m")
            print(f"Max Wheel Position: {self.model.wheelPosData.max():.4f} m")
            print(f"Max Spring Force: {self.model.springForceData.max():.2f} N")
            print(f"Max Dashpot Force: {self.model.dashpotForceData.max():.2f} N")
            if self.model.accelBodyData is not None:
                print(f"Max Body Acceleration: {self.model.accelBodyData.max():.4f} g")

    def setupCanvasMoveEvent(self, window):
        """
        Set up the canvas move event for the view.

        Args:
            window: Window object handling the events
        """
        self.view.setupCanvasMoveEvent(window)

    def setupEventFilter(self, window):
        """
        Set up the event filter for the view.

        Args:
            window: Window object handling the events
        """
        self.view.setupEventFilter(window=window)

    def getZoom(self):
        """
        Get the current zoom level of the schematic.

        Returns:
            float: Zoom level (scale factor)
        """
        return self.view.getZoom()

    def setZoom(self, val):
        """
        Set the zoom level of the schematic.

        Args:
            val (float): Zoom level (scale factor)
        """
        self.view.setZoom(val=val)

    def updateSchematic(self):
        """Update the schematic display."""
        self.view.updateSchematic()

    def doCalc(self, doPlot=True, doAccel=True):
        """
        Perform the core simulation steps: solve ODEs, compute forces, and optionally plot.

        Args:
            doPlot (bool): Whether to update the plots
            doAccel (bool): Whether to calculate acceleration
        """
        try:
            v = 1000 * self.model.v / 3600  # Convert speed to m/s
            self.model.angrad = self.model.yangdeg * math.pi / 180.0  # Convert angle to radians
            self.model.tramp = self.model.ymag / (math.sin(self.model.angrad) * v)  # Time to reach ramp height

            # Set up time points for simulation (logarithmic scale)
            t_eval = np.logspace(np.log10(1e-6), np.log10(self.model.tmax), 200)
            ic = [0, 0, 0, 0]  # Initial conditions: [x1, x1dot, x2, x2dot]
            self.step = 0  # Reset step counter
            logging.debug("Starting solve_ivp")
            self.model.results = solve_ivp(self.ode_system, t_span=[0, self.model.tmax], y0=ic, t_eval=t_eval, method='LSODA')
            logging.debug("Finished solve_ivp")

            if not self.model.results.success:
                logging.error(f"solve_ivp failed: {self.model.results.message}")
                raise RuntimeError(f"solve_ivp failed: {self.model.results.message}")

            # Store simulation results
            self.model.timeData = self.model.results.t
            # Calculate road position at each time point
            self.model.roadPosData = np.array([self.model.ymag if t > self.model.tramp else t * self.model.ymag / self.model.tramp for t in self.model.timeData])
            self.model.bodyPosData = self.model.results.y[0]  # Car body position
            self.model.wheelPosData = self.model.results.y[2]  # Wheel position

            # Compute spring and dashpot forces
            x1 = self.model.results.y[0]
            x2 = self.model.results.y[2]
            v1 = self.model.results.y[1]
            v2 = self.model.results.y[3]
            self.model.springForceData = self.model.k1 * (x2 - x1)
            self.model.dashpotForceData = self.model.c1 * (v2 - v1)

            # Log min/max positions for debugging
            logging.debug(f"bodyPosData: min={self.model.bodyPosData.min()}, max={self.model.bodyPosData.max()}")
            logging.debug(f"wheelPosData: min={self.model.wheelPosData.min()}, max={self.model.wheelPosData.max()}")

            # Check for non-finite values
            if not np.all(np.isfinite(self.model.bodyPosData)) or not np.all(np.isfinite(self.model.wheelPosData)):
                logging.error("Non-finite values in bodyPosData or wheelPosData")
                raise ValueError("Non-finite values in simulation results")

            if doAccel:
                self.calcAccel()
            if doPlot:
                self.doPlot()
        except Exception as e:
            logging.error(f"Error in doCalc: {str(e)}", exc_info=True)
            raise

    def calcAccel(self):
        """
        Calculate the car body acceleration from velocity data.

        Returns:
            bool: True if successful
        """
        try:
            N = len(self.model.timeData)
            self.model.accelBodyData = np.zeros(shape=N)
            vel = self.model.results.y[1]  # Car body velocity
            # Compute acceleration using finite differences
            for i in range(N):
                if i == N - 1:
                    h = self.model.timeData[i] - self.model.timeData[i - 1]
                    self.model.accelBodyData[i] = (vel[i] - vel[i - 1]) / (9.81 * h)
                else:
                    h = self.model.timeData[i + 1] - self.model.timeData[i]
                    self.model.accelBodyData[i] = (vel[i + 1] - vel[i]) / (9.81 * h)
            self.model.accelMax = self.model.accelBodyData.max()

            # Check for non-finite values
            if not np.all(np.isfinite(self.model.accelBodyData)):
                logging.error("Non-finite values in accelBodyData")
                raise ValueError("Non-finite values in acceleration data")
            return True
        except Exception as e:
            logging.error(f"Error in calcAccel: {str(e)}", exc_info=True)
            raise

    def OptimizeSuspension(self):
        """Optimize the suspension parameters (k1, c1, k2) to minimize SSE."""
        try:
            self.calculate(doCalc=False)  # Update model parameters without recalculating
            x0 = np.array([self.model.k1, self.model.c1, self.model.k2])  # Initial guess
            # Define bounds for optimization
            bounds = [
                (self.model.mink1 * 0.9, self.model.maxk1 * 1.1),
                (100, 5000),
                (self.model.mink2 * 0.9, self.model.maxk2 * 1.1)
            ]
            logging.debug(f"Optimization starting with x0={x0}, bounds={bounds}")
            # Perform optimization using scipy's minimize
            answer = minimize(
                self.SSE,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'disp': True, 'maxiter': 50}
            )
            logging.debug(f"Optimization result: {answer}")
            self.model.k1, self.model.c1, self.model.k2 = answer.x  # Update model with optimized parameters
            self.view.updateView(self.model, doPlot=True)  # Update view
            # Print optimization results
            print("\n=== Optimization Results ===")
            print(f"Optimized k1: {self.model.k1:.2f} N/m")
            print(f"Optimized c1: {self.model.c1:.2f} N·s/m")
            print(f"Optimized k2: {self.model.k2:.2f} N/m")
            print(f"SSE: {self.model.SSE:.2f}")
            print(f"k1 bounds: [{self.model.mink1:.2f}, {self.model.maxk1:.2f}]")
            print(f"c1 bounds: [100, 5000]")
            print(f"k2 bounds: [{self.model.mink2:.2f}, {self.model.maxk2:.2f}]")
        except Exception as e:
            logging.error(f"Error in OptimizeSuspension: {str(e)}", exc_info=True)
            raise

    def SSE(self, vals, optimizing=True):
        """
        Compute the sum of squared errors (SSE) for optimization.

        Args:
            vals (tuple): Parameters to evaluate (k1, c1, k2)
            optimizing (bool): Whether this is part of an optimization run

        Returns:
            float: SSE value
        """
        try:
            k1, c1, k2 = vals
            logging.debug(f"SSE iteration with k1={k1}, c1={c1}, k2={k2}")
            self.model.k1 = k1
            self.model.c1 = c1
            self.model.k2 = k2

            # Reset model data
            self.model.results = None
            self.model.timeData = None
            self.model.roadPosData = None
            self.model.bodyPosData = None
            self.model.wheelPosData = None
            self.model.springForceData = None
            self.model.dashpotForceData = None
            self.model.accelBodyData = None

            self.doCalc(doPlot=False)  # Recalculate with new parameters

            # Check for excessive displacement
            max_displacement = max(np.max(np.abs(self.model.bodyPosData)), np.max(np.abs(self.model.wheelPosData)))
            if max_displacement > 1.0:
                logging.warning(f"Excessive displacement detected: {max_displacement}")
                return 1e6

            # Calculate SSE based on deviation from target position
            SSE = 0
            for i in range(len(self.model.results.y[0])):
                t = self.model.timeData[i]
                y = self.model.results.y[0][i]
                if t < self.model.tramp:
                    ytarget = self.model.ymag * (t / self.model.tramp)
                else:
                    ytarget = self.model.ymag
                SSE += (y - ytarget) ** 2

            # Add penalties during optimization
            if optimizing:
                if k1 < self.model.mink1 or k1 > self.model.maxk1:
                    SSE += 100
                if c1 < 10:
                    SSE += 100
                if k2 < self.model.mink2 or k2 > self.model.maxk2:
                    SSE += 100
                o_IncludeAccel = self.chk_IncludeAccel.isChecked()
                if self.model.accelMax > self.model.accelLim and o_IncludeAccel:
                    SSE += 10 + 10 * (self.model.accelMax - self.model.accelLim) ** 2
            self.model.SSE = SSE
            return SSE
        except Exception as e:
            logging.error(f"Error in SSE: {str(e)}", exc_info=True)
            return 1e6

    def doPlot(self):
        """Update the plots in the view."""
        try:
            self.view.doPlot(self.model)
        except Exception as e:
            logging.error(f"Error in doPlot: {str(e)}", exc_info=True)

    def animate(self, t):
        """
        Animate the schematic at a specific time.

        Args:
            t (float): Time point (s)
        """
        try:
            self.view.animate(self.model, t)
        except Exception as e:
            logging.error(f"Error in animate: {str(e)}", exc_info=True)

    def getPoints(self, t):
        """
        Get the positions and acceleration at a specific time.

        Args:
            t (float): Time point (s)

        Returns:
            tuple: (wheel position, body position, road position, acceleration)
        """
        try:
            return self.view.getPoints(self.model, t)
        except Exception as e:
            logging.error(f"Error in getPoints: {str(e)}", exc_info=True)
#endregion
#endregion