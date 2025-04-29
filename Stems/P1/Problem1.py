import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel


# Model
class STOModel:
    """
    Model class to calculate Short Take-Off (STO) distance based on thrust and weight using given equations.

    This class implements the STO calculation using the provided formulas:
    - V_stall = sqrt(Weight / (0.5 * rho * S * CL_max))
    - V_TO = 1.2 * V_stall
    - A = gc * (Thrust / Weight)
    - B = (gc / Weight) * (0.5 * rho * S * CD)
    - STO = integral from 0 to V_TO of V / (A - B * V^2) dV
    """

    def __init__(self):
        # Constants in English units
        self.gc = 32.174  # lbm·ft/lbf·s², gravitational constant
        self.g = 32.2  # ft/s², gravitational acceleration
        self.rho = 0.002377  # slug/ft³, air density
        self.S = 100  # ft², wing area
        self.CL_max = 2.4  # Maximum lift coefficient
        self.CD = 0.35  # Drag coefficient, adjusted to match provided graph

    def calculate_v_stall(self, weight):
        """
        Calculate stall speed V_stall.

        Args:
            weight (float): Aircraft weight in pounds (lb).

        Returns:
            float: Stall speed in ft/s.
        """
        # V_stall = sqrt(Weight / (0.5 * rho * S * CL_max))
        return np.sqrt(weight / (0.5 * self.rho * self.S * self.CL_max))

    def calculate_v_to(self, v_stall):
        """
        Calculate take-off speed V_TO.

        Args:
            v_stall (float): Stall speed in ft/s.

        Returns:
            float: Take-off speed in ft/s.
        """
        # V_TO = 1.2 * V_stall
        return 1.2 * v_stall

    def calculate_A(self, thrust, weight):
        """
        Calculate A parameter.

        Args:
            thrust (float): Engine thrust in lbf.
            weight (float): Aircraft weight in lb.

        Returns:
            float: A parameter (dimensionless in this context).
        """
        # A = gc * (Thrust / Weight)
        return self.gc * (thrust / weight)

    def calculate_B(self, weight):
        """
        Calculate B parameter.

        Args:
            weight (float): Aircraft weight in lb.

        Returns:
            float: B parameter in 1/(ft²).
        """
        # B = (gc / Weight) * (0.5 * rho * S * CD)
        return (self.gc / weight) * (0.5 * self.rho * self.S * self.CD)

    def calculate_sto(self, thrust, weight):
        """
        Calculate STO using the analytical solution of the integral.

        The integral STO = ∫(0 to V_TO) V / (A - B * V^2) dV is solved as:
        STO = (1 / (2 * B)) * ln(A / (A - B * V_TO^2)).

        Args:
            thrust (float): Engine thrust in lbf.
            weight (float): Aircraft weight in lb.

        Returns:
            float: Short Take-Off distance in ft.
        """
        v_stall = self.calculate_v_stall(weight)
        v_to = self.calculate_v_to(v_stall)
        A = self.calculate_A(thrust, weight)
        B = self.calculate_B(weight)
        # Check for valid logarithm
        term = A - B * v_to ** 2
        if term <= 0:  # Avoid log of negative or zero
            return float('inf')
        # STO = (1 / (2 * B)) * ln(A / (A - B * V_TO^2))
        sto = (1 / (2 * B)) * np.log(A / term)
        return sto

    def get_sto_data(self, thrust_range, weight, thrust_input):
        """
        Compute STO for three weights and the specified point.

        Args:
            thrust_range (numpy.ndarray): Array of thrust values to compute STO for.
            weight (float): Specified aircraft weight in lb.
            thrust_input (float): Specified thrust input in lbf.

        Returns:
            tuple: (list of STO data for three weights, STO at specified point, list of weights used).
        """
        min_weight = 1000  # Minimum allowable weight to avoid negative values
        weights = [max(min_weight, weight - 10000), weight, weight + 10000]
        sto_data = []
        for w in weights:
            # Compute STO for each thrust value in the range
            sto = [self.calculate_sto(t, w) if t > 0 else float('inf') for t in thrust_range]
            sto_data.append(sto)
        # Calculate STO at the specified thrust and weight
        sto_point = self.calculate_sto(thrust_input, weight)
        return sto_data, sto_point, weights


# View
class STOView(QMainWindow):
    """
    GUI View class for STO calculation.

    This class sets up the graphical interface with input fields for Weight and Thrust,
    a Calculate button, and a matplotlib canvas to display the STO vs. Thrust plot.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("STO Calculator")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Input fields for Weight and Thrust
        input_layout = QHBoxLayout()
        self.weight_input = QLineEdit()
        self.thrust_input = QLineEdit()
        input_layout.addWidget(QLabel("Weight (lb):"))
        input_layout.addWidget(self.weight_input)
        input_layout.addWidget(QLabel("Thrust (lbf):"))
        input_layout.addWidget(self.thrust_input)
        layout.addLayout(input_layout)

        # Calculate button
        self.calc_button = QPushButton("Calculate")
        layout.addWidget(self.calc_button)

        # Matplotlib canvas for plotting
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def update_plot(self, thrust_range, sto_data, weights, thrust_input, sto_point):
        """
        Update the matplotlib plot with three STO lines and a marked point.

        Args:
            thrust_range (numpy.ndarray): Array of thrust values for the x-axis.
            sto_data (list): List of STO values for three weights.
            weights (list): List of the three weights used.
            thrust_input (float): Specified thrust input in lbf.
            sto_point (float): STO value at the specified thrust and weight.
        """
        self.ax.clear()
        # Plot three lines for different weights
        labels = [f"Weight = {weights[0]} lb", f"Weight = {weights[1]} lb", f"Weight = {weights[2]} lb"]
        for sto, label in zip(sto_data, labels):
            self.ax.plot(thrust_range, sto, label=label)
        # Mark the specified point with a red circle
        self.ax.plot(thrust_input, sto_point, 'ro', label="Specified Point")
        self.ax.set_xlabel("Thrust (lbf)")
        self.ax.set_ylabel("STO (ft)")
        self.ax.set_ylim(0, 6000)  # Match the scale of the provided graph
        self.ax.set_xlim(0, 3000)  # Match the thrust range of the provided graph
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()


# Controller
class STOController:
    """
    Controller class to connect the model and view.

    This class handles user interactions, retrieves inputs from the view,
    triggers calculations in the model, and updates the plot in the view.
    """

    def __init__(self, model, view):
        self.model = model
        self.view = view
        # Connect the calculate button to the calculate method
        self.view.calc_button.clicked.connect(self.calculate)

    def calculate(self):
        """
        Handle the calculate button click event.

        Retrieves Weight and Thrust inputs, computes STO data, and updates the plot.
        """
        try:
            weight = float(self.view.weight_input.text())
            thrust_input = float(self.view.thrust_input.text())
            # Define thrust range to match the provided graph
            thrust_range = np.linspace(0, 3000, 100)
            # Get STO data from the model
            sto_data, sto_point, weights = self.model.get_sto_data(thrust_range, weight, thrust_input)
            # Update the plot with the computed data
            self.view.update_plot(thrust_range, sto_data, weights, thrust_input, sto_point)
        except ValueError:
            print("Invalid input. Please enter numeric values.")


# Main
if __name__ == "__main__":
    """
    Main entry point for the STO Calculator application.

    Initializes the QApplication, creates the model, view, and controller,
    and starts the GUI event loop.
    """
    app = QApplication(sys.argv)
    model = STOModel()
    view = STOView()
    controller = STOController(model, view)
    view.show()
    sys.exit(app.exec_())