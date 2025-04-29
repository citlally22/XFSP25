import numpy as np
# This import is from the provided polymerClases.py file
from polymerClases import macroMolecule

class PolymerSimulation:
    """
    Class to simulate a set of polymer molecules and compute statistical metrics.

    Simulates poly(ethylene) molecules using the freely jointed chain model and calculates
    metrics like center of mass, end-to-end distance, radius of gyration, and PDI.
    """
    def __init__(self, targetN, num_molecules):
        """
        Initialize the simulation with a target degree of polymerization and number of molecules.

        Args:
            targetN (int): Target degree of polymerization (mean of the normal distribution).
            num_molecules (int): Number of molecules to simulate.
        """
        self.targetN = targetN
        self.num_molecules = num_molecules
        self.molecules = []  # List to store macroMolecule objects
        self.create_molecules()  # Generate the molecules

    def create_molecules(self):
        """
        Create a set of macroMolecule objects and run the freely jointed chain model for each.

        Each molecule is simulated with the specified target degree of polymerization.
        """
        for _ in range(self.num_molecules):
            # Create a macromolecule with the target degree of polymerization
            # Segment length is set to 4.84 nm to match the example end-to-end distance of 0.153 μm
            mol = macroMolecule(targetN=self.targetN, segmentLength=4.84e-9, merWt=14)
            # Run the freely jointed chain simulation to build the molecule
            mol.freelyJointedChainModel()
            self.molecules.append(mol)

    def compute_metrics(self):
        """
        Compute the required metrics for the set of molecules.

        Metrics calculated:
        - Average center of mass (nm, x, y, z coordinates).
        - Average and standard deviation of end-to-end distance (μm).
        - Average and standard deviation of radius of gyration (μm).
        - Polydispersity index (PDI = Mw / Mn).

        Returns:
            dict: Dictionary containing all computed metrics.
        """
        # Center of mass: Convert from meters to nm (1 m = 1e9 nm)
        coms = np.array([mol.centerOfMass.getTup() for mol in self.molecules]) * 1e9
        avg_com = np.mean(coms, axis=0)

        # End-to-end distance: Convert from meters to μm (1 m = 1e6 μm)
        end_to_end = np.array([mol.endToEndDistance for mol in self.molecules]) * 1e6
        avg_end_to_end = np.mean(end_to_end)
        std_end_to_end = np.std(end_to_end)

        # Radius of gyration: Convert from meters to μm (1 m = 1e6 μm)
        rog = np.array([mol.radiusOfGyration for mol in self.molecules]) * 1e6
        avg_rog = np.mean(rog)
        std_rog = np.std(rog)

        # PDI = Mw / Mn
        weights = np.array([mol.get_molecular_weight() for mol in self.molecules])
        Mn = np.mean(weights)  # Number-average molecular weight
        Mw = np.mean(weights**2) / Mn  # Weight-average molecular weight
        PDI = Mw / Mn

        return {
            "avg_com": avg_com,
            "avg_end_to_end": avg_end_to_end,
            "std_end_to_end": std_end_to_end,
            "avg_rog": avg_rog,
            "std_rog": std_rog,
            "PDI": PDI
        }

def main():
    """
    Main function to run the CLI program for polymer simulation.

    Prompts the user for inputs, runs the simulation, and displays the results in the specified format.
    """
    # Prompt user for inputs with defaults
    targetN_input = input("degree of polymerization (1000)?: ")
    targetN = int(targetN_input) if targetN_input.strip() else 1000

    num_molecules_input = input("How many molecules (50)?: ")
    num_molecules = int(num_molecules_input) if num_molecules_input.strip() else 50

    # Run the simulation with the provided inputs
    sim = PolymerSimulation(targetN, num_molecules)
    metrics = sim.compute_metrics()

    # Display results in the specified format
    print(f"\nMetrics for {num_molecules} molecules of degree of polymerization = {targetN}")
    print(f"Avg. Center of Mass (nm) = {metrics['avg_com'][0]:.3f}, {metrics['avg_com'][1]:.3f}, {metrics['avg_com'][2]:.3f}")
    print("End-to-end distance (μm):")
    print(f"Average = {metrics['avg_end_to_end']:.3f}")
    print(f"Std. Dev. = {metrics['std_end_to_end']:.3f}")
    print("Radius of gyration (μm):")
    print(f"Average = {metrics['avg_rog']:.3f}")
    print(f"Std. Dev. = {metrics['std_rog']:.3f}")
    print(f"PDI = {metrics['PDI']:.2f}")

if __name__ == "__main__":
    main()