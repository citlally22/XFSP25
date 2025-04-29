# region imports
import math
import random as rnd
from datetime import datetime
from copy import deepcopy as dc
import numpy as np  # For normal distribution sampling
# endregion

# region class definitions
class Position:
    """
    Class for representing a point in 3D space with vector arithmetic capabilities.

    This class, originally by Jim Smay (last edit: 04/27/2022), models a position in 3D space.
    It supports vector operations like addition, subtraction, and dot products through operator overloading.
    """
    def __init__(self, pos=None, x=None, y=None, z=None):
        """
        Initialize a Position object with x, y, z coordinates.

        Args:
            pos (tuple, optional): A tuple of (x, y, z) coordinates.
            x (float, optional): X-coordinate.
            y (float, optional): Y-coordinate.
            z (float, optional): Z-coordinate.
        """
        # Set default coordinates to 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        # Unpack coordinates from a tuple if provided
        if pos is not None:
            self.x, self.y, self.z = pos
        # Override defaults with provided x, y, z values
        self.x = x if x is not None else self.x
        self.y = y if y is not None else self.y
        self.z = z if z is not None else self.z

    # region operator overloads
    def __add__(self, other):
        """
        Overload the addition operator to add two Position objects.

        Args:
            other (Position): Another Position object to add.

        Returns:
            Position: A new Position object with coordinates (self.x + other.x, self.y + other.y, self.z + other.z).
        """
        return Position((self.x + other.x, self.y + other.y, self.z + other.z))

    def __iadd__(self, other):
        """
        Overload the iterative addition operator (+=).

        Args:
            other (Position or float/int): Position object or scalar to add to self.

        Returns:
            Position: Updated self with new coordinates.
        """
        if type(other) in (float, int):
            self.x += other
            self.y += other
            self.z += other
            return self
        if type(other) == Position:
            self.x += other.x
            self.y += other.y
            self.z += other.z
            return self

    def __sub__(self, other):
        """
        Overload the subtraction operator to subtract two Position objects.

        Args:
            other (Position): Another Position object to subtract.

        Returns:
            Position: A new Position object with coordinates (self.x - other.x, self.y - other.y, self.z - other.z).
        """
        return Position((self.x - other.x, self.y - other.y, self.z - other.z))

    def __isub__(self, other):
        """
        Overload the iterative subtraction operator (-=).

        Args:
            other (Position or float/int): Position object or scalar to subtract from self.

        Returns:
            Position: Updated self with new coordinates.
        """
        if type(other) in (float, int):
            self.x -= other
            self.y -= other
            self.z -= other
            return self
        if type(other) == Position:
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
            return self

    def __mul__(self, other):
        """
        Overload the multiplication operator for scalar multiplication or dot product.

        Args:
            other (float/int or Position): Scalar for scaling or Position for dot product.

        Returns:
            Position: A new Position object with scaled or dot product coordinates.
        """
        if type(other) in (float, int):
            return Position((self.x * other, self.y * other, self.z * other))
        if type(other) is Position:
            return Position((self.x * other.x, self.y * other.y, self.z * other.z))

    def __rmul__(self, other):
        """
        Overload the right multiplication operator for scalar * Position.

        Args:
            other (float/int): Scalar to multiply with self.

        Returns:
            Position: A new Position object with scaled coordinates.
        """
        return self * other

    def __imul__(self, other):
        """
        Overload the iterative multiplication operator (*=).

        Args:
            other (float/int): Scalar to scale self.

        Returns:
            Position: Updated self with scaled coordinates.
        """
        if type(other) in (float, int):
            self.x *= other
            self.y *= other
            self.z *= other
            return self

    def __truediv__(self, other):
        """
        Overload the division operator for scalar division.

        Args:
            other (float/int): Scalar to divide self by.

        Returns:
            Position: A new Position object with divided coordinates.
        """
        if type(other) in (float, int):
            return Position((self.x / other, self.y / other, self.z / other))

    def __idiv__(self, other):
        """
        Overload the iterative division operator (/=).

        Args:
            other (float/int): Scalar to divide self by.

        Returns:
            Position: Updated self with divided coordinates.
        """
        if type(other) in (float, int):
            self.x /= other
            self.y /= other
            self.z /= other
            return self

    def __round__(self, n=None):
        """
        Round the coordinates to a specified number of decimal places.

        Args:
            n (int, optional): Number of decimal places to round to.

        Returns:
            Position: A new Position object with rounded coordinates.
        """
        if n is not None:
            return Position(x=round(self.x, n), y=round(self.y, n), z=round(self.z, n))
        return self
    # endregion

    def set(self, strXYZ=None, tupXYZ=None, SI=True):
        """
        Set the position using a string or tuple of coordinates.

        Args:
            strXYZ (str, optional): String of coordinates in format "x,y,z".
            tupXYZ (tuple, optional): Tuple of coordinates (x, y, z).
            SI (bool): If True, use SI units; if False, apply a conversion factor (3.3).
        """
        lenCF = 1 if SI else 3.3
        if strXYZ is not None:
            cells = strXYZ.replace('(', '').replace(')', '').strip().split(',')
            x, y, z = float(cells[0]), float(cells[1]), float(cells[2])
            self.x = lenCF * float(x)
            self.y = lenCF * float(y)
            self.z = lenCF * float(z)
        elif tupXYZ is not None:
            x, y, z = tupXYZ
            self.x = lenCF * float(x)
            self.y = lenCF * float(y)
            self.z = lenCF * float(z)

    def getTup(self):
        """
        Get the coordinates as a tuple.

        Returns:
            tuple: (x, y, z) coordinates.
        """
        return (self.x, self.y, self.z)

    def getStr(self, nPlaces=3, SI=True, scientific=False):
        """
        Get a string representation of the position.

        Args:
            nPlaces (int): Number of decimal places in formatted string.
            SI (bool): If True, use SI units; if False, apply conversion factor (3.3).
            scientific (bool): If True, use scientific notation.

        Returns:
            str: Formatted string of coordinates "x, y, z".
        """
        lenCF = 1 if SI else 3.3
        fmtStr = '{:.' + str(nPlaces) + ('e}' if scientific else 'f}')
        return '' + fmtStr.format(self.x * lenCF) + ', ' + fmtStr.format(self.y * lenCF) + ', ' + fmtStr.format(self.z * lenCF) + ''

    def mag(self):
        """
        Calculate the magnitude of the position vector.

        Returns:
            float: Magnitude sqrt(x^2 + y^2 + z^2).
        """
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def normalize(self):
        """
        Normalize the position vector to a unit vector.

        Divides each coordinate by the magnitude to create a unit vector.
        """
        l = self.mag()
        if l <= 0.0:
            return
        self.__idiv__(l)

    def normalize2D(self):
        """
        Normalize the position vector in the XY plane (sets z to 0).
        """
        self.z = 0.0
        self.normalize()

    def getAngleRad_XYPlane(self):
        """
        Calculate the angle of the position in the XY plane relative to the origin.

        Returns:
            float: Angle in radians.
        """
        l = self.mag()
        if l <= 0.0:
            return 0
        if self.y >= 0.0:
            return math.acos(self.x / l)
        return 2.0 * math.pi - math.acos(self.x / l)

    def getAngleDeg_XYPlane(self):
        """
        Calculate the angle of the position in the XY plane in degrees.

        Returns:
            float: Angle in degrees.
        """
        return 180.0 / math.pi * self.getAngleRad_XYPlane()

    def midPt(self, p2=None):
        """
        Find the midpoint between self and another position.

        Args:
            p2 (Position, optional): Another position to find the midpoint with.

        Returns:
            Position: Midpoint position.
        """
        return Position(x=self.x + 0.5 * (p2.x - self.x), y=self.y + 0.5 * (p2.y - self.y), z=self.z + 0.5 * (p2.z - self.z))

    def distTo(self, p2=None):
        """
        Calculate the distance to another position or the origin.

        Args:
            p2 (Position, optional): Another position to calculate distance to.

        Returns:
            float: Distance between positions.
        """
        if p2 is None:
            return self.mag()
        return (self - p2).mag()

    def getRndDir(self):
        """
        Generate a random unit vector direction.

        Returns:
            Position: A unit vector in a random direction.
        """
        d = Position(x=rnd.random(), y=rnd.random(), z=rnd.random())
        d -= 0.5  # Center the random values around 0
        d.normalize()  # Normalize to unit vector
        return d

    def getRndPosOnSphere(self, radius=1.0):
        """
        Calculate a random position on a sphere centered at self.

        Args:
            radius (float): Radius of the sphere.

        Returns:
            Position: A random position radius away from self.
        """
        rndVec = radius * self.getRndDir()
        return self + rndVec

class molecule:
    """
    Class for modeling a single particle (mer) in a polymer chain.

    Represents a single mer with a molecular weight and position in 3D space.
    """
    def __init__(self, molecularWeight=12, position=Position()):
        """
        Initialize a molecule object.

        Args:
            molecularWeight (float): Molecular weight in Daltons (g/mol).
            position (Position): Location of the particle in 3D space.
        """
        self.MW = molecularWeight
        self.position = position

class macroMolecule:
    """
    Class for modeling a polymer molecule using the freely jointed chain model.

    Represents a poly(ethylene) molecule as a chain of mers, with methods to simulate
    its structure and calculate properties like center of mass, end-to-end distance,
    and radius of gyration.
    """
    def __init__(self, targetN=1000, segmentLength=4.84e-9, merWt=14):
        """
        Initialize a macroMolecule object.

        Args:
            targetN (int): Target degree of polymerization (mean of the normal distribution).
            segmentLength (float): Effective segment length (set to 4.84 nm to match example output).
            merWt (float): Molecular weight of a single mer (default 14 for CH2).
        """
        self.merWt = merWt
        # Sample degree of polymerization from a normal distribution (mean = targetN, std = 0.1 * targetN)
        std = 0.1 * targetN
        self.N = max(1, round(np.random.normal(targetN, std)))  # Ensure N is at least 1
        self.MW = self.N * merWt + 2  # Total molecular weight, +2 for extra hydrogens at ends
        self.segmentLength = segmentLength  # Segment length in meters (4.84 nm = 4.84e-9 m)
        self.centerOfMass = Position()  # Initialize center of mass
        self.radiusOfGyration = 0  # Initialize radius of gyration
        self.radiusOfInfluence = 0  # Not used in this problem
        self.endToEndDistance = 0  # Initialize end-to-end distance
        self.mers = []  # List to store mers in the polymer chain

    def freelyJointedChainModel(self):
        """
        Simulate the polymer structure using the freely jointed chain model.

        Steps:
        1. Pin the initial mer to location (0,0,0).
        2. Use Position.getRndDir() to get a random direction.
        3. Place the next mer at a distance of segmentLength in that direction.
        4. Repeat steps 2 & 3 for remaining links.
        5. Calculate center of mass, radius of gyration, and end-to-end distance.
        """
        # Step 1: Pin the initial mer to (0,0,0)
        lastPosition = Position(x=0, y=0, z=0)
        self.mers = []
        # Steps 2, 3, 4: Build the chain by placing each mer
        M = int(self.N)
        for n in range(M):
            m = molecule(molecularWeight=self.merWt)
            # Add extra hydrogen weight (1 Da) to first and last mers
            m.MW += 1 if (n == 0 or n == (self.N - 1)) else 0
            # Place the mer at a distance of segmentLength in a random direction
            m.position = lastPosition.getRndPosOnSphere(self.segmentLength)
            self.mers.append(m)
            lastPosition = m.position

        # Step 5: Calculate metrics
        # Center of mass: sum(m_i * r_i) / sum(m_i)
        for m in self.mers:
            self.centerOfMass += m.MW * m.position
        self.centerOfMass /= self.MW
        # End-to-end distance: distance between first and last mers
        self.endToEndDistance = (self.mers[0].position - self.mers[-1].position).mag()
        # Radius of gyration: sqrt(sum(m_i * (r_i - r_com)^2) / sum(m_i))
        self.radiusOfGyration = (sum([mer.MW * (mer.position.distTo(self.centerOfMass))**2 for mer in self.mers]) / self.MW)**0.5

    def get_molecular_weight(self):
        """
        Get the molecular weight of the macromolecule.

        Returns:
            float: Molecular weight in Daltons (g/mol).
        """
        return self.MW
# endregion