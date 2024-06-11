import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from enum import Enum
import itertools
import pickle
import pathlib
import time
import scipy
import scipy.sparse
from numba import jit


class BoundaryType(Enum):
    """Indicate the type of boundary
    """
    VELOCITY_SPECIFIED = 0
    PRESSURE_SPECIFIED = 1
    CONNECTED = 2


class Direction(Enum):
    """Indicate the four directions
    """
    LEFT = 0
    RIGHT = 1
    BOTTOM = 2
    TOP = 3


class Boundary:
    """
    A class to represent boundary conditions.

    Attributes
    ----------
    type : int
        The type of boundary condition. It can be one of the following:
        - BoundaryType.VELOCITY_SPECIFIED
        - BoundaryType.PRESSURE_SPECIFIED
        - BoundaryType.CONNECTED
    velocity : numpy.ndarray
        The velocity value for velocity-specified boundary conditions.
    pressure : float
        The pressure value for pressure-specified boundary conditions.
    block_id : int
        The block ID for connected boundary conditions.

    Methods
    -------
    None
    """

    def __init__(self, type, value=None):
        """
        Constructs all the necessary attributes for the BoundaryCondition object.

        Parameters
        ----------
        type : int
            The type of boundary condition.
        value : any, optional
            The value for velocity-specified or pressure-specified boundary conditions.
            For connected boundary conditions, it should be the block ID.
            The default value is None.

        Raises
        ------
        ValueError
            If the provided type is not one of the defined boundary types.
        """
        self.type = type
        self.velocity = None
        self.pressure = None
        self.block_id = None
        if self.type == BoundaryType.VELOCITY_SPECIFIED:
            if value is None:
                self.velocity = np.zeros(2)
            else:
                self.velocity = np.array(value)
        elif self.type == BoundaryType.PRESSURE_SPECIFIED:
            if value is None:
                self.pressure = 0.0
            else:
                self.pressure = value
        elif self.type == BoundaryType.CONNECTED:
            if value is None:
                raise ValueError("block_id must be specified for connected boundary")
            else:
                self.block_id = value
        else:
            raise ValueError("Invalid boundary type")


class Block:
    """
    Represents a block in a fluid simulation.
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.bc_left = Boundary(BoundaryType.VELOCITY_SPECIFIED)
        self.bc_right = Boundary(BoundaryType.VELOCITY_SPECIFIED)
        self.bc_bottom = Boundary(BoundaryType.VELOCITY_SPECIFIED)
        self.bc_top = Boundary(BoundaryType.VELOCITY_SPECIFIED)

    def make_mesh(self, target_cellsize) -> None:
        """
        Generates a mesh for fluid simulation based on the target cell size.

        Args:
            target_cellsize (float): The desired size of each cell in the mesh.

        Returns:
            None
        """
        self.ncells_x = max(1, int((self.x_max - self.x_min) / target_cellsize + 0.5))
        self.ncells_y = max(1, int((self.y_max - self.y_min) / target_cellsize + 0.5))
        self.dx = (self.x_max - self.x_min) / self.ncells_x
        self.dy = (self.y_max - self.y_min) / self.ncells_y
        self.edge_x_values = np.linspace(self.x_min, self.x_max, self.ncells_x + 1)
        self.edge_y_values = np.linspace(self.y_min, self.y_max, self.ncells_y + 1)
        self.center_x_values = 0.5 * (self.edge_x_values[:-1] + self.edge_x_values[1:])
        self.center_y_values = 0.5 * (self.edge_y_values[:-1] + self.edge_y_values[1:])


class Simulation:
    def __init__(self):
        self.blocks = []  # all the blocks
        self.density = 1.0  # 密度 [kg/m^3]
        self.viscosity = 1.0  # 粘性係数 [Pa*s]
        self.time_end = None  # [s]
        self.step_end = None
        self.dt = 1.0  # [s]
        self.plot_interval = 1  # 可視化の間隔
        self.save_interval = 100  # 保存の間隔
        self.target_cellsize = 0.1

    def save(self, filepath):
            """
            Save the current object to a file using pickle.

            Parameters:
            - filepath (str): The path to the file where the object will be saved.

            Returns:
            - None
            """
            with open(filepath, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath):
        """
        Load the object state from a file.

        Parameters:
            filepath (str): The path to the file containing the object state.

        Returns:
            None
        """
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)

    def connect_blocks(self):
        """
        Connects the blocks in the fluid simulation.

        This method iterates over all combinations of blocks and checks if they can be connected.
        If two blocks have the same y-coordinate range, they are connected horizontally.
        If two blocks have the same x-coordinate range, they are connected vertically.

        Returns:
            None
        """
        for i, j in itertools.combinations(range(len(self.blocks)), 2):
            block_i = self.blocks[i]
            block_j = self.blocks[j]
            if (block_i.y_min, block_i.y_max) == (block_j.y_min, block_j.y_max):
                if block_i.x_max == block_j.x_min:
                    block_i.bc_right = Boundary(BoundaryType.CONNECTED, j)
                    block_j.bc_left = Boundary(BoundaryType.CONNECTED, i)
                elif block_i.x_min == block_j.x_max:
                    block_i.bc_left = Boundary(BoundaryType.CONNECTED, j)
                    block_j.bc_right = Boundary(BoundaryType.CONNECTED, i)
            elif (block_i.x_min, block_i.x_max) == (block_j.x_min, block_j.x_max):
                if block_i.y_max == block_j.y_min:
                    block_i.bc_top = Boundary(BoundaryType.CONNECTED, j)
                    block_j.bc_bottom = Boundary(BoundaryType.CONNECTED, i)
                elif block_i.y_min == block_j.y_max:
                    block_i.bc_bottom = Boundary(BoundaryType.CONNECTED, j)
                    block_j.bc_top = Boundary(BoundaryType.CONNECTED, i)

    def set_domain(self, domain_matrix):
        """
        Sets the domain of the fluid simulation by creating blocks based on the given domain matrix.

        Parameters:
        - domain_matrix (list): A 2D list representing the domain matrix. Each row in the matrix should contain the
                                coordinates (x, y, width, height) of a block.

        Returns:
        - None
        """
        for row in domain_matrix:
            self.blocks.append(Block(row[0], row[1], row[2], row[3]))
        self.connect_blocks()

    def set_physical_values(
        self,
        density,
        viscosity,
        target_cellsize,
        dt,
        time_end,
        plot_interval,
        save_interval,
    ):
        """
        Set the physical values for the fluid simulation.

        Args:
            density (float): The density of the fluid.
            viscosity (float): The viscosity of the fluid.
            target_cellsize (float): The desired size of the simulation cells.
            dt (float): The time step size for the simulation.
            time_end (float): The total duration of the simulation.
            plot_interval (int): The interval at which to plot the simulation results.
            save_interval (int): The interval at which to save the simulation results.

        Returns:
            None
        """
        self.density = density
        self.viscosity = viscosity
        self.target_cellsize = target_cellsize
        self.dt = dt
        self.time_end = time_end
        self.step_end = step_end = int(time_end / dt + 0.5)
        self.plot_interval = plot_interval
        self.save_interval = save_interval

    def set_boundary_conditions(self, index, direction, boundary_type, value):
        """
        Sets the boundary conditions for a specific block in the fluid simulation.

        Args:
            index (int): The index of the block.
            direction (str): The direction of the boundary ("left", "right", "bottom", or "top").
            boundary_type (str): The type of the boundary ("v" for velocity specified, "p" for pressure specified).
            value (list, tuple, np.ndarray, int, float): The value of the boundary condition.

        Raises:
            ValueError: If the value is not of the expected type for the specified boundary type.

        Returns:
            None
        """
        if direction == "left":
            direction = Direction.LEFT
        elif direction == "right":
            direction = Direction.RIGHT
        elif direction == "bottom":
            direction = Direction.BOTTOM
        elif direction == "top":
            direction = Direction.TOP

        if boundary_type == "v":
            boundary_type = BoundaryType.VELOCITY_SPECIFIED
        elif boundary_type == "p":
            boundary_type = BoundaryType.PRESSURE_SPECIFIED

        if (boundary_type == BoundaryType.VELOCITY_SPECIFIED) and (
            type(value) not in [list, tuple, np.ndarray]
        ):
            raise ValueError("value must be a list, tuple or numpy.ndarray")
        elif (boundary_type == BoundaryType.PRESSURE_SPECIFIED) and (
            type(value) not in [int, float]
        ):
            raise ValueError("value must be an int or float")

        if direction == Direction.LEFT:
            self.blocks[index].bc_left = Boundary(boundary_type, value)
        elif direction == Direction.RIGHT:
            self.blocks[index].bc_right = Boundary(boundary_type, value)
        elif direction == Direction.BOTTOM:
            self.blocks[index].bc_bottom = Boundary(boundary_type, value)
        elif direction == Direction.TOP:
            self.blocks[index].bc_top = Boundary(boundary_type, value)

    def set_boundary_conditions_in(self, index_arr, direction, boundary_type, value_arr):
        """
        Sets the boundary conditions for multiple indices.

        Args:
            index_arr (list): A list of indices for which the boundary conditions need to be set.
            direction (str): The direction of the boundary conditions.
            boundary_type (str): The type of the boundary conditions.
            value_arr (list): A list of values corresponding to each index.

        Returns:
            None
        """
        k = 0
        for index in index_arr:
            print(index)
            self.set_boundary_conditions(index, direction, boundary_type, value_arr[k])
            k += 1


    def make_mesh(self):
        """
        Generates a mesh for fluid stimulation.

        This method calculates the mesh for fluid stimulation based on the target cell size and the blocks defined in the class.
        It calculates the total volume of the blocks, creates meshes for each block, and stores the cell offsets, indexes, centers, and other properties.

        Returns:
            None
        """
        target_cellsize = self.target_cellsize
        total_volume = 0.0
        cell_offsets = [0]
        cell_indexes = []
        cell_centers = []
        for ib, block_i in enumerate(self.blocks):
            total_volume += (block_i.x_max - block_i.x_min) * (
                block_i.y_max - block_i.y_min
            )
            block_i.make_mesh(target_cellsize)
            cell_offsets.append(cell_offsets[-1] + block_i.ncells_x * block_i.ncells_y)
            for ix, iy in np.ndindex(block_i.ncells_x, block_i.ncells_y):
                cell_indexes.append([ib, ix, iy])
                cell_centers.append(
                    [block_i.center_x_values[ix], block_i.center_y_values[iy]]
                )
        self.cell_offsets = np.array(cell_offsets)
        self.cell_indexes = np.array(cell_indexes)
        self.cell_centers = np.array(cell_centers)
        self.ncells = len(self.cell_centers)
        self.average_cellsize = np.sqrt(total_volume / self.ncells)

    def initialize(self):
        self.step = 0
        self.time = 0.0  # [s]
        self.velocity = np.zeros((self.ncells, 2))  # 速度 [m/s]
        self.pressure = np.zeros(self.ncells)  # 圧力 [Pa]

    def is_end(self):
        """
        Checks if the simulation has reached its end.

        Returns:
            bool: True if the simulation has reached its end, False otherwise.
        """
        if self.step_end:
            if self.step >= self.step_end:
                return True
        if self.time_end:
            if self.time >= self.time_end:
                return True
        return False

    def plot(self):
        """
        Plot the simulation results.

        This method plots the simulation results, including the mesh, velocity vectors, and pressure.

        Returns:
            None
        """
        # Set the domain of the plot
        x_min = min(block_i.x_min for block_i in self.blocks)
        x_max = max(block_i.x_max for block_i in self.blocks)
        y_min = min(block_i.y_min for block_i in self.blocks)
        y_max = max(block_i.y_max for block_i in self.blocks)
        draw_margin = min((x_max - x_min) / 10, (y_max - y_min) / 10)
        draw_x_min = x_min - draw_margin
        draw_x_max = x_max + draw_margin
        draw_y_min = y_min - draw_margin
        draw_y_max = y_max + draw_margin
        # nrow,ncols = 1,2
        nrow, ncols = 2, 1
        # if global fig is not available, generate a new figure 
        global fig
        if "fig" not in globals():
            fig = plt.figure(figsize=(14.0, 8.0))
        plt.pause(0.001)
        fig.clf()
        fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.15, hspace=0.1
        )
        fig.suptitle(f"Step: {self.step}, Time: {self.time:g} s")
        for k in range(2):
            ax = fig.add_subplot(nrow, ncols, k + 1)
            ax.set_xlim(draw_x_min, draw_x_max)
            ax.set_ylim(draw_y_min, draw_y_max)
            ax.set_aspect("equal")
            # first plot the mesh
            for ib, block_i in enumerate(self.blocks):
                list_x = block_i.edge_x_values
                list_y = block_i.edge_y_values
                for i, x in enumerate(list_x):
                    if i == 0 and block_i.bc_left.type != BoundaryType.CONNECTED:
                        alpha = 1.0
                    elif (
                        i == len(list_x) - 1
                        and block_i.bc_right.type != BoundaryType.CONNECTED
                    ):
                        alpha = 1.0
                    elif (
                        i == len(list_x) - 1
                        and block_i.bc_right.type == BoundaryType.CONNECTED
                    ):
                        continue
                    else:
                        alpha = 0.1
                    ax.plot(
                        [x, x],
                        [block_i.y_min, block_i.y_max],
                        color="black",
                        linewidth=0.5,
                        alpha=alpha,
                    )
                for i, y in enumerate(list_y):
                    if i == 0 and block_i.bc_bottom.type != BoundaryType.CONNECTED:
                        alpha = 1.0
                    elif (
                        i == len(list_y) - 1
                        and block_i.bc_top.type != BoundaryType.CONNECTED
                    ):
                        alpha = 1.0
                    elif (
                        i == len(list_y) - 1
                        and block_i.bc_top.type == BoundaryType.CONNECTED
                    ):
                        continue
                    else:
                        alpha = 0.1
                    ax.plot(
                        [block_i.x_min, block_i.x_max],
                        [y, y],
                        color="black",
                        linewidth=0.5,
                        alpha=alpha,
                    )
            # visualize the velocity vector
            if k == 0:
                label = "Velocity [m/s]"
                vectors = self.velocity.copy()
                vectors_mag = np.linalg.norm(vectors, axis=1)
                nonzero = vectors_mag > 0.0
                vectors[nonzero, 0] /= vectors_mag[nonzero]
                vectors[nonzero, 1] /= vectors_mag[nonzero]
                scale = 1 / (self.average_cellsize * 1.5)
                width = self.average_cellsize / 10
                m = ax.quiver(
                    self.cell_centers[:, 0],
                    self.cell_centers[:, 1],
                    vectors[:, 0],
                    vectors[:, 1],
                    vectors_mag,
                    scale_units="xy",
                    units="xy",
                    scale=scale,
                    width=width,
                    cmap="jet",
                    headwidth=3,
                    headlength=4,
                )
                cbar = fig.colorbar(m, ax=ax, shrink=0.8, orientation="vertical")
                cbar.set_label(label)
            # visualize the pressure using pmeshcolor
            if k == 1:
                label = "Pressure [Pa]"
                scalars = self.pressure
                scalars_min = np.min(scalars)
                scalars_max = np.max(scalars)
                if scalars_min == scalars_max:
                    scalars_min -= 0.01
                    scalars_max += 0.01
                for ib, block_i in enumerate(self.blocks):
                    block_scalars = scalars[
                        self.cell_offsets[ib] : self.cell_offsets[ib + 1]
                    ].reshape(block_i.ncells_x, block_i.ncells_y)
                    list_x = block_i.edge_x_values
                    list_y = block_i.edge_y_values
                    xx, yy = np.meshgrid(list_x, list_y, indexing="ij")
                    m = ax.pcolormesh(
                        xx,
                        yy,
                        block_scalars,
                        cmap="jet",
                        shading="flat",
                        vmin=scalars_min,
                        vmax=scalars_max,
                    )
                cbar = fig.colorbar(m, ax=ax, shrink=0.8, orientation="vertical")
                cbar.set_label(label)
        # plt.show()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def cell_id(self, ib, ix, iy):
        """
        Calculates the cell ID based on the block index (ib), x-coordinate (ix), and y-coordinate (iy).

        Parameters:
            ib (int): The block index.
            ix (int): The x-coordinate.
            iy (int): The y-coordinate.

        Returns:
            int: The calculated cell ID.
        """
        return self.cell_offsets[ib] + ix * self.blocks[ib].ncells_y + iy

    def neighbor_cells(self, i):
        """
        Generates neighboring cells for a given cell index.

        Args:
            i (int): Index of the cell.

        Yields:
            tuple: A tuple containing information about the neighboring cell.
                - If the neighboring cell exists, the tuple contains:
                    - None: Indicates no boundary condition on the current cell's side.
                    - cell_id: The ID of the neighboring cell.
                    - dy: The height of the neighboring cell.
                    - dx: The width of the neighboring cell.
                - If the neighboring cell does not exist, the tuple contains:
                    - boundary: The boundary condition on the current cell's side.
                    - None: Indicates no neighboring cell.
                    - dy: The height of the current cell.
                    - dx: The width of the current cell.
        """
        ib, ix, iy = self.cell_indexes[i]
        block_i = self.blocks[ib]
        # left neighbor
        if ix > 0:
            yield None, self.cell_id(ib, ix - 1, iy), block_i.dy, block_i.dx
        elif block_i.bc_left.type == BoundaryType.CONNECTED:
            jb = block_i.bc_left.block_id
            block_j = self.blocks[jb]
            yield None, self.cell_id(jb, block_j.ncells_x - 1, iy), block_i.dy, 0.5 * (
                block_i.dx + block_j.dx
            )
        else:
            yield block_i.bc_left, None, block_i.dy, block_i.dx
        # right neighbor
        if ix < block_i.ncells_x - 1:
            yield None, self.cell_id(ib, ix + 1, iy), block_i.dy, block_i.dx
        elif block_i.bc_right.type == BoundaryType.CONNECTED:
            jb = block_i.bc_right.block_id
            block_j = self.blocks[jb]
            yield None, self.cell_id(jb, 0, iy), block_i.dy, 0.5 * (
                block_i.dx + block_j.dx
            )
        else:
            yield block_i.bc_right, None, block_i.dy, block_i.dx
        # bottom neighbor
        if iy > 0:
            yield None, self.cell_id(ib, ix, iy - 1), block_i.dx, block_i.dy
        elif block_i.bc_bottom.type == BoundaryType.CONNECTED:
            jb = block_i.bc_bottom.block_id
            block_j = self.blocks[jb]
            yield None, self.cell_id(jb, ix, block_j.ncells_y - 1), 0.5 * (
                block_i.dy + block_j.dy
            ), block_i.dx
        else:
            yield block_i.bc_bottom, None, block_i.dx, block_i.dy
        # top neighbor
        if iy < block_i.ncells_y - 1:
            yield None, self.cell_id(ib, ix, iy + 1), block_i.dx, block_i.dy
        elif block_i.bc_top.type == BoundaryType.CONNECTED:
            jb = block_i.bc_top.block_id
            block_j = self.blocks[jb]
            yield None, self.cell_id(jb, ix, 0), 0.5 * (
                block_i.dy + block_j.dy
            ), block_i.dx
        else:
            yield block_i.bc_top, None, block_i.dx, block_i.dy

    def neighbor_velocities(self, i):
        """
        Returns the neighbor velocities for a given cell index.

        Args:
            i (int): The index of the cell.

        Yields:
            tuple: A tuple containing the boundary condition, neighbor index, S_ij, d_ij, and velocity.

        Raises:
            ValueError: If an invalid boundary type is encountered.
        """
        for bc, j, S_ij, d_ij in self.neighbor_cells(i):
            if bc is None:
                yield bc, j, S_ij, d_ij, self.velocity[j]
            elif bc.type == BoundaryType.VELOCITY_SPECIFIED:
                yield bc, j, S_ij, d_ij, bc.velocity * 2 - self.velocity[i]
            elif bc.type == BoundaryType.PRESSURE_SPECIFIED:
                yield bc, j, S_ij, d_ij, self.velocity[i]
            else:
                raise ValueError("Invalid boundary type")

    def neighbor_pressures(self, i):
        """
        Generates neighbor pressures for a given cell index.

        Args:
            i (int): Index of the cell.

        Yields:
            tuple: A tuple containing the boundary condition, neighbor index, S_ij, d_ij, and pressure.

        Raises:
            ValueError: If an invalid boundary type is encountered.
        """
        for bc, j, S_ij, d_ij in self.neighbor_cells(i):
            if bc is None:
                yield bc, j, S_ij, d_ij, self.pressure[j]
            elif bc.type == BoundaryType.VELOCITY_SPECIFIED:
                yield bc, j, S_ij, d_ij, self.pressure[i]
            elif bc.type == BoundaryType.PRESSURE_SPECIFIED:
                yield bc, j, S_ij, d_ij, bc.pressure * 2 - self.pressure[i]
            else:
                raise ValueError("Invalid boundary type")

    def check(self) -> None:
        """
        Checks the validity of the Courant and diffusion numbers for fluid simulation.

        Raises:
            ValueError: If the Courant number is greater than 1.0 or the diffusion number is greater than 0.5.
        """
        U_max = 1.0  # [m/s]
        nu = self.viscosity / self.density
        courant_number = U_max * self.dt / self.target_cellsize
        diffusion_number = nu * self.dt / self.target_cellsize**2
        print(f"courant_number = {courant_number}")
        print(f"diffusion_number = {diffusion_number}")
        if courant_number > 1.0:
            raise ValueError("Invalid Courant number")
        if diffusion_number > 0.5:
            raise ValueError("Invalid diffusion number")

    def solve_poisson(self):
        """
        Solve the Poisson equation for pressure.

        This method calculates the coefficient matrix and the right-hand side vector
        for the Poisson equation based on the specified boundary conditions. It then
        solves the equation and returns the coefficient matrix, the right-hand side
        vector, and a flag indicating whether the coefficient matrix is singular.

        Returns:
            ppe_A (scipy.sparse.csr_matrix): The coefficient matrix of the Poisson equation.
            ppe_b (numpy.ndarray): The right-hand side vector of the Poisson equation.
            ppe_is_singular (bool): Flag indicating whether the coefficient matrix is singular.
        """
        comp_start = time.perf_counter()
        N = self.ncells
        ppe_is_singular = True  # Whether the coefficient matrix is singular
        for i in range(self.ncells):
            for bc, j, S_ij, d_ij in self.neighbor_cells(i):
                if bc and bc.type == BoundaryType.PRESSURE_SPECIFIED:
                    ppe_is_singular = False
                    break
            if not ppe_is_singular:
                break
        if ppe_is_singular:
            ppe_A = scipy.sparse.lil_matrix((N + 1, N + 1))  # Coefficient matrix
            ppe_b = np.zeros(N + 1)  # Right-hand side vector (boundary conditions)
        else:
            ppe_A = scipy.sparse.lil_matrix((N, N))  # Coefficient matrix
            ppe_b = np.zeros(N)  # Right-hand side vector (boundary conditions)
        for i in range(self.ncells):
            for bc, j, S_ij, d_ij in self.neighbor_cells(i):
                if bc is None:
                    ppe_A[i, j] += S_ij / d_ij
                    ppe_A[i, i] -= S_ij / d_ij
                elif bc.type == BoundaryType.VELOCITY_SPECIFIED:
                    pass
                elif bc.type == BoundaryType.PRESSURE_SPECIFIED:
                    ppe_A[i, i] -= 2 * S_ij / d_ij
                    ppe_b[i] -= bc.pressure * 2 * S_ij / d_ij
                else:
                    raise ValueError("Invalid boundary type")
        ppe_A = ppe_A.tocsr()
        comp_end = time.perf_counter()
        print(f"PPE comptime: {comp_end-comp_start:.6f} s")
        return ppe_A, ppe_b, ppe_is_singular

    def simulate(self, filepath, ppe_A, ppe_b, ppe_is_singular):
        while not self.is_end():
            comptimes = []
            comp_start = time.perf_counter()
            # (1) 仮速度を計算する
            convU = np.zeros((self.ncells, 2))  # 対流項
            lapU = np.zeros((self.ncells, 2))  # 速度のラプラシアン
            for ib, block_i in enumerate(self.blocks):
                dx0 = block_i.dx
                dy0 = block_i.dy
                V = dx0 * dy0
                for i in range(self.cell_offsets[ib], self.cell_offsets[ib + 1]):
                    u0 = self.velocity[i]  # 中心
                    neighbors = self.neighbor_velocities(i)
                    *_, d1, u1 = next(neighbors)
                    *_, d2, u2 = next(neighbors)
                    *_, d3, u3 = next(neighbors)
                    *_, d4, u4 = next(neighbors)
                    convU[i] = calc_convU(dx0, dy0, d1, d2, d3, d4, u0, u1, u2, u3, u4)
                    lapU[i] = calc_lapU(dx0, dy0, d1, d2, d3, d4, u0, u1, u2, u3, u4)
            self.velocity += self.dt * (-convU + lapU * (self.viscosity / self.density))

            comp_end = time.perf_counter()
            comptimes.append(comp_end - comp_start)
            comp_start = time.perf_counter()

            # (2) 圧力を求める
            divU = np.zeros(self.ncells)  # 仮速度の発散
            for ib, block_i in enumerate(self.blocks):
                dx0 = block_i.dx
                dy0 = block_i.dy
                V = dx0 * dy0
                for i in range(self.cell_offsets[ib], self.cell_offsets[ib + 1]):
                    u0 = self.velocity[i]  # 中心
                    neighbors = self.neighbor_velocities(i)
                    *_, d1, u1 = next(neighbors)
                    *_, d2, u2 = next(neighbors)
                    *_, d3, u3 = next(neighbors)
                    *_, d4, u4 = next(neighbors)
                    divU[i] = calc_divU(dx0, dy0, d1, d2, d3, d4, u0, u1, u2, u3, u4)
            if ppe_is_singular:
                tmp = np.zeros(self.ncells + 1)
                tmp[: self.ncells] = divU * (self.density / self.dt)
                tmp += ppe_b
                self.pressure = scipy.sparse.linalg.spsolve(ppe_A, tmp)[: self.ncells]
            else:
                self.pressure = scipy.sparse.linalg.spsolve(
                    ppe_A, ppe_b + divU * (self.density / self.dt)
                )

            comp_end = time.perf_counter()
            comptimes.append(comp_end - comp_start)
            comp_start = time.perf_counter()

            # (3) 速度を修正する
            gradP = np.zeros((self.ncells, 2))  # 圧力勾配
            for ib, block_i in enumerate(self.blocks):
                dx0 = block_i.dx
                dy0 = block_i.dy
                V = dx0 * dy0
                for i in range(self.cell_offsets[ib], self.cell_offsets[ib + 1]):
                    p0 = self.pressure[i]  # 中心
                    neighbors = self.neighbor_pressures(i)
                    *_, d1, p1 = next(neighbors)
                    *_, d2, p2 = next(neighbors)
                    *_, d3, p3 = next(neighbors)
                    *_, d4, p4 = next(neighbors)
                    gradP[i] = calc_gradP(dx0, dy0, d1, d2, d3, d4, p0, p1, p2, p3, p4)
            self.velocity -= gradP * (self.dt / self.density)

            comp_end = time.perf_counter()
            comptimes.append(comp_end - comp_start)
            comp_start = time.perf_counter()

            # (4) 時刻を進める
            self.step += 1
            self.time += self.dt
            print(f"step: {self.step}, time: {self.time:g} s", end=", ")
            print(f"comptimes: {comptimes}, {sum(comptimes):.6f} s")
            if self.step % self.plot_interval == 0:
                self.plot()
            if self.step % self.save_interval == 0:
                self.save(filepath)


@jit(nopython=True, cache=True)
def calc_convU(dx0, dy0, d1, d2, d3, d4, u0, u1, u2, u3, u4):
    """
    Calculate the convolution of the velocity field at each cell.

    Parameters:
    dx0, dy0 (float): Cell size in x and y directions.
    d1, d2, d3, d4 (float): Distances to neighboring cells in x and y directions.
    u0, u1, u2, u3, u4 (float): Velocity values at the center and neighboring cells.

    Returns:
    np.ndarray: Convolution of the velocity field at the center cell.

    This function calculates the convolution of the velocity field at the center cell based on the velocity values of the neighboring cells. It uses the distances to the neighboring cells (d1, d2, d3, d4) and the velocity values (u0, u1, u2, u3, u4) to compute the convolution. The resulting convolution is then returned as an np.ndarray.
    """
    S1, S2, S3, S4 = dy0, dy0, dx0, dx0
    dx1 = d1 * 2 - dx0
    dx2 = d2 * 2 - dx0
    dy3 = d3 * 2 - dy0
    dy4 = d4 * 2 - dy0
    f1 = -(u0[0] * dx1 + u1[0] * dx0) / (dx1 + dx0)
    f2 = +(u0[0] * dx2 + u2[0] * dx0) / (dx2 + dx0)
    f3 = -(u0[1] * dy3 + u3[1] * dy0) / (dy3 + dy0)
    f4 = +(u0[1] * dy4 + u4[1] * dy0) / (dy4 + dy0)
    convU_i = np.array([0.0, 0.0])
    convU_i += S1 * f1 * u0 if f1 >= 0 else S1 * f1 * u1
    convU_i += S2 * f2 * u0 if f2 >= 0 else S2 * f2 * u2
    convU_i += S3 * f3 * u0 if f3 >= 0 else S3 * f3 * u3
    convU_i += S4 * f4 * u0 if f4 >= 0 else S4 * f4 * u4
    convU_i /= dx0 * dy0
    return convU_i


@jit(nopython=True, cache=True)
def calc_lapU(dx0, dy0, d1, d2, d3, d4, u0, u1, u2, u3, u4):
    """
    Calculate the Laplacian of the velocity field at each cell.

    Parameters:
    dx0, dy0 (float): Cell size in x and y directions.
    d1, d2, d3, d4 (float): Distances to neighboring cells in x and y directions.
    u0, u1, u2, u3, u4 (float): Velocity values at the center and neighboring cells.

    Returns:
    np.ndarray: Laplacian of the velocity field at the center cell.

    This function calculates the Laplacian of the velocity field at the center cell based on the velocity values of the neighboring cells. It uses the distances to the neighboring cells (d1, d2, d3, d4) and the velocity values (u0, u1, u2, u3, u4) to compute the Laplacian components in the x and y directions. The resulting Laplacian is then returned as an np.ndarray.
    """
    S1, S2, S3, S4 = dy0, dy0, dx0, dx0
    lapU_i = np.array([0.0, 0.0])
    lapU_i += (u1 - u0) * (S1 / d1)
    lapU_i += (u2 - u0) * (S2 / d2)
    lapU_i += (u3 - u0) * (S3 / d3)
    lapU_i += (u4 - u0) * (S4 / d4)
    lapU_i /= dx0 * dy0
    return lapU_i


@jit(nopython=True, cache=True)
def calc_divU(dx0, dy0, d1, d2, d3, d4, u0, u1, u2, u3, u4):
    """
    Calculate the divergence of the velocity field at each cell.

    Parameters:
    dx0, dy0 (float): Cell size in x and y directions.
    d1, d2, d3, d4 (float): Distances to neighboring cells in x and y directions.
    u0, u1, u2, u3, u4 (float): Velocity values at the center and neighboring cells.

    Returns:
    np.ndarray: Divergence of the velocity field at the center cell.

    This function calculates the divergence of the velocity field at the center cell based on the velocity values of the neighboring cells. It uses the distances to the neighboring cells (d1, d2, d3, d4) and the velocity values (u0, u1, u2, u3, u4) to compute the divergence. The resulting divergence is then returned as an np.ndarray.
    """
    S1, S2, S3, S4 = dy0, dy0, dx0, dx0
    dx1 = d1 * 2 - dx0
    dx2 = d2 * 2 - dx0
    dy3 = d3 * 2 - dy0
    dy4 = d4 * 2 - dy0
    f1 = -(u0[0] * dx1 + u1[0] * dx0) / (dx1 + dx0)
    f2 = +(u0[0] * dx2 + u2[0] * dx0) / (dx2 + dx0)
    f3 = -(u0[1] * dy3 + u3[1] * dy0) / (dy3 + dy0)
    f4 = +(u0[1] * dy4 + u4[1] * dy0) / (dy4 + dy0)
    divU_i = S1 * f1 + S2 * f2 + S3 * f3 + S4 * f4
    return divU_i


@jit(nopython=True, cache=True)
def calc_gradP(dx0, dy0, d1, d2, d3, d4, p0, p1, p2, p3, p4):
    """
    Calculate the gradient of pressure (plikachar) at each cell.

    Parameters:
    dx0, dy0 (float): Cell size in x and y directions.
    d1, d2, d3, d4 (float): Distances to neighboring cells in x and y directions.
    p0, p1, p2, p3, p4 (float): Pressure values at the center and neighboring cells.

    Returns:
    np.ndarray: Gradient of pressure at the center cell.

    This function calculates the gradient of pressure at the center cell based on the pressure values of the neighboring cells. It uses the distances to the neighboring cells (d1, d2, d3, d4) and the pressure values (p0, p1, p2, p3, p4) to compute the gradient components in the x and y directions. The resulting gradient is then returned as an np.ndarray.
    """
    S1, S2, S3, S4 = dy0, dy0, dx0, dx0
    dx1 = d1 * 2 - dx0
    dx2 = d2 * 2 - dx0
    dy3 = d3 * 2 - dy0
    dy4 = d4 * 2 - dy0
    f1 = -(p0 * dx1 + p1 * dx0) / (dx1 + dx0)
    f2 = +(p0 * dx2 + p2 * dx0) / (dx2 + dx0)
    f3 = -(p0 * dy3 + p3 * dy0) / (dy3 + dy0)
    f4 = +(p0 * dy4 + p4 * dy0) / (dy4 + dy0)
    gradP_i = np.array([S1 * f1 + S2 * f2, S3 * f3 + S4 * f4])
    gradP_i /= dx0 * dy0
    return gradP_i

def main():
    """
    0. Initialize the simulation object
    """
    sim = Simulation()

    """" 
    1. Set the computational domain
    """
    x0, x1, x2, x3, x4, x5, x6, x7 = 0.0, 1.0, 1.1, 1.3, 1.7, 1.9, 2.0, 20.0
    y0, y1, y2, y3, y4, y5, y6, y7 = 0.0, 1.1, 1.2, 1.4, 1.8, 2.0, 2.1, 3.0
    domain_matrix = np.array(
        [
            # the firsr layer
            [x0, x1, y0, y1],  # 0
            [x1, x2, y0, y1],  # 1
            [x2, x3, y0, y1],  # 2
            [x3, x4, y0, y1],  # 3
            [x4, x5, y0, y1],  # 4
            [x5, x6, y0, y1],  # 5
            [x6, x7, y0, y1],  # 6
            # the second layer
            [x0, x1, y1, y2],  # 7
            [x1, x2, y1, y2],  # 8
            [x2, x3, y1, y2],  # 9
            [x4, x5, y1, y2],  # 10
            [x5, x6, y1, y2],  # 11
            [x6, x7, y1, y2],  # 12
            # the third layer
            [x0, x1, y2, y3],  # 13
            [x1, x2, y2, y3],  # 14
            [x5, x6, y2, y3],  # 15
            [x6, x7, y2, y3],  # 16
            # the fourth layer
            [x0, x1, y3, y4],  # 17
            [x6, x7, y3, y4],  # 18
            # the fifth layer
            [x0, x1, y4, y5],  # 19
            [x1, x2, y4, y5],  # 20
            [x5, x6, y4, y5],  # 21
            [x6, x7, y4, y5],  # 22
            # the sixth layer
            [x0, x1, y5, y6],  # 23
            [x1, x2, y5, y6],  # 24
            [x2, x3, y5, y6],  # 25
            [x4, x5, y5, y6],  # 26
            [x5, x6, y5, y6],  # 27
            [x6, x7, y5, y6],  # 28
            # the seventh layer
            [x0, x1, y6, y7],  # 29
            [x1, x2, y6, y7],  # 30
            [x2, x3, y6, y7],  # 31
            [x3, x4, y6, y7],  # 32
            [x4, x5, y6, y7],  # 33
            [x5, x6, y6, y7],  # 34
            [x6, x7, y6, y7],  # 35
        ]
    ) 
    sim.set_domain(domain_matrix)

    """ 
    2. Set the boundary conditions
    """
    try:
        # set_boudary_conditions_in method allow us to set many boundary conditions in one time
        # left boundary
        sim.set_boundary_conditions_in(index_arr=[0, 7, 13, 17, 19, 23, 29],
                                       direction="left",
                                       boundary_type="v",
                                       value_arr=[[1.0, 0.0] for i in range(7)])
        # bottom boundary
        sim.set_boundary_conditions_in(index_arr=[0, 1, 2, 3, 4, 5, 6],
                                       direction="bottom",
                                       boundary_type="v",
                                       value_arr=[[1.0, 0.0] for i in range(7)])
        # right boundary
        sim.set_boundary_conditions_in(index_arr=[6, 12, 16, 18, 22, 28, 35],
                                       direction="right",
                                       boundary_type="p",
                                       value_arr=[0.0 for i in range(7)])
        # top boundary
        sim.set_boundary_conditions_in(index_arr=[29, 30, 31, 32, 33, 34, 35],
                                       direction="top",
                                       boundary_type="v",
                                       value_arr=[[1.0, 0.0] for i in range(7)])
    except Exception:
        print(f'{type(Exception)}' + "are encountered")
        return

    """
    3. Set physical values and check the numerical stability
    """
    # set physical values
    sim.set_physical_values(
        density=1.0,
        viscosity=0.005,
        target_cellsize=0.1,
        dt=0.025,
        time_end=1000000.0,
        plot_interval=10,
        save_interval=100,
    )

    # check the numerical stability
    try:
        sim.check()
    except ValueError as v_error:
        print(f'{type(v_error)}' + "are encountered")
        return

    """
    4. Generate the mesh structure and initialize it
    """
    # generate the mesh structure
    sim.make_mesh()

    # initialize it
    sim.initialize()

    # check if previous data exists and if it does, load the data from the file.
    filepath = pathlib.Path("sim.pickle")
    if pathlib.Path(filepath).exists():
        sim.load(filepath)


    """
    5. Solve the Poisson Equation for pressure
    """
    # The Poisson Equation for pressure
    ppe_A, ppe_b, ppe_is_singular = sim.solve_poisson()

    """
    6. Simulate the fluid
    """
    sim.simulate("my_demo.pickle", ppe_A, ppe_b, ppe_is_singular)

    """
    7. Plot the results
    """
    sim.plot()
    plt.show()

if __name__ == "__main__":
    main()
