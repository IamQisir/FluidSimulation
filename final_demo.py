from cfd import *

def fluid_simulate():
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
    fluid_simulate()