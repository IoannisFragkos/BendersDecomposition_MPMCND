from __future__ import division

from helpers import read_data, get_2d_index
from sys import argv
from time import time
from itertools import product
from gurobipy import Model, GRB, quicksum, LinExpr, GurobiError
from graph_helpers import make_graph
from collections import namedtuple
import numpy as np

__author__ = 'ioannis'
# File to gather data and Implement Benders Decomposition
# Ioannis Fragkos, June 2015

# Container that holds the subproblem dual vectors. The vectors should be
# entered as numpy arrays.
# Dimensions: flow_duals(node, commodity, period); capacity_duals(period, arc)
# We carry numpy arrays of Subproblem_Duals so that we add many cuts at once
Subproblem_Duals = namedtuple('Subproblem_Duals', 'flow_duals capacity_duals '
                                                  'bounds_duals '
                                                  'optimality_dual')


def main():
    filename = 'R_single_period/r01.2.dow' if len(argv) <= 1 else argv[1]
    data = read_data(filename)
    start = time()
    data.graph = make_graph(data)
    master = populate_master(data, None)
    subproblem = populate_dual_subproblem(data, None)
    master_callback = callback_data(subproblem, data)
    master.optimize(master_callback)
    stop = time()
    print 'Total time: {} seconds'.format(round(stop - start, 0))


def populate_master(data, duals):
    """
    Function that populates the Benders Master problem
    :param duals:  Array of dual values used to initialize the master
                   constraints. If not provided, we set it to zero. If
                   provided, we assume it is optimality cuts (why?)
    :rtype:        Gurobi model object
    """
    master = Model('master-model')
    arcs, periods = xrange(data.arcs.size), xrange(data.periods)
    variables = np.empty(shape=(data.periods, data.arcs.size), dtype=object)

    # Add variables
    for period, arc in product(periods, arcs):
        variables[period, arc] = master.addVar(vtype=GRB.BINARY,
                                               obj=data.fixed_cost[period, arc],
                                               name='arc_open{}_{}'.format(
                                                   period, arc))
    # Continuous flow_cost variable
    master.addVar(lb=0., obj=1., vtype=GRB.CONTINUOUS, name='flow_cost')
    master.update()

    # Add constraints
    for arc in arcs:
        lhs = LinExpr()
        for period in periods:
            lhs.addTerms(1., variables[period, arc])
        master.addConstr(lhs=lhs, rhs=1., sense=GRB.LESS_EQUAL,
                         name='arc{}'.format(arc))

    # If an array of initial dual vectors is given, add them as cuts:
    # sum{t in T, (i,j) in A} [y{ijt} * cap{ij} *
    # sum{l=t to |T|} capacity_duals{ijl}] - flow_cost <=
    # sum{t, i, k)} b{ik} * flow_duals{ikt}, bik in {1, 0, -1}
    if duals is not None:
        for count, dual in enumerate(xrange(duals.size)):
            rhs, lhs = populate_benders_cut(duals, variables, data)
            master.addConstr(lhs=lhs, rhs=rhs, sense=GRB.LESS_EQUAL,
                             name='heuristic_{}'.format(count))

    master.params.LazyConstraints = 1
    master.setParam('Presolve', 0)
    master.update()
    # Store the variables inside the model, we cannot access them later!
    master._variables = master.getVars()
    return master


def populate_subproblem(data, open_arcs):
    """
    Function that populates the Benders Subproblem
    :param open_arcs:   Array that is used to initialize the subproblem rhs.
                        Typically, we should get it from a heuristic solution.
                        We set it to a zero array if it is not initialized
    :return:            Gurobi model object
    """

    subproblem = Model('subproblem')

    # Ranges we are going to need in order to define the variables and
    # constraints elegantly
    arcs, periods, commodities, nodes = xrange(data.arcs.size), xrange(
        data.periods), xrange(data.commodities), xrange(data.nodes)
    # Get the o-d pairs from data
    origins, destinations = data.origins, data.destinations
    count = 0

    # We need to store the rhs coefficients of the capacity constraints
    # somewhere, so that we can retrieve them later during the callback. It
    # seems that the only possible way to do this is to add a "private"
    # member to the gurobi model.. here it is then
    capacity_rhs_array = np.zeros(
        shape=(data.periods, data.arcs.size), dtype=float)

    # Check if a set of open arcs is provided. If not, initialize it to zero
    if open_arcs is None:
        open_arcs = np.zeros(shape=(len(periods), len(arcs)))

    # Add variables
    for period, arc, commodity in product(periods, arcs, commodities):
        cost = data.variable_cost[arc] * data.demand[period, commodity]
        subproblem.addVar(
            lb=0., obj=cost, vtype=GRB.CONTINUOUS,
            name='x_{}_{}_{}'.format(period, arc, commodity))

    subproblem.update()

    variables = np.array(subproblem.getVars(), dtype=object).reshape((
        len(periods), len(arcs), len(commodities)))

    # Adding constraints: Flow balance
    for node in nodes:
        in_arcs = get_2d_index(data.arcs, data.nodes)[1] == node + 1
        out_arcs = get_2d_index(data.arcs, data.nodes)[0] == node + 1
        for commodity in commodities:
            rhs = 0.
            if node == origins[commodity]:
                rhs = 1.
            if node == destinations[commodity]:
                rhs = -1.
            for period in periods:
                lhs = quicksum(variables[period, in_arcs, commodity]) - \
                      quicksum(variables[period, out_arcs, commodity])

                subproblem.addConstr(
                    lhs=lhs, rhs=rhs, sense=GRB.EQUAL,
                    name='fl_n{}_c{}_p{}_{}'.format(node, commodity, period,
                                                    count))
                count += 1

    # Adding constraints: Arc capacities
    for period, arc in product(periods, arcs):
        lhs = quicksum(LinExpr(
            data.demand[period, c], variables[period, arc, c]
        ) for c in commodities)
        rhs = data.capacity[arc] * np.sum(open_arcs[:period + 1, arc], axis=0)
        capacity_rhs_array[period, arcs] = rhs

        subproblem.addConstr(
            lhs=lhs, rhs=rhs, sense=GRB.LESS_EQUAL,
            name='cap_p{}_a{}_{}'.format(period, arc, count))
        count += 1

    # OK, this is very inefficient, just a hack to see if upper bound
    # constraints make a difference. I should delete later on
    variables = subproblem.getVars()
    count = 0
    for period, arc, commodity in product(periods, arcs, commodities):
        subproblem.addConstr(variables[count] <= 1.)
        count += 1

    # Turn off presolve to make sure we get the right duals
    subproblem.setParam('Presolve', 0)
    # Switch on the additional parameters that calculate dual values when
    # then dual problem is unbounded
    subproblem.setParam('InfUnbdInfo', 1)
    # Save the rhs array under the model object
    subproblem._capacity_rhs_array = capacity_rhs_array
    subproblem.update()

    # subproblem.write('benders_subproblem.lp')

    return subproblem


def populate_dual_subproblem(data, open_arcs, flow_cost=0):
    """
    Function that populates the Benders Dual Subproblem, as suggested by the
    paper "Minimal Infeasible Subsystems and Bender's cuts" by Fischetti,
    Salvagnin and Zanette.
    :param open_arcs:   Array that is used to initialize the subproblem rhs.
                        Typically, we should get it from a heuristic solution.
                        We set it to a zero array if it is not initialized
    :param flow_cost:   This is the cost of the continuous variable of the
                        master problem, as explained in the paper
    :return:            Gurobi model object
    """

    dual_subproblem = Model('dual_subproblem')

    # Ranges we are going to need
    arcs, periods, commodities, nodes = xrange(data.arcs.size), xrange(
        data.periods), xrange(data.commodities), xrange(data.nodes)

    # We use arrays to store variable indexes and variable objects. Why use
    # both? Gurobi wont let us get the values of individual variables
    # within a callback.. We just get the values of a large array of
    # variables, in the order they were initially defined. To separate them
    # in variable categories, we will have to use index arrays
    flow_index = np.zeros(shape=(data.nodes, data.commodities, data.periods),
                          dtype=int)
    flow_duals = np.empty_like(flow_index, dtype=object)
    capacity_index = np.zeros(shape=(data.periods, data.arcs.size), dtype=int)
    capacity_duals = np.empty_like(capacity_index, dtype=object)
    ubounds_index = np.zeros(shape=(len(arcs), data.commodities, data.periods),
                             dtype=int)
    ubounds_duals = np.empty_like(ubounds_index, dtype=object)

    # Makes sure we don't add variables more than once
    flow_duals_names = set()

    if open_arcs is None:
        open_arcs = np.zeros(shape=(len(periods), len(arcs)))

    # Populate all variables in one loop, keep track of their indexes
    count = 0
    for period, arc in product(periods, arcs):
        obj = - np.sum(open_arcs[:period + 1, arc], axis=0) * data.capacity[arc]
        capacity_duals[period, arc] = dual_subproblem.addVar(
            obj=obj, name='capacity_dual_p{}a{}'.format(period, arc))
        capacity_index[period, arc] = count
        count += 1
        for commodity in commodities:
            start_node, end_node = get_2d_index(data.arcs[arc], data.nodes)
            start_node, end_node = start_node - 1, end_node - 1
            for node in (start_node, end_node):
                var_name = 'flow_dual_n{}c{}p{}'.format(node, commodity, period)
                if var_name not in flow_duals_names:
                    flow_duals_names.add(var_name)
                    obj = 0.
                    if data.origins[commodity] == node:
                        obj = 1.
                    if data.destinations[commodity] == node:
                        obj = -1.
                    flow_duals[node, commodity, period] = \
                        dual_subproblem.addVar(
                            obj=obj, lb=-GRB.INFINITY, name=var_name)
                    flow_index[node, commodity, period] = count
                    count += 1
            ubounds_duals[arc, commodity, period] = dual_subproblem.addVar(
                obj=-1., name='u_bound_dual_a{}c{}p{}'.format(arc, commodity,
                                                              period))
            ubounds_index[arc, commodity, period] = count
            count += 1
    opt_var = dual_subproblem.addVar(obj=-flow_cost, name='optimality_var')
    dual_subproblem.update()

    for arc, commodity, period in product(arcs, commodities, periods):
        start_node, end_node = get_2d_index(data.arcs[arc], data.nodes)
        start_node, end_node = start_node - 1, end_node - 1
        demand = data.demand[period, commodity]
        lhs = flow_duals[start_node, commodity, period] \
              - flow_duals[end_node, commodity, period] \
              - capacity_duals[period, arc] * demand - \
              ubounds_duals[arc, commodity, period] - \
              opt_var * data.variable_cost[arc] * demand
        dual_subproblem.addConstr(
            lhs <= 0., name='flow_a{}c{}p{}'.format(
                arc, commodity, period))

    # Original Fischetti model
    lhs = np.sum(capacity_duals) + opt_var
    dual_subproblem.addConstr(lhs == 1, name='normalization_constraint')

    dual_subproblem._capacity_index = capacity_index
    dual_subproblem._flow_index = flow_index
    dual_subproblem._ubounds_index = ubounds_index

    dual_subproblem.setParam('OutputFlag', 0)
    # Switch on the additional parameters that calculate dual values when
    # then dual problem is unbounded
    # dual_subproblem.setParam('PreSolve', 0)
    dual_subproblem.setParam('InfUnbdInfo', 1)
    dual_subproblem.modelSense = GRB.MAXIMIZE
    dual_subproblem.update()
    # dual_subproblem.write('benders-dual.lp')

    return dual_subproblem


def testing_duals(data):
    """
    Just a small tester to fill up the master problem with initial dual prices
    :return: numpy array of Subproblem_Duals objects
    """

    periods, arcs, nodes, commodities = \
        data.periods, data.arcs.size, data.nodes, data.commodities

    solutions = 10
    duals = np.empty(dtype=object, shape=10)

    for solution in xrange(solutions):
        flow_duals = np.random.random_integers(
            low=0, high=100, size=(nodes, commodities, periods))
        capacity_duals = np.random.random_integers(
            low=0, high=100, size=(periods, arcs))
        # duals[solution] = Subproblem_Duals(
        # flow_duals=flow_duals, capacity_duals=capacity_duals)

    return duals


def callback_data(subproblem, data):
    """
    This is a closure that passes whatever data we want to the actual
    callback function. We have to use this because gurobi callbacks have a
    certain signature (model, where)
    :param subproblem: Gurobi subproblem model
    :param data:       Problem data
    :return:           master_callback function

    """

    def solve_subproblem(open_arcs, flow_cost=0):
        """
        Solves the benders subproblem. If infeasible, it returns a feasibility
        cut, i.e., a set of dual prices and a status that the subproblem is
        infeasible. If feasible, and the objective function equals the flow cost
        of the master, we (optionally) store the solution and  accept the new
        incumbent (stating we have an incumbent). If feasible, and the objective
        function is higher than flow_cost, we add an optimality cut.
        :param flow_cost:   Continuous variable of Benders master problem
        :param open_arcs:   Arcs that are open at the master incumbent
        :return:            gurobi status message, Subproblem_Duals object
        """

        arcs, periods = xrange(data.arcs.size), xrange(data.periods)
        capacity_rhs_array = subproblem._capacity_rhs_array
        # Capacity constraints start after the flow balance constraints
        count = data.nodes * data.commodities * data.periods
        constraints = subproblem.getConstrs()

        # Modify the rhs coefficients according to the new open_arcs values
        for period, arc in product(periods, arcs):
            sum_of_arcs = np.sum(open_arcs[:period + 1, arc], axis=0)
            if abs(capacity_rhs_array[period, arc] - data.capacity[arc] *
                    sum_of_arcs) > 10e-3:
                constraints[count].rhs = data.capacity[arc] * sum_of_arcs
                capacity_rhs_array[count] = constraints[count].rhs
            count += 1

        subproblem.update()
        subproblem.optimize()
        status = subproblem.status

        start_cap_cons = data.nodes * data.commodities * data.periods

        if status == GRB.status.OPTIMAL:
            subproblem_objective = subproblem.objVal
            if flow_cost < subproblem_objective - 10e-4:
                # We need to add an optimality cut. First, grab the duals
                dual_array = np.fromiter((con.Pi for con in constraints),
                                         dtype=float)
                # Here are the cut coefficients
                duals = Subproblem_Duals(
                    flow_duals=dual_array[:start_cap_cons],
                    capacity_duals=dual_array[start_cap_cons:])
            else:
                print "New incumbent found"
                duals = None
        elif status == GRB.status.INFEASIBLE:
            # Here we have to add a feasibility cut
            dual_array = np.fromiter((con.FarkasDual for con in constraints),
                                     dtype=float)
            # Here are the cut coefficients
            end_cap_cons = start_cap_cons + data.arcs.size * data.periods
            duals = Subproblem_Duals(
                flow_duals=dual_array[:start_cap_cons],
                capacity_duals=dual_array[start_cap_cons:end_cap_cons],
                bounds_duals=dual_array[end_cap_cons:])

        return status, duals

    def solve_dual_subproblem(open_arcs, flow_cost=0):
        """
        Solves the dual Benders subproblem. Comments to follow...
        :param flow_cost:   Continuous variable of Benders master problem
        :param open_arcs:   Arcs that are open at the master incumbent
        :return:            gurobi status message, Subproblem_Duals object
        """
        arcs, periods, nodes, commodities = xrange(data.arcs.size), xrange(
            data.periods), xrange(data.nodes), xrange(data.commodities)

        all_variables = subproblem.getVars()
        optimality_var = all_variables[-1]
        all_variables = all_variables[:-1]
        capacity_index = subproblem._capacity_index
        flow_index = subproblem._flow_index
        ubound_index = subproblem._ubounds_index
        capacity_duals = np.array([
            all_variables[capacity_index[period, arc]] for period, arc in
            product(periods, arcs)]).reshape(data.periods, data.arcs.size)
        flow_duals = np.array([
            all_variables[flow_index[node, commodity, period]] for
            node, commodity, period in
            product(nodes, commodities, periods)]).reshape(
            data.nodes, data.commodities, data.periods)
        ubound_duals = np.array([
            all_variables[ubound_index[arc, commodity, period]] for
            arc, commodity, period in
            product(arcs, commodities, periods)]).reshape(
            data.arcs.size, data.commodities, data.periods)

        for arc, period in product(arcs, periods):
            var = capacity_duals[period, arc]
            cap = data.capacity[arc]
            coeff = -cap * np.sum(open_arcs[:period + 1, arc])
            var.setAttr('Obj', coeff)

        optimality_var.setAttr('Obj', -flow_cost)

        subproblem.update()
        subproblem.optimize()
        status = subproblem.status

        if status == GRB.status.OPTIMAL:
            subproblem_objective = subproblem.objVal
            # We need to add a cut. First, grab the duals
            capacity_duals_vals = np.array([
                capacity_duals[period, arc].X
                for period, arc in product(periods, arcs)])
            flow_duals_vals = np.array([
                flow_duals[node, commodity, period].X
                for node, commodity, period in
                product(nodes, commodities, periods)])
            ubound_duals_vals = np.array([
                ubound_duals[arc, commodity, period].X
                for arc, commodity, period in
                product(arcs, commodities, periods)])

            # Here are the cut coefficients
            duals = Subproblem_Duals(
                flow_duals=flow_duals_vals,
                capacity_duals=capacity_duals_vals,
                bounds_duals=ubound_duals_vals,
                optimality_dual=optimality_var.X)
        else:
            raise RuntimeWarning('Ssomething went wrong..')

        return status, duals

    def master_callback(model, where):
        if where == GRB.callback.MIPSOL:
            node_count = int(model.cbGet(GRB.callback.MIPSOL_NODCNT))
            objective = model.cbGet(GRB.callback.MIPSOL_OBJ)
            master_variables = model._variables
            variables = model.cbGetSolution(model._variables)
            flow_cost = variables[-1]
            variables = np.array(variables[:-1]).reshape(
                data.periods, data.arcs.size)
            subproblem_status, duals = solve_dual_subproblem(
                flow_cost=flow_cost, open_arcs=variables)
            if subproblem_status == GRB.status.OPTIMAL:
                if duals is not None:
                    if duals.optimality_dual > 10e-7:
                        print 'Node {}, optimality cut, Objective: {}'.format(
                            node_count, objective)
                    else:
                        print 'Node {}, feasibility cut, Objective: {}'.format(
                            node_count, objective)
                    lhs = populate_benders_cut(duals, master_variables,
                                               data)
                else:
                    print 'model terminated'
                    return
                model.cbLazy(lhs=lhs, rhs=0., sense=GRB.LESS_EQUAL)
            else:
                print 'Error Gurobi status - subproblem not optimal'
                raise RuntimeWarning('Subproblem returned unknown status')

    return master_callback


def populate_benders_cut(duals, variables, data):
    """
    Returns the lhs and rhs parts of a benders cut. It does not determine if
    the cut is an optimality or a feasibility one (their coefficients are the
    same regardless)

    :param duals:       model dual values (structure Subproblem_Duals)
    :param variables:   gurobi model variables
    :param data:        problem data
    :return:            rhs (double), lhs (Gurobi linear expression)
    """
    nodes, commodities, periods, arcs = data.nodes, data.commodities, \
    data.periods, data.arcs.size
    flow_duals = duals.flow_duals.reshape(nodes, commodities, periods)
    ubound_duals = duals.bounds_duals.reshape(arcs, commodities, periods)
    capacity_duals = duals.capacity_duals.reshape(data.periods, data.arcs.size)
    optimality_dual = duals.optimality_dual
    origins, destinations = data.origins, data.destinations
    arcs, periods = xrange(data.arcs.size), xrange(data.periods)
    continuous_variable = variables[len(variables) - 1]

    lhs = LinExpr()
    for arc, period in product(arcs, periods):
        y_coeff = - data.capacity[arc] * np.sum(capacity_duals[period:, arc])
        if abs(y_coeff) > 10e-6:
            lhs.addTerms(y_coeff, variables[period * data.arcs.size + arc])

    lhs += np.sum([flow_duals[i] for i in zip(origins, xrange(commodities))]) - \
           np.sum([flow_duals[i] for i in zip(
               destinations, xrange(commodities))]) - ubound_duals.sum()

    lhs -= optimality_dual * continuous_variable

    return lhs


if __name__ == '__main__':
    main()