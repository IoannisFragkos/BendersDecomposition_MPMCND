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
LOG_LEVEL = 0


def main():
    filename = 'R_single_period/r01.1_R_H_10.dow' if len(argv) <= 1 else argv[1]
    data = read_data(filename)
    start = time()
    data.graph = make_graph(data)
    master = populate_master(data, None)
    subproblems = populate_dual_subproblem(data, None)
    master_callback = callback_data(subproblems, data)
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
    continous_variables = np.empty(shape=data.periods, dtype=object)

    # Add variables
    for period, arc in product(periods, arcs):
        variables[period, arc] = master.addVar(vtype=GRB.BINARY,
                                               obj=data.fixed_cost[period, arc],
                                               name='arc_open{}_{}'.format(
                                                   period, arc))
    # Continuous flow_cost variables
    for period in periods:
        continous_variables[period] = master.addVar(
            lb=0., obj=1., vtype=GRB.CONTINUOUS, name='flow_cost{}'.format(
                period))
    master.update()

    # Add constraints
    for arc in arcs:
        lhs = LinExpr()
        for period in periods:
            lhs.addTerms(1., variables[period, arc])
        master.addConstr(lhs=lhs, rhs=1., sense=GRB.LESS_EQUAL,
                         name='arc{}'.format(arc))

        # Add Origin - Destination Cuts for each Commodity
        for commodity in xrange(data.commodities):
            arc_origin = data.origins[commodity]
            arc_destination = data.destinations[commodity]
            out_origin = get_2d_index(data.arcs, data.nodes)[
                             0] - 1 == arc_origin
            in_destination = get_2d_index(
                data.arcs, data.nodes)[1] - 1 == arc_destination
            master.addConstr(
                lhs=np.sum(variables[0, in_destination]), rhs=1.,
                sense=GRB.GREATER_EQUAL,
                name='destinations_c{}'.format(commodity))
            master.addConstr(
                lhs=np.sum(variables[0, out_origin]), rhs=1.,
                sense=GRB.GREATER_EQUAL, name='origins_c{}'.format(commodity))

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
    # Find feasible solutions quickly, works better
    master.params.MIPFocus = 1
    master.update()
    # Store the variables inside the model, we cannot access them later!
    master._variables = master.getVars()
    return master


def populate_dual_subproblem(data, open_arcs, flow_cost=None):
    """
    Function that populates the Benders Dual Subproblem, as suggested by the
    paper "Minimal Infeasible Subsystems and Bender's cuts" by Fischetti,
    Salvagnin and Zanette.
    :param open_arcs:   Array that is used to initialize the subproblem rhs.
                        Typically, we should get it from a heuristic solution.
                        We set it to a zero array if it is not initialized
    :param flow_cost:   This is the cost of the continuous variables of the
                        master problem, as explained in the paper
    :return:            Numpy array of Gurobi model objects
    """

    # Stores Gurobi model objects
    subproblems = np.empty(shape=data.periods, dtype=object)

    # Construct model for period 0. Then, copy this and change the coefficients
    dual_subproblem = Model('dual_subproblem_0')

    # Ranges we are going to need
    arcs, periods, commodities, nodes = xrange(data.arcs.size), xrange(
        data.periods), xrange(data.commodities), xrange(data.nodes)

    # We use arrays to store variable indexes and variable objects. Why use
    # both? Gurobi wont let us get the values of individual variables
    # within a callback.. We just get the values of a large array of
    # variables, in the order they were initially defined. To separate them
    # in variable categories, we will have to use index arrays
    flow_index = np.zeros(shape=(data.nodes, data.commodities), dtype=int)
    flow_duals = np.empty_like(flow_index, dtype=object)
    capacity_index = np.zeros(shape=data.arcs.size, dtype=int)
    capacity_duals = np.empty_like(capacity_index, dtype=object)
    ubounds_index = np.zeros(shape=(len(arcs), data.commodities), dtype=int)
    ubounds_duals = np.empty_like(ubounds_index, dtype=object)

    # Makes sure we don't add variables more than once
    flow_duals_names = set()

    if open_arcs is None:
        open_arcs = np.zeros(shape=(len(periods), len(arcs)), dtype=float)
    if flow_cost is None:
        flow_cost = np.zeros(shape=len(periods), dtype=float)

    # Populate all variables in one loop, keep track of their indexes
    count = 0
    for arc in arcs:
        obj = - open_arcs[0, arc] * data.capacity[arc]
        capacity_duals[arc] = dual_subproblem.addVar(
            obj=obj, name='capacity_dual_a{}'.format(arc))
        capacity_index[arc] = count
        count += 1
        for commodity in commodities:
            start_node, end_node = get_2d_index(data.arcs[arc], data.nodes)
            start_node, end_node = start_node - 1, end_node - 1
            for node in (start_node, end_node):
                var_name = 'flow_dual_n{}c{}'.format(node, commodity)
                if var_name not in flow_duals_names:
                    flow_duals_names.add(var_name)
                    obj = 0.
                    if data.origins[commodity] == node:
                        obj = 1.
                    if data.destinations[commodity] == node:
                        obj = -1.
                    flow_duals[node, commodity] = \
                        dual_subproblem.addVar(
                            obj=obj, lb=-GRB.INFINITY, name=var_name)
                    flow_index[node, commodity] = count
                    count += 1
            ubounds_duals[arc, commodity] = dual_subproblem.addVar(
                obj=-1., name='u_bound_dual_a{}c{}'.format(arc, commodity))
            ubounds_index[arc, commodity] = count
            count += 1
    opt_var = dual_subproblem.addVar(obj=-flow_cost[0], name='optimality_var')
    dual_subproblem.update()

    for arc, commodity in product(arcs, commodities):
        start_node, end_node = get_2d_index(data.arcs[arc], data.nodes)
        start_node, end_node = start_node - 1, end_node - 1
        demand = data.demand[0, commodity]
        lhs = flow_duals[start_node, commodity] \
              - flow_duals[end_node, commodity] \
              - capacity_duals[arc] * demand - \
              ubounds_duals[arc, commodity] - \
              opt_var * data.variable_cost[arc] * demand
        dual_subproblem.addConstr(
            lhs <= 0., name='flow_a{}c{}'.format(arc, commodity))

    # Original Fischetti model
    lhs = np.sum(capacity_duals) + opt_var
    dual_subproblem.addConstr(lhs == 1, name='normalization_constraint')

    dual_subproblem._capacity_index = capacity_index
    dual_subproblem._flow_index = flow_index
    dual_subproblem._ubounds_index = ubounds_index

    dual_subproblem.setParam('OutputFlag', 0)
    # Switch on the additional parameters that calculate dual values when
    # then dual problem is unbounded
    dual_subproblem.setParam('PreSolve', 0)
    dual_subproblem.setParam('InfUnbdInfo', 1)
    dual_subproblem.modelSense = GRB.MAXIMIZE
    dual_subproblem.update()

    subproblems[0] = dual_subproblem

    if data.periods > 1:
        for period in list(periods)[1:]:
            model = dual_subproblem.copy()
            optimality_var = model.getVarByName('optimality_var')
            optimality_var.Obj = -flow_cost[period]
            for arc in arcs:
                variable = model.getVarByName('capacity_dual_a{}'.format(arc))
                variable.Obj = - data.capacity[arc] * np.sum(open_arcs[
                                                             :period + 1, arc])
                for commodity in commodities:
                    demand = data.demand[period, commodity]
                    constraint = model.getConstrByName(
                        'flow_a{}c{}'.format(arc, commodity))
                    model.chgCoeff(constraint, optimality_var,
                                   - demand * data.variable_cost[arc])
                    model.chgCoeff(constraint, variable, -demand)
            model.update()
            subproblems[period] = model
    return subproblems


def callback_data(subproblems, data):
    """
    This is a closure that passes whatever data we want to the actual
    callback function. We have to use this because gurobi callbacks have a
    certain signature (model, where)
    :param subproblem: Gurobi subproblem models (one per period)
    :param data:       Problem data
    :return:           master_callback function

    """

    def solve_dual_subproblem(open_arcs, flow_cost=None):
        """
        Solves the dual Benders subproblems.
        :param flow_cost:   Continuous variables of Benders master problem
        :param open_arcs:   Arcs that are open at the master incumbent
        :return:            gurobi status message, Subproblem_Duals object
        """
        arcs, periods, nodes, commodities = xrange(data.arcs.size), xrange(
            data.periods), xrange(data.nodes), xrange(data.commodities)

        # Indices of subproblem variables
        capacity_index = subproblems[0]._capacity_index
        flow_index = subproblems[0]._flow_index
        ubound_index = subproblems[0]._ubounds_index

        if flow_cost is None:
            flow_cost = np.zeros(shape=data.periods, dtype=float)
        # We need to initialize this array here, because it is cumulative
        # capacity_duals_vals = np.zeros(shape=data.arcs.size, dtype=float)

        # Return arrays
        status_arr = np.zeros(shape=data.periods, dtype=int)
        duals_arr = np.empty(shape=data.periods, dtype=object)

        # We loop backwards because for capacity duals we need to store their
        #  sum from each period to the last period
        for period in reversed(periods):
            subproblem = subproblems[period]
            all_variables = subproblem.getVars()
            optimality_var = all_variables[-1]
            all_variables = all_variables[:-1]
            capacity_duals = np.take(all_variables, capacity_index)
            flow_duals = np.take(all_variables, flow_index)
            ubound_duals = np.take(all_variables, ubound_index)

            for arc in arcs:
                var = capacity_duals[arc]
                cap = data.capacity[arc]
                coeff = -cap * np.sum(open_arcs[:period+1, arc])
                var.setAttr('Obj', coeff)

            optimality_var.setAttr('Obj', -flow_cost[period])

            subproblem.update()
            subproblem.optimize()
            status_arr[period] = subproblem.status

            if status_arr[period] == GRB.status.OPTIMAL:
                # We need to add a cut. First, grab the duals
                capacity_duals_vals = np.array([
                    capacity_duals[arc].X for arc in arcs])
                flow_duals_vals = np.array([
                    flow_duals[node, commodity].X for node, commodity in
                    product(nodes, commodities)])
                ubound_duals_vals = np.array([
                    ubound_duals[arc, commodity].X
                    for arc, commodity in product(arcs, commodities)])

                # Here are the cut coefficients
                duals = Subproblem_Duals(
                    flow_duals=flow_duals_vals,
                    capacity_duals=capacity_duals_vals.copy(),
                    bounds_duals=ubound_duals_vals,
                    optimality_dual=optimality_var.X)
                duals_arr[period] = duals
            else:
                raise RuntimeWarning('Something went wrong..')

        return status_arr, duals_arr

    def master_callback(model, where):
        if where == GRB.callback.MIPSOL:
            node_count = int(model.cbGet(GRB.callback.MIPSOL_NODCNT))
            master_variables = model._variables
            variables = model.cbGetSolution(model._variables)
            flow_cost = variables[-data.periods:]
            variables = np.array(variables[:-data.periods]).reshape(
                data.periods, data.arcs.size)
            subproblem_status_arr, duals_arr = solve_dual_subproblem(
                flow_cost=flow_cost, open_arcs=variables)
            for period in xrange(data.periods):
                subproblem_status = subproblem_status_arr[period]
                duals = duals_arr[period]
                if subproblem_status == GRB.status.OPTIMAL:
                    if LOG_LEVEL:
                        if duals.optimality_dual > 10e-7:
                            print 'Node {}, optimality cut, Period: {}'.format(
                                node_count, period+1)
                        else:
                            print 'Node {}, feasibility cut, Period: {}'.format(
                                node_count, period+1)
                    lhs = populate_benders_cut(duals, master_variables,
                                               period, data)
                    model.cbLazy(lhs=lhs, rhs=0., sense=GRB.LESS_EQUAL)
                else:
                    raise RuntimeWarning('Subproblem unknown status')
        elif where == GRB.callback.MIPNODE:
            node_count = int(model.cbGet(GRB.callback.MIPNODE_NODCNT))
            if (node_count % 1000 == 0 or node_count < 10) and model.cbGet(
                    GRB.callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                master_variables = model._variables
                variables = model.cbGetNodeRel(model._variables)
                flow_cost = variables[-data.periods:]
                variables = np.array(variables[:-data.periods]).reshape(
                    data.periods, data.arcs.size)
                subproblem_status_arr, duals_arr = solve_dual_subproblem(
                    flow_cost=flow_cost, open_arcs=variables)
                for period in xrange(data.periods):
                    subproblem_status = subproblem_status_arr[period]
                    duals = duals_arr[period]
                    if subproblem_status == GRB.status.OPTIMAL:
                        if LOG_LEVEL:
                            if duals.optimality_dual > 10e-7:
                                print 'optimality cut, Period: {}'.format(
                                    period+1)
                            else:
                                print 'feasibility cut, Period: {}'.format(
                                    period+1)
                        lhs = populate_benders_cut(duals, master_variables,
                                                   period, data)
                        if node_count < 10:
                            model.cbCut(lhs=lhs, rhs=0., sense=GRB.LESS_EQUAL)
                        else:
                            model.cbLazy(lhs=lhs, rhs=0., sense=GRB.LESS_EQUAL)
                    else:
                        raise RuntimeWarning('Subproblem unknown status')
    return master_callback


def populate_benders_cut(duals, variables, period, data):
    """
    Returns the lhs and rhs parts of a benders cut. It does not determine if
    the cut is an optimality or a feasibility one (their coefficients are the
    same regardless)

    :param duals:       model dual values (structure Subproblem_Duals)
    :param variables:   gurobi model variables
    :param period:      period in which we add the cut
    :param data:        problem data
    :return:            rhs (double), lhs (Gurobi linear expression)
    """
    nodes, commodities, periods, arcs = data.nodes, data.commodities, \
                                        data.periods, data.arcs.size
    flow_duals = duals.flow_duals.reshape(nodes, commodities)
    ubound_duals = duals.bounds_duals.reshape(arcs, commodities)
    capacity_duals = duals.capacity_duals
    optimality_dual = duals.optimality_dual
    origins, destinations = data.origins, data.destinations
    arcs, periods = xrange(data.arcs.size), xrange(data.periods)
    continuous_variable = variables[period-data.periods]

    lhs = LinExpr()
    for arc in arcs:
        y_coeff = - data.capacity[arc] * capacity_duals[arc]
        for period2 in xrange(0, period+1):
            if abs(y_coeff) > 10e-6:
                lhs.addTerms(y_coeff, variables[period2 * data.arcs.size + arc])

    lhs += np.sum([flow_duals[i] for i in zip(origins, xrange(commodities))]) - \
           np.sum([flow_duals[i] for i in zip(
               destinations, xrange(commodities))]) - ubound_duals.sum()

    lhs -= optimality_dual * continuous_variable

    # print lhs

    return lhs


if __name__ == '__main__':
    main()