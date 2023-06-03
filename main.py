"""
Factor graph for rain world
"""

from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

# max t
t = 5

# nodes for variables
nodes = []
nodes.extend(['Rain_' + str(i + 1) for i in range(-1, t)])
nodes.extend(['Umbrella_' + str(i + 1) for i in range(t)])

# factor for initial distribution
factor_r_0 = DiscreteFactor(['Rain_0'], [2], [0.6, 0.4])
# transition factors
factors_p = [DiscreteFactor(['Rain_' + str(i), 'Rain_' + str(i + 1)],
                            [2, 2],
                            [[0.7, 0.3],
                            [0.3, 0.7]]) for i in range(t)]
# sensor factors
factors_s = [DiscreteFactor(['Rain_' + str(i + 1), 'Umbrella_' + str(i + 1)],
                            [2, 2],
                            [[0.9, 0.1],
                            [0.2, 0.8]]) for i in range(t)]

# edges between variables and factors
edges = [('Rain_0', factor_r_0)]
edges.extend([('Rain_' + str(i + 1), factors_s[i]) for i in range(t)])
edges.extend([('Umbrella_' + str(i + 1), factors_s[i]) for i in range(t)])
edges.extend([('Rain_' + str(i), factors_p[i]) for i in range(t - 1)])
edges.extend([('Rain_' + str(i + 1), factors_p[i]) for i in range(t - 1)])

G = FactorGraph()
G.add_nodes_from(nodes)
G.add_node(factor_r_0)
G.add_nodes_from(factors_p)
G.add_nodes_from(factors_s)
G.add_factors(factor_r_0)
G.add_factors(*factors_p)
G.add_factors(*factors_s)
G.add_edges_from(edges)

print('Check model :', G.check_model())

bp = BeliefPropagation(G)
bp.calibrate()

for i in range(1, 6):
    # In pgmpy true is 0 and false is 1
    result = bp.map_query(variables=[f'Rain_{i}'], evidence={'Umbrella_1': 0, 'Umbrella_2': 0, 'Umbrella_3': 1, 'Umbrella_4': 0, 'Umbrella_5': 0})
    print(result)
