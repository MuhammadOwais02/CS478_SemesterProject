import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import time

class DisjointSet:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, vertex1, vertex2):
        root1 = self.find(vertex1)
        root2 = self.find(vertex2)

        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                self.rank[root2] += 1

def kruskal(graph, vertices):
    disjoint_set = DisjointSet(vertices)
    edges = []

    # Transform the graph into a list of edges
    for vertex, neighbors in graph.items():
        for neighbor, weight in neighbors:
            edges.append((weight, vertex, neighbor))

    # Sort edges in ascending order by weight
    edges.sort()

    minimum_spanning_tree = []

    for edge in edges:
        weight, vertex1, vertex2 = edge

        if disjoint_set.find(vertex1) != disjoint_set.find(vertex2):
            disjoint_set.union(vertex1, vertex2)
            minimum_spanning_tree.append((vertex1, vertex2, {'weight': weight}))
    del edges,disjoint_set
    return minimum_spanning_tree

def generate_random_graph(n, density):
    graph = {}
    vertices = [str(i) for i in range(n)]

    # Create edges based on density
    for i in range(n):
        neighbors = random.sample(vertices[:i] + vertices[i+1:], int(density * n))
        weights = [random.randint(1, 100) for _ in range(len(neighbors))]
        graph[vertices[i]] = list(zip(neighbors, weights))
    del vertices
    return graph

def save_graph_plot(graph, minimum_spanning_tree, num_nodes, density, directory):
    H = nx.Graph()
    H.add_edges_from(minimum_spanning_tree)
    pos_h = nx.spring_layout(H)
    plt.figure(figsize=(10, 8))
    nx.draw(H, pos_h, with_labels=True, font_weight='bold', node_size=70, node_color='lightcoral', font_size=3)
    #nx.draw_networkx_edge_labels(H, pos_h,edge_labels={(e[0], e[1]): e[2]['weight'] for e in H.edges(data=True)})
    plt.title(f"Minimum Spanning Tree (Nodes: {num_nodes}, Density: {density})")
    plt.savefig(os.path.join(directory, str(num_nodes)+"_"+str(density)+"mst.png"))
    plt.close()
    del H

def measure_execution_time_kruskal(graph, vertices):
    start_time = time.time()
    mst=kruskal(graph, vertices)
    end_time = time.time()
    return mst,end_time - start_time

# Test cases: n values and density values
n_values = [100, 1000, 5000, 10000]
density_values = [0.1,0.6]

NN=[]
ddensity=[]
EXEC_TIME=[]

for n in n_values:
    for density in density_values:
        graph = generate_random_graph(n, density)
        vertices = list(graph.keys())
        mst,execution_time = measure_execution_time_kruskal(graph, vertices)
        #-------
        save_graph_plot(graph, mst, n, density, "./plots")
        #------
        NN.append(n)
        ddensity.append(density)
        EXEC_TIME.append(execution_time)
        print(f"Graph with n={n} vertices and density={density}: {execution_time:.10f} seconds")
        del graph,mst,vertices
results={ "Number of Nodes":NN,"Density":ddensity,"Execution Time (sec)":EXEC_TIME}
df = pd.DataFrame(results)
df.to_csv('kruskal_results.csv', index=False)