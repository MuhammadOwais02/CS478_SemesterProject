import heapq
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import time

def prim(graph):
    start_vertex = list(graph.keys())[0]
    edge_heap = [(weight, start_vertex, neighbor) for neighbor, weight in graph[start_vertex]]
    visited = set(start_vertex)
    mst = []
    heapq.heapify(edge_heap)
    
    while edge_heap:
        weight, current_vertex, next_vertex = heapq.heappop(edge_heap)
        
        if next_vertex not in visited:
            visited.add(next_vertex)
            mst.append((current_vertex, next_vertex, weight))
            
            for neighbor, weight in graph[next_vertex]:
                if neighbor not in visited:
                    heapq.heappush(edge_heap, (weight, next_vertex, neighbor))
    del visited,edge_heap,start_vertex
    return mst

def save_graph_plot(minimum_spanning_tree, num_nodes, density, directory):
    H = nx.Graph() 
    edges = [(edge[0], edge[1]) for edge in minimum_spanning_tree]
    
    H.add_edges_from(edges)
    del edges
    pos_h = nx.spring_layout(H)
    plt.figure(figsize=(10, 8))
    nx.draw(H, pos_h, with_labels=True, font_weight='bold', node_size=70, node_color='lightcoral', font_size=3)
    plt.title(f"Minimum Spanning Tree (Nodes: {num_nodes}, Density: {density})")
    plt.savefig(os.path.join(directory, str(num_nodes) + "_" + str(density) + "mst.png"))
    plt.close()
    del H

def generate_random_graph(n, density):
    graph = {}
    vertices = [str(i) for i in range(n)]
    for i in range(n):
        neighbors = random.sample(vertices[:i] + vertices[i+1:], int(density * n))
        weights = [random.randint(1, 100) for _ in range(len(neighbors))]
        graph[vertices[i]] = list(zip(neighbors, weights))
    del vertices
    return graph

def measure_execution_time(graph):
    start_time = time.time()
    mst=prim(graph)
    end_time = time.time()
    return mst,end_time - start_time

# Test cases: n values and density values
n_values = [100, 1000, 5000, 7500, 10000]
density_values = [0.1,0.6]

NN=[]
ddensity=[]
EXEC_TIME=[]

for n in n_values:
    for density in density_values:
        graph = generate_random_graph(n, density)
        mst,execution_time = measure_execution_time(graph)

        #-------
        save_graph_plot(mst, n, density, "./plots/prims_result_visualisation")
        #------

        NN.append(n)
        ddensity.append(density)
        EXEC_TIME.append(execution_time)
        print(f"Graph with n={n} vertices and density={density}: {execution_time:.6f} seconds")

results={ "Number of Nodes":NN,"Density":ddensity,"Execution Time (sec)":EXEC_TIME}
df = pd.DataFrame(results)
df.to_csv('primresults.csv', index=False)