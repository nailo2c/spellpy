import os
import pickle
from graphviz import Digraph

def helper(node, layer):
    if node.childD == {}:
        return

    if node.token != '':
        parent = node.token
        for child in node.childD:
            g.edge(parent+f' ({layer-1})', child+f' ({layer})')
    else:
        with g.subgraph() as s:
            s.attr(rank='same')
            for child in node.childD:
                s.node(child+f' ({layer})')

    for child in node.childD:
        helper(node.childD[child], layer+1)

if __name__ == '__main__':
    if not os.path.exists('./result/rootNode.pkl'):
        print('[ERROR] Please run example.py first to output rootNode.pkl')
        os._exit(0)

    with open('./result/rootNode.pkl', 'rb') as f:
        node = pickle.load(f)

    g = Digraph('G', filename='./plot/tree.gv', format='png')
    helper(node, 1)
    g.view()
