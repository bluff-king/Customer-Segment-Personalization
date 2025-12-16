import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import os

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    """
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    root: the root node of current branch 
    width: horizontal space allocated for this branch - avoids overlap with other branches
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        # Fallback if component is not strictly a tree (e.g. has cycles or undirected), 
        # though we expect trees here.
        # As a fallback for "almost trees", we can try a DFS tree or just spring.
        # But for this specific task, we'll try to treat it as tree.
        pass

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def visualize_category_tree(csv_path, output_path='category_tree_graph.html'):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return

    # Check columns
    if 'categoryid' not in df.columns or 'parentid' not in df.columns:
        print("Error: CSV must contain 'categoryid' and 'parentid' columns.")
        return

    # Build full graph first
    G_full = nx.DiGraph()
    for _, row in df.iterrows():
        child = row['categoryid']
        parent = row['parentid']
        if pd.notna(parent):
            try:
                G_full.add_edge(int(parent), int(child))
            except ValueError:
                G_full.add_edge(parent, child)
        else:
            # Add root node explicitly if it has no parent edge
             if pd.notna(child):
                 G_full.add_node(int(child) if isinstance(child, (int, float)) else child)

    # Find weakly connected components (trees/forests)
    # We use undirected version to find components
    undirected_G = G_full.to_undirected()
    components = list(nx.weakly_connected_components(G_full))
    
    # Filter top 5 largest components
    components.sort(key=len, reverse=True)
    top_components = components[:5]
    print(f"Selecting top {len(top_components)} trees from {len(components)} total components.")

    # Create figure
    fig = go.Figure()

    # Layout parameters
    global_x_offset = 0
    tree_spacing = 2.0  # Spacing between trees

    for i, nodes in enumerate(top_components):
        subG = G_full.subgraph(nodes).copy()
        
        # Find root(s): nodes with in-degree 0
        roots = [n for n, d in subG.in_degree() if d == 0]
        if not roots:
            # If cycle or strange structure, pick one with max out degree
            print(f"Warning: Component {i} has no clear root (cycle?). Picking arbitrary root.")
            roots = [sorted(nodes)[0]]
        
        # Taking the first root found (if multiple roots in one component, it's actually separate trees 
        # but weakly connected usually implies one structure if we treated it right. 
        # If a component has 2 roots, it might be W-shape. 
        # For simplicity, we iterate all roots in this component (effectively forest within component?)
        # actually weakly connected component with multiple roots is just multiple trees converging possibly?
        # Let's just visualize from the main root.
        
        # Better: run layout for the subgraph
        # We need to ensure it's a tree for the layout function.
        # If not, convert to DFS tree to break cycles/multi-parents for visualization
        try:
            if not nx.is_tree(subG):
                 # subgraph might not be a tree (e.g. multiple parents for a node? or cycles?)
                 # For visualization, we treat it as a tree from the root.
                 pass
        except:
             pass

        root = roots[0] # Primary root
        
        # Compute positions
        # width correlates to number of leaves or width of tree
        # Let's count leaves to guess width
        # leaves = [x for x in subG.nodes() if subG.out_degree(x)==0 and subG.in_degree(x)==1]
        # tree_width = max(len(leaves), 1) * 1.0
        
        pos = hierarchy_pos(subG, root=root, width=max(len(subG.nodes())/2, 5), vert_gap=1.0)
        
        # Shift positions by global offset
        # And normalize distinct tree x-range
        xs = [p[0] for p in pos.values()]
        min_x = min(xs)
        shift = global_x_offset - min_x
        
        final_pos = {n: (x + shift, y) for n, (x, y) in pos.items()}
        
        # Update global offset for next tree
        max_x = max([x for x, y in final_pos.values()])
        global_x_offset = max_x + tree_spacing

        # Draw Edges (Orthogonal)
        edge_x = []
        edge_y = []
        
        for edge in subG.edges():
            start_node = edge[0]
            end_node = edge[1]
            if start_node in final_pos and end_node in final_pos:
                x0, y0 = final_pos[start_node]
                x1, y1 = final_pos[end_node]
                
                # Orthogonal path: (x0, y0) -> (x0, mid) -> (x1, mid) -> (x1, y1)
                # mid y is halfway between y0 and y1
                ymid = (y0 + y1) / 2
                
                edge_x.extend([x0, x0, x1, x1, None])
                edge_y.extend([y0, ymid, ymid, y1, None])

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888', shape='hv'), # shape='hv' hints plotly but we drew manual segments
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Draw Nodes
        node_x = []
        node_y = []
        node_text = []
        
        for node in subG.nodes():
            if node in final_pos:
                x, y = final_pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="bottom center",
            hoverinfo='text',
            marker=dict(
                symbol='square',
                size=20,
                color='lightblue',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text='Refined Category Tree Visualization (Top 5 Trees)',
            font=dict(size=20)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=20,r=20,t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        plot_bgcolor='white'
    )

    print(f"Saving visualization to {output_path}...")
    fig.write_html(output_path)
    print("Done!")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'category_tree.csv')
    OUTPUT_FILE = os.path.join(os.path.join(BASE_DIR, 'EDA'), 'category_tree_graph.html')
    
    visualize_category_tree(DATASET_PATH, OUTPUT_FILE)
