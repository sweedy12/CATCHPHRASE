import plotly.graph_objs as go
import plotly
import igraph as ig


def create_edges(root, num_dict):
    """
    this function creates the edges list of the tree who's root is given by root
    :param root: the root of the tree
    :param num_dict: a dictionary, mapping each node to its serial number.
    :return:
    """
    nodes  = []
    edges = []
    group = []
    labels = []
    nodes.append(root)
    counter = 0
    #bfs iteration on all nodes in order to create edges, group, labels.
    while (nodes):
        node = nodes.pop()
        group.append(node.get_depth())
        label = node.get_full_sent()
        labels.append(label)
        children = node.get_children()
        counter+= len(children)
        new_edges = [(num_dict[node.get_id()],num_dict[child.get_id()]) for child in
                     children]
        edges+=new_edges
        nodes.extend(children)
    return edges, labels,group



def get_num_dict(root):
    """
    this function creates a mapping between each node and its serial number
    :param root: the root of the tree to traverse.
    :return:
    """
    num_dict = {}
    nodes = []
    nodes.append(root)
    i = 0
    #bfs iteration on the tree.
    while(nodes):
        node = nodes.pop()
        num_dict[node.get_id()] = i
        i+=1
        nodes.extend(node.get_children())
    return num_dict

def plot_tree(root, name):
    """
    this function creates a plot (shown in a plotly account online) if the tree given in root.
    :param root: the root of the tree to plot.
    :param name: the name of the table this tree is defined by.
    :return:
    """
    num_dict = get_num_dict(root)
    Edges,labels,group = create_edges(root, num_dict)
    N = len(Edges) + 1
    G = ig.Graph(Edges, directed = True)
    layt=G.layout('kk', dim=3)
    print("N IS " +str(N))
    print("kength is " + str(len(layt)))
    if (N ==1):
        print("this is an empty graph")
        return
    Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[layt[k][1] for k in range(N)]# y-coordinates
    Zn=[layt[k][2] for k in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in Edges:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]
        Ze+=[layt[e[0]][2],layt[e[1]][2], None]

    trace1=go.Scatter3d(x=Xe,
                   y=Ye,
                   z=Ze,
                   mode='lines',
                   line=dict(color='rgb(125,125,125)', width=1),
                   hoverinfo='none'
                   )

    trace2=go.Scatter3d(x=Xn,
                   y=Yn,
                   z=Zn,
                   mode='markers',
                   name='actors',
                   marker=dict(symbol='circle',
                                 size=6,
                                 color=group,
                                 colorscale='Viridis',
                                 line=dict(color='rgb(50,50,50)', width=0.5)
                                 ),
                   text=labels,
                   hoverinfo='text'
                   )

    axis=dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )

    layout = go.Layout(
             title="Cluster tree for the sentnce " + name,
             width=1000,
             height=1000,
             showlegend=False,
             scene=dict(
                 xaxis=dict(axis),
                 yaxis=dict(axis),
                 zaxis=dict(axis),
            ),
     margin=dict(
        t=100
    ),
    hovermode='closest',
    annotations=[
           dict(
           showarrow=False,
            text="Data taken from reddit", #try to denote here the subreddit\url this is taken from
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=14
            )
            )
        ],    )

    data=[trace1, trace2]
    fig=go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename="trees/_" + name + ".html", auto_open=False)