# From https://gist.github.com/apaszke/01aae7a0494c55af6242f06fad1f8b70
from graphviz import Digraph
import torch
from torch.autograd import Variable
import sys
sys.path.append('../deeplab_3D/')
sys.path.append('../unet_3D/')
sys.path.append('../vnet_3D/')
sys.path.append('../hrnet_3D/')
sys.path.append('../new_nets_3D/')

import deeplab_resnet_3D
import unet_3D
import vnet_3D
import highresnet_3D
import highresnet_x_deeplab_3D

def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph
    
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)
    
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    
    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

inputs = torch.randn(1,1,80,80,80)
#model = deeplab_resnet_3D.Res_Deeplab(3)
#model = unet_3D.UNet3D(1, 3)
model = highresnet_x_deeplab_3D.getNewNet(2)
#model = highresnet_3D.HighResNet(3)
y = model(Variable(inputs))

g = make_dot(y,  model.state_dict())
g.view()