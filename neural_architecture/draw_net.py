from importlib import reload

# External files
import neural_net
import utils
import colors
import DenseLayer
import PlaneLayer
import legend
import draw_net

reload(neural_net)
reload(utils)
reload(colors)
reload(DenseLayer)
reload(PlaneLayer)
reload(legend)
reload(draw_net)

from neural_net     import *
from colors         import *
from utils          import *
from DenseLayer     import *
from PlaneLayer     import *
from legend         import *
from draw_net         import *

def draw_net(model, input_size = (1, 32, 32), figure_name = None):
    model_dict = torch_model_dict(model, input_size)
    nn = neural_net.NN()
    for layer, layer_dict in model_dict.items():
        layer_type = layer.split('-')[0]
        if layer_type == 'Conv2d':
            nn.add(Conv2D())
        elif layer_type == 'Linear':
            # print(layer_dict)
            num_un = int(5 + 10/(1 + np.exp(-0.1*layer_dict['output_shape'][1])))
            nn.add(DenseLayer(num_units = num_un))
        elif layer_type == 'MaxPool2d':
            pool_size = int(round(layer_dict['input_shape'][-1]/layer_dict['output_shape'][-1]))
            nn.add(Pooling(pool_size=pool_size))
        elif layer_type == 'BatchNorm2d':
            nn.add(BatchNorm())
    
    # print(nn.layers)
    nn.compile()
    nn.add_legend(bbox_to_anchor=(0.2, -0.1, 0.6, 0.1), fontsize=20)
    if figure_name is not None:
        nn.draw(save=True, save_path=figure_name)
    else:
        nn.draw()

