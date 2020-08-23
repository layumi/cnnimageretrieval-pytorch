import os
import torch
import yaml
import torch.nn as nn
import parser
from cirtorch.networks.university1652.model import ft_net, two_view_net, three_view_net

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1 # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s'%dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
#---------------------------
def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./model/'+dirname):
        os.mkdir('./model/'+dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth'% epoch_label
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
#---------------------------
def load_network(name):
    # Load config
    dirname = name
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if not epoch=='last':
       epoch = int(epoch)
    config_path = os.path.join(dirname,'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    model = three_view_net( config['nclasses'], droprate=config['droprate'], stride = config['stride'], pool = config['pool'], share_weight = config['share'])


    if isinstance(epoch, int):
        save_path = name + '/net_%03d.pth'% epoch
    else:
        save_path = name + '/net_%s.pth'% epoch
    print('Load the model from %s'%save_path)
    network = model
    #network.load_state_dict(torch.load(save_path))
    return network, None, epoch

def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toogle_grad(model_src, True)

