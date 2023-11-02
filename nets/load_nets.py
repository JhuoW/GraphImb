from nets.MLP_net import MLPNet

def get_model(MODEL_NAME, net_params):
    models_class = {'mlp': MLPNet}

    return models_class[MODEL_NAME](net_params)