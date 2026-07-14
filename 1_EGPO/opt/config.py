import os
import yaml
# This is the key change: import the 'args' object, not a function.
from opt.parser import args

def init_config():
    config = dict()

    # Using os.getcwd() is more reliable in Colab
    project_root = os.getcwd() 
    basic_init_file = os.path.join(project_root, 'assets/overall.yaml')

    basic_conf = yaml.load(open(basic_init_file), Loader=yaml.loader.SafeLoader)
    config.update(basic_conf)

    # Use the 'model' setting from the imported args object
    model_file = args.model
    model_init_file = os.path.join(project_root, f'assets/{model_file}.yaml')
    model_conf = yaml.load(open(model_init_file), Loader=yaml.loader.SafeLoader)
    config.update(model_conf)

    args_conf = vars(args)

    for k, v in config.items():
        if k in args_conf.keys() and args_conf[k] is not None:
            config[k] = args_conf[k]
        else:
            config[k] = v
    return config