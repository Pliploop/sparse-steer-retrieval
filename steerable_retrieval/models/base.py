from torch import nn

class BaseModule(nn.Module):
    def __init__(self, ckpt_path = None, freeze = False):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.freeze = freeze


         
    @classmethod
    def from_config(cls, config, device=None):
        config.update(device=device)
        return cls(**config)
    
    @classmethod
    def from_yaml(cls, yaml_path, device=None):
        
        if 's3://' in yaml_path:
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
        else:
            with open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
        
        config = config.get('model', config)
        config = config.get('init_args', config)
        
        return cls.from_config(config, device=device)
    
    @classmethod
    def from_pretrained(cls, yaml_or_config, ckpt_path, device=None):
        if isinstance(yaml_or_config, str):
            model = cls.from_yaml(yaml_or_config, device=device)
        else:
            model = cls.from_config(yaml_or_config, device=device)
            
        if 's3://' in ckpt_path:
            from s3torchconnector import S3Checkpoint
            checkpoint= S3Checkpoint(region='us-east-1')
            with checkpoint.reader(ckpt_path) as f:
                ckpt = torch.load(f, map_location=device)
                model.load_state_dict(ckpt['state_dict'], strict=True)
                print(f"Model loaded from {ckpt_path}")
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['state_dict'], strict=True)
            print(f"Model loaded from {ckpt_path}")
        
        return model
    

    def configure_optimizers(self):
        from steerable_retrieval.utils.instantiators import instantiate
        
        if not hasattr(self, 'optimizer'):
            self.optimizer = None
        if self.optimizer is None:
            optimizer = optim.Adam(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        else:
            # If optimizer is a config dict or partial, instantiate it with model parameters
            if isinstance(self.optimizer, dict):
                # Handle _partial_ configs - Hydra creates a functools.partial
                optimizer_cfg = dict(self.optimizer)
                
                # If using _target_ style (with or without _partial_)
                if '_target_' in optimizer_cfg:
                    # Remove _partial_ flag if present (it was just to prevent instantiation)
                    optimizer_cfg.pop('_partial_', None)
                    optimizer_cfg.pop('_convert_', None)
                    optimizer_cfg['params'] = self.parameters()
                    optimizer = instantiate(optimizer_cfg)
                # If using class_path style
                elif 'class_path' in optimizer_cfg:
                    import importlib
                    module_path, class_name = optimizer_cfg['class_path'].rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    optimizer_class = getattr(module, class_name)
                    init_args = optimizer_cfg.get('init_args', {})
                    init_args['params'] = self.parameters()
                    optimizer = optimizer_class(**init_args)
                else:
                    # Fallback: assume it's a direct config
                    optimizer_cfg['params'] = self.parameters()
                    optimizer = instantiate(optimizer_cfg)
            elif hasattr(self.optimizer, 'func') and hasattr(self.optimizer, 'keywords'):
                # It's a functools.partial (from _partial_=true) - call it with params
                optimizer = self.optimizer(params=self.parameters())
            elif callable(self.optimizer):
                # If it's a callable (old style), call it with parameters
                optimizer = self.optimizer(self.parameters())
            else:
                # Fallback to default
                optimizer = optim.Adam(
                    self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
            
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            # copy of the scheduler applied to the optimizer
            ## retrocompatibilty with old schedulers
            if isinstance(self.scheduler, dict) and 'class_name' in self.scheduler.keys():
                scheduler_class = eval(self.scheduler['class_name'])
                scheduler_kwargs = self.scheduler.get('init_args', {})
                scheduler = scheduler_class(optimizer, **scheduler_kwargs)
                self.scheduler = scheduler  # Store instantiated scheduler
                # Return with proper Lightning configuration
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',  # Step after optimizer steps (respects accumulate_grad_batches)
                        'frequency': 1,      # Step every optimizer step
                    }
                }
            elif isinstance(self.scheduler, dict):
                # Handle configs that were prevented from instantiation
                scheduler_cfg = dict(self.scheduler)
                
                # If using class_path style (not _target_)
                if 'class_path' in scheduler_cfg:
                    import importlib
                    module_path, class_name = scheduler_cfg['class_path'].rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    scheduler_class = getattr(module, class_name)
                    init_args = scheduler_cfg.get('init_args', {})
                    init_args['optimizer'] = optimizer
                    scheduler = scheduler_class(**init_args)
                # If using _target_ style
                elif '_target_' in scheduler_cfg:
                    # Remove _partial_ flag and add optimizer
                    scheduler_cfg.pop('_partial_', None)
                    scheduler_cfg.pop('_convert_', None)
                    scheduler_cfg['optimizer'] = optimizer
                    scheduler = instantiate(scheduler_cfg)
                else:
                    # Fallback: assume it's a direct config
                    scheduler_cfg['optimizer'] = optimizer
                    scheduler = instantiate(scheduler_cfg)
                self.scheduler = scheduler  # Store instantiated scheduler
                # Return with proper Lightning configuration
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',  # Step after optimizer steps (respects accumulate_grad_batches)
                        'frequency': 1,      # Step every optimizer step
                    }
                }
            elif hasattr(self.scheduler, 'func') and hasattr(self.scheduler, 'keywords'):
                # It's a functools.partial (from _partial_=true) - call it with optimizer
                scheduler = self.scheduler(optimizer=optimizer)
                self.scheduler = scheduler  # Store instantiated scheduler
                # Return with proper Lightning configuration
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',  # Step after optimizer steps (respects accumulate_grad_batches)
                        'frequency': 1,      # Step every optimizer step
                    }
                }
            else:
                # If scheduler is already instantiated, just return it
                # Return with proper Lightning configuration
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': self.scheduler,
                        'interval': 'step',  # Step after optimizer steps (respects accumulate_grad_batches)
                        'frequency': 1,      # Step every optimizer step
                    }
                }
        
        return optimizer

    def load_ckpt(self, ckpt_path, device = None, prefix = ''):

        if device is None:
            device = next(self.parameters()).device
            
        if 's3://' in ckpt_path:
            from s3torchconnector import S3Checkpoint
            checkpoint= S3Checkpoint(region='us-east-1')
            with checkpoint.reader(ckpt_path) as f:
                state_dict = torch.load(f, map_location=device)['state_dict']
                print(f"Model loaded from {ckpt_path}")
        else:
            state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
            print(f"Model loaded from {ckpt_path}")
        
        print(state_dict)
        
        try:
            self.load_state_dict(state_dict)
            print("Loaded full state dict")
        except:
            print("Could not load state dict, trying to load only ['encoder'] keys")
            
            try:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k in list(state_dict.keys()):
                    if prefix in k:
                        new_key = k.replace('encoder.','')
                        new_state_dict[new_key] = state_dict[k]
                        
                self.load_state_dict(new_state_dict)
                print(f"Loaded only {prefix} keys")
                
            except Exception as e:
                print(f"Could not load state dict, error: {e}")
            
        

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False