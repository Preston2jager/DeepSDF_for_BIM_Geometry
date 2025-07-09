import os
import yaml

class DictWrapper:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = DictWrapper(v)
            elif isinstance(v, list):
                v = [DictWrapper(i) if isinstance(i, dict) else i for i in v]
            setattr(self, k, v)

class Config:
    def __init__(self):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.main_cfg_path = os.path.join(self.root_dir, 'Configs/configs.yaml')
        with open(self.main_cfg_path, 'r') as f:
            main_cfg_raw = yaml.load(f, Loader=yaml.FullLoader)
        for key, cfg_entry in main_cfg_raw.items():
            # ➤ 如果是 dict 且含 path 字段 → 加载子配置文件并合并
            if isinstance(cfg_entry, dict) and "path" in cfg_entry:
                relative_path = cfg_entry["path"]
                meta_fields = {k: v for k, v in cfg_entry.items() if k != "path"}
                full_path = os.path.normpath(os.path.join(self.root_dir, relative_path))
                with open(full_path, 'r') as subf:
                    sub_config = yaml.load(subf, Loader=yaml.FullLoader) or {}
                combined = {**meta_fields, **sub_config}
                setattr(self, key, DictWrapper(combined))
            elif isinstance(cfg_entry, str):
                full_path = os.path.normpath(os.path.join(self.root_dir, cfg_entry))
                with open(full_path, 'r') as subf:
                    sub_config = yaml.load(subf, Loader=yaml.FullLoader) or {}
                setattr(self, key, DictWrapper(sub_config))
            elif isinstance(cfg_entry, dict):
                setattr(self, key, DictWrapper(cfg_entry))
            else:
                raise ValueError(f"[Config] Invalid entry: {key} = {cfg_entry}")
