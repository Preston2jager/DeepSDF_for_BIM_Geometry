from cls_configs import Config
from InquirerPy import inquirer

options = ["Train", "Extract", "Reconstruct", "Pathes"]
choice = inquirer.select(
    message="请选择要加载的配置：",
    choices=options,
    default="Train"
).execute()





cfg = Config()
print(cfg.Train.patience)
print(cfg.Pathes.Raw_IFC_folder_path)
print(cfg.Train.batch_size)
print(cfg.Extract.ifc_classes)


