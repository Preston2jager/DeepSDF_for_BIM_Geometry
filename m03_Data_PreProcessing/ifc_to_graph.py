import yaml
import os
import json

def ifc_to_graph(ifc_cfg_path, output_dir, project_index, ifc, output_prefix='ifc_graph'):
    """
    Generate graphs from ifc, output as json, without latent codes.
    """
    # Get ifc classes from config files
    ifc_cfg = os.path.join(ifc_cfg_path,"ifc.yaml")
    with open(ifc_cfg, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    target_types = data['ifc_classes']

    # Target_types = ["IfcWall", "IfcWindow", "IfcDoor", "IfcSlab"]

    # Initialise containers for element data and mappings
    elements = []
    id_to_index = {}
    index_to_info = {}
    main_wall = []
    all_wall = []

    # Extract elements of target types and record their information.
    for ifc_type in target_types:
        objs = ifc.by_type(ifc_type)
        for obj in objs:
            index = len(elements)
            elements.append(obj)
            global_id = obj.GlobalId
            id_to_index[global_id] = index

            # Use the element index as key so each element's info is stored separately.
            index_to_info[index] = {
                "index": index,
                "GlobalId": global_id,
                "type": ifc_type
            }

    # Build the graph structure with nodes and empty edges list.
    graph = {
        "project": project_index,
        "nodes": list(index_to_info.values()),
        "edges": []  
    }   
    
    globalid_to_index = {v["GlobalId"]: k for k, v in index_to_info.items()}
    slab_index = next(idx for idx, info in index_to_info.items() if info["type"] == "IfcSlab")
    
    # Process IfcWall elements and build edges based on relationships
    for wall in ifc.by_type("IfcWall"):
        wall_index = globalid_to_index[wall.GlobalId]
        graph["edges"].append((slab_index, wall_index))
        all_wall.append(wall_index)
        if hasattr(wall, "HasOpenings"):
            for rel in wall.HasOpenings:
                opening = rel.RelatedOpeningElement
                for fill in getattr(opening, "HasFillings", []):
                    filled = fill.RelatedBuildingElement
                    filled_index = globalid_to_index[filled.GlobalId]
                    graph["edges"].append((wall_index, filled_index))
                    main_wall.append(wall_index)

    side_wall = list(set(all_wall) - set(main_wall))
    for side_index in side_wall:
        for main_index in main_wall:
            graph["edges"].append((side_index, main_index))

    Output_file_name = os.path.join(output_dir, f"{project_index}:{output_prefix}.json")
    with open(Output_file_name, "w") as f:
        json.dump(graph, f, indent=2)