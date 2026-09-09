import pandas as pd
from pathlib import Path
from .rauwolfia_engine import process_rauwolfia_batch
from .embelia_engine import process_embelia_batch
from .hirda_engine import process_hirda_batch
from .plant_detector import detect_plant
from .utils import detect_modality

def route_and_process(file_obj, models_dir, data_dir, target_plant="Auto-Detect"):
    """Routes the uploaded CSV to the correct engine based on user selection or auto-detection."""
    try:
        sample_df = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
        
    filename = file_obj.name if hasattr(file_obj, 'name') else "uploaded.csv"
    
    if target_plant == "Auto-Detect":
        # FTIR is supported by the Rauwolfia engine. UV files are routed using
        # learned spectral patterns rather than wavelength-column shape.
        if detect_modality(sample_df) == "FTIR":
            res, mod = process_rauwolfia_batch(sample_df, models_dir, filename)
            return res, "Rauwolfia/Terminalia"

        plant_labels, plant_scores, plant_confidence, _ = detect_plant(sample_df, models_dir)
        if any(label == "UNKNOWN / LOW CONFIDENCE" for label in plant_labels):
            raise ValueError("One or more rows could not be identified with sufficient plant-level confidence.")

        # Specialized engines operate on one plant type at a time. Split a
        # mixed upload into groups, process each group, and restore row order.
        groups = {}
        for row_index, plant_label in enumerate(plant_labels):
            groups.setdefault(str(plant_label), []).append(row_index)

        combined_results = []
        route_names = []
        for detected_plant, row_indices in groups.items():
            group_df = sample_df.iloc[row_indices].copy()
            if detected_plant == "Embelia":
                res, mod = process_embelia_batch(group_df, data_dir, filename)
                route_name = "Embelia"
            elif detected_plant == "Hirda":
                res, mod = process_hirda_batch(group_df, models_dir, filename)
                route_name = "Hirda"
            elif detected_plant == "Rauwolfia":
                res, mod = process_rauwolfia_batch(group_df, models_dir, filename)
                route_name = "Rauwolfia/Terminalia"
            else:
                raise ValueError(f"Unsupported detected plant: {detected_plant}")

            for group_position, result in enumerate(res):
                original_index = row_indices[group_position]
                result["sample_name"] = f"{filename} (Row {original_index + 1})"
                result["plant_detection_confidence"] = float(plant_scores[original_index])
                result["plant_detection"] = plant_confidence[original_index]
                result["_input_row"] = original_index
                combined_results.append(result)
            route_names.append(route_name)

        combined_results.sort(key=lambda result: result.pop("_input_row"))
        modality_name = "Mixed UV (" + ", ".join(route_names) + ")" if len(route_names) > 1 else route_names[0]
        return combined_results, modality_name
        
    elif target_plant == "Rauwolfia/Terminalia":
        res, mod = process_rauwolfia_batch(sample_df, models_dir, filename)
        return res, "Rauwolfia/Terminalia"
        
    elif target_plant == "Embelia":
        res, mod = process_embelia_batch(sample_df, data_dir, filename)
        return res, "Embelia"
        
    elif target_plant == "Hirda":
        res, mod = process_hirda_batch(sample_df, models_dir, filename)
        return res, "Hirda"
        
    else:
        raise ValueError(f"Unknown target plant: {target_plant}")
