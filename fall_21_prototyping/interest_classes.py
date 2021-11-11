interest_class_names = [
    "Corn",
    "Soybeans",
    "Rice",
    "Alfalfa",
    "Grapes",
    "Almonds",
    "Pecans",
    "Peanuts",
    # "Pistachios",
    "Walnuts",
    "Potatoes",
    "Oats",
    # "Apples",
    "Cotton",
    "Dry Beans",
    "Sugarbeets",
    "Winter Wheat",
    "Spring Wheat",
    "Durum Wheat",
    "Sorghum",
    "Canola",
    "Barley",
    "Sunflower",
    "Pop or Orn Corn",
    "Other Hay-Non Alfalfa",
    # "Grass-Pasture",
    "Woody Wetlands",
    # "Herbaceous Wetlands",
    # "Developed-Open Space",
    # "Deciduous Forest",
    "Fallow-Idle Cropland"
]

import json

with open('classes.json', 'r') as f:
    class_name_to_class_id = json.load(f)

all_classes = list(class_name_to_class_id.values())
interest_classes = [class_name_to_class_id[c] for c in interest_class_names]
# interest_classes = all_classes
class_id_to_class_name = {v: k for k, v in class_name_to_class_id.items()}
