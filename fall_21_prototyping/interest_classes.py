interest_class_names = [
    "Corn",
    "Soybeans",
    "Rice",
    "Alfalfa",
    "Grapes",
    # "Almonds",
    "Pecans",
    # "Peanuts",
    # "Pistachios",
    # "Walnuts",
    "Potatoes",
    "Oats",
    "Apples",
    "Cotton",
    "Dry Beans",
    "Sugarbeets",
    "Winter Wheat",
    "Spring Wheat",
    "Pop or Orn Corn",
    "Other Hay-Non Alfalfa",
    "Grass-Pasture",
    "Woody Wetlands",
    "Herbaceous Wetlands",
    "Developed-Open Space",
    "Deciduous Forest",
    "Fallow-Idle Cropland"
  ]

import json

class_name_to_class_id = json.load(open("classes.json"))
interest_classes = [class_name_to_class_id[c] for c in interest_class_names]
class_id_to_class_name = {v:k for k,v in class_name_to_class_id.items()}

