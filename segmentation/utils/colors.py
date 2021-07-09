import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


def get_color_choice(i):
    """
    Creates an `(r, g, b)` tuple (values from 0 to 255) for a given index. \n
    Requires: \n
      `i`: Any non-negative integer. \n
    Returns: \n
      `color`: An `(r, g, b)` tuple of 8-bit values from 0 to 255.
    """
    sh = lambda m: (i << m) % 255
    color_choice = {
        0: (255, sh(6), sh(3)), 1: (sh(6), 255, sh(3)), 2:(sh(6), sh(3), 255),
        3: (255, sh(2), sh(4)), 4: (sh(2), 255, sh(4)), 5: (sh(2), sh(4), 255),
        6: (255, 255, sh(3)), 7:(255, sh(3), 255), 8:(sh(3), 255, 255)
    }
    return color_choice.get(i % 9)


def plot_color_legend(classes, save_image=False):
    """
    Plots color legend for each class in `classes`.
    Requires:
      `classes`: Dictionary of `class_name` --> `class_index` \n
    """
    handles = []
    for class_name, class_idx in classes.items():
        r, g, b = get_color_choice(class_idx)
        color = r/255, g/255, b/255
        patch = mpatches.Patch(color=color, label=class_name)

        handles.append(patch)

    plt.legend(
        handles=handles,
        ncol=5,
        mode="expand",
        fontsize="x-small",
        title="Color legend")
    plt.show()


def get_cmap(classes):
    color_list = ['xkcd:dark purple'] * (max(classes.values()) + 1)
    color_list[0] = 'xkcd:black'

    # Set value at index of class_name to xkcd color string
    for class_name, color in COLORS.items():
        if type(color) is str:
            color_list[classes[class_name]] = color

    return ListedColormap(color_list)


COLORS = {
  "Corn": "xkcd:red",
  "Cotton": "xkcd:pale cyan",
  "Rice": "xkcd:blue",
  "Sorghum": "xkcd:brown",
  "Soybeans": "xkcd:green",
  "Sunflower": "xkcd:sunflower",
  "Peanuts": "xkcd:ochre",
  "Tobacco": "xkcd:crimson",
  "Sweet Corn": "xkcd:pale purple",
  "Pop or Orn Corn": "xkcd:butter yellow",
  "Mint": "xkcd:mint green",
  "Barley": "xkcd:dark tan",
  "Durum Wheat": "xkcd:orange brown",
  "Spring Wheat": "xkcd:purplish red",
  "Winter Wheat": "xkcd:wheat",
  "Other Small Grains": "xkcd:puke",
  "Dbl Crop WinWht-Soybeans": "xkcd:maroon",
  "Rye": "xkcd:lilac",
  "Oats": "xkcd:off white",
  "Millet": "xkcd:periwinkle",
  "Speltz": "xkcd:indigo",
  "Canola": "xkcd:rose",
  "Flaxseed": "xkcd:khaki",
  "Safflower": "xkcd:saffron",
  "Rape Seed": "xkcd:pale teal",
  "Mustard": "xkcd:mustard",
  "Alfalfa": "xkcd:yellow",
  "Camelina": 38,
  "Buckwheat": 39,
  "Sugarbeets": "xkcd:jade",
  "Dry Beans": "xkcd:ugly green",
  "Potatoes": "xkcd:gold",
  "Other Crops": 44,
  "Sugarcane": 45,
  "Sweet Potatoes": 46,
  "Misc Vegs & Fruits": 47,
  "Watermelons": "xkcd:watermelon",
  "Onions": 49,
  "Cucumbers": 50,
  "Chick Peas": 51,
  "Lentils": 52,
  "Peas": "xkcd:pea green",
  "Tomatoes": "xkcd:tomato",
  "Caneberries": "xkcd:cranberry",
  "Hops": 56,
  "Herbs": 57,
  "Clover-Wildflowers": 58,
  "Fallow-Idle Cropland": "xkcd:white",
  "Cherries": "xkcd:cherry",
  "Peaches": "xkcd:peach",
  "Apples": "xkcd:light red",
  "Grapes": "xkcd:grape",
  "Christmas Trees": 70,
  "Other Tree Crops": 71,
  "Citrus": 72,
  "Pecans": "xkcd:goldenrod",
  "Almonds": "xkcd:brick",
  "Walnuts": "xkcd:sienna",
  "Pears": "xkcd:pear",
  "Pistachios": "xkcd:pistachio",
  "Triticale": 205,
  "Carrots": 206,
  "Asparagus": 207,
  "Garlic": 208,
  "Cantaloupes": 209,
  "Prunes": 210,
  "Olives": "xkcd:olive",
  "Oranges": "xkcd:pale orange",
  "Honeydew Melons": 213,
  "Broccoli": 214,
  "Avocado": "xkcd:avocado",
  "Peppers": 216,
  "Pomegranates": 217,
  "Nectarines": 218,
  "Greens": 219,
  "Plums": 220,
  "Strawberries": "xkcd:strawberry",
  "Squash": 222,
  "Apricots": "xkcd:apricot",
  "Vetch": 224,
  "Dbl Crop WinWht-Corn": 225,
  "Dbl Crop Oats-Corn": 226,
  "Lettuce": 227,
  "Dbl Crop Triticale-Corn": 228,
  "Pumpkins": "xkcd:pumpkin",
  "Dbl Crop Lettuce-Durum Wht": 230,
  "Dbl Crop Lettuce-Cantaloupe": 231,
  "Dbl Crop Lettuce-Cotton": 232,
  "Dbl Crop Lettuce-Barley": 233,
  "Dbl Crop Durum Wht-Sorghum": 234,
  "Dbl Crop Barley-Sorghum": 235,
  "Dbl Crop WinWht-Sorghum": 236,
  "Dbl Crop Barley-Corn": 237,
  "Dbl Crop WinWht-Cotton": 238,
  "Dbl Crop Soybeans-Cotton": 239,
  "Dbl Crop Soybeans-Oats": 240,
  "Dbl Crop Corn-Soybeans": 241,
  "Blueberries": "xkcd:blueberry",
  "Cabbage": 243,
  "Cauliflower": 244,
  "Celery": "xkcd:celery",
  "Radishes": 246,
  "Turnips": 247,
  "Eggplants": "xkcd:eggplant",
  "Gourds": 249,
  "Cranberries": "xkcd:cranberry",
  "Dbl Crop Barley-Soybeans": 254,
  "Other Hay-Non Alfalfa": "xkcd:bruise",
  "Sod-Grass Seed": 59,
  "Switchgrass": 60,
  "Forest": 63,
  "Shrubland-0": 64,
  "Barren-0": 65,
  "Clouds-No Data": 81,
  "Developed": 82,
  "Water": "xkcd:ocean blue",
  "Wetlands": 87,
  "Nonag-Undefined": 88,
  "Aquaculture": 92,
  "Open Water": "xkcd:ocean blue",
  "Perennial Ice-Snow": 112,
  "Developed-Open Space": "xkcd:light grey",
  "Developed-Low Intensity": "xkcd:grey",
  "Developed-Med Intensity": "xkcd:medium grey",
  "Developed-High Intensity": "xkcd:dark grey",
  "Barren": 131,
  "Deciduous Forest": 141,
  "Evergreen Forest": 142,
  "Mixed Forest": 143,
  "Shrubland": 152,
  "Grass-Pasture": "xkcd:grass",
  "Woody Wetlands": "xkcd:dark green",
  "Herbaceous Wetlands": "xkcd:greenish blue"
}