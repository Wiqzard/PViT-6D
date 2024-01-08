"""
This file contains the meta data for the datasets used in the project.
"""


class YCBVMetaData:
    height: int = 480
    width: int = 640
    z_near: float = 0.25
    z_far: float = 6.0
    num_classes: int = 21
    diameters: dict[int, float] = {
        0: 172.063 / 1000,
        1: 269.573 / 1000,
        2: 198.377 / 1000,
        3: 120.543 / 1000,
        4: 196.463 / 1000,
        5: 89.797 / 1000,
        6: 142.543 / 1000,
        7: 114.053 / 1000,
        8: 129.540 / 1000,
        9: 197.796 / 1000,
        10: 259.534 / 1000,
        11: 259.566 / 1000,
        12: 161.922 / 1000,
        13: 124.990 / 1000,
        14: 226.170 / 1000,
        15: 237.299 / 1000,
        16: 203.973 / 1000,
        17: 121.365 / 1000,
        18: 174.746 / 1000,
        19: 217.094 / 1000,
        20: 102.903 / 1000,
    }

    class_names: dict[int, str] = {
        0: "013_master_chef_can",
        1: "014_cracker_box",
        2: "015_sugar_box",
        3: "015_tomato_soup_can",
        4: "015_mustard_bottle",
        5: "015_tuna_fish_can",
        6: "13_pudding_box",
        7: "13_gelatin_box",
        8: "013_potted_meat_can",
        9: "011_banana",
        10: "019_pitcher_base",
        11: "021_bleach_cleanser",
        12: "024_bowl",
        13: "025_mug",
        14: "035_power_drill",
        15: "036_wood_block",
        16: "037_scissors",
        17: "040_large_marker",
        18: "051_large_clamp",
        19: "052_extra_large_clamp",
        20: "061_foam_brick",
    }

    binary_symmetries: dict[str, bool] = {
        0: False,
        1: False,
        2: False,
        3: False,
        4: False,
        5: False,
        6: False,
        7: False,
        8: False,
        9: False,
        10: False,
        11: False,
        12: True,
        13: False,
        14: False,
        15: True,
        16: False,
        17: False,
        18: True,
        19: True,
        20: True,
    }

    mapping: dict[int, int] = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        18: 18,
        19: 19,
        20: 20,
    }


class LMMetaData:
    height: int = 480
    width: int = 640
    z_near: float = 0.25
    z_far: float = 6.0
    num_classes: int = 15
    class_names: dict[int, str] = {
        0: "ape",
        1: "benchvise",
        2: "bowl",
        3: "camera",
        4: "can",
        5: "cat",
        6: "cup",
        7: "driller",
        8: "duck",
        9: "eggbox",
        10: "glue",
        11: "holepuncher",
        12: "iron",
        13: "lamp",
        14: "phone",
    }
    binary_symmetries: dict[int, bool] = {
        0: False,
        1: False,
        2: False,
        3: False,
        4: False,
        5: False,
        6: False,
        7: False,
        8: False,
        9: True,
        10: True,
        11: False,
        12: False,
        13: False,
        14: False,
    }
    diameters: dict[int, float] = {
        0: 102.099 / 1000,
        1: 247.506 / 1000,
        2: 167.355 / 1000,
        3: 172.492 / 1000,
        4: 201.404 / 1000,
        5: 154.546 / 1000,
        6: 124.264 / 1000,
        7: 261.472 / 1000,
        8: 108.999 / 1000,
        9: 164.628 / 1000,
        10: 175.889 / 1000,
        11: 145.543 / 1000,
        12: 278.078 / 1000,
        13: 282.601 / 1000,
        14: 212.358 / 1000,
    }


class LMOMetaData:
    height: int = 480
    width: int = 640
    z_near: float = 0.25
    z_far: float = 6.0
    num_classes: int = 8
    mapping: dict[int, int] = {
        0: 0,
        1: 4,
        2: 5,
        3: 7,
        4: 8,
        5: 9,
        6: 10,
        7: 11,
    }

    class_names: dict[int, str] = {
        0: "ape",
        4: "can",
        5: "cat",
        7: "driller",
        8: "duck",
        9: "eggbox",
        10: "glue",
        11: "holepuncher",
    }
    binary_symmetries: dict[str, bool] = {
        0: False,
        4: False,
        5: False,
        7: False,
        8: False,
        9: True,
        10: True,
        11: False,
    }
    diameters = {
        0: 102.099 / 1000,
        4: 201.404 / 1000,
        5: 154.546 / 1000,
        7: 261.472 / 1000,
        8: 108.999 / 1000,
        9: 164.628 / 1000,
        10: 175.889 / 1000,
        11: 145.543 / 1000,
    }


