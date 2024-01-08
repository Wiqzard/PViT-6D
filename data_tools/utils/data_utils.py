import struct
from pathlib import Path

import numpy as np
import cv2

from utils import LOGGER
from utils.pose_ops import rotation_matrix


def require_dataset(method):
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_dataset"):
            return method(self, *args, **kwargs)
        else:
            raise AttributeError(
                "Dataset not loaded yet. Call _create_dataset() first."
            )

    return wrapper


def correct_suffix(path: Path) -> Path:
    """
    Correct the suffix of a path to be .png or .jpg.
    Args:
        path: The path to correct.
    Returns:
        The corrected path.
    """
    path = Path(path)
    if path.is_file():
        return path
    path = path.with_suffix(".png")
    if path.is_file():
        return path
    path = path.with_suffix(".jpg")
    if not path.is_file():
        raise FileNotFoundError(f"Mask {path} does not exist.")
    return path


def crop_square_resize(
    img: np.ndarray, bbox: int, crop_size: int = None, interpolation=None
) -> np.ndarray:
    """
    Crop and resize an image to a square of size crop_size, centered on the given bounding box.

    Args:
    -----------
    img : numpy.ndarray
        Input image to be cropped and resized.
    bbox : int
        Bounding box coordinates of the object of interest. Must be in the format x1, y1, x2, y2.
    crop_size : int
        The size of the output square image. Default is None, which will use the largest dimension of the bbox as the crop_size.
    interpolation : int, optional
        The interpolation method to use when resizing the image. Default is None, which will use cv2.INTER_LINEAR.

    Returns:
    --------
    numpy.ndarray
        The cropped and resized square image.

    Raises:
    -------
    ValueError:
        If crop_size is not an integer.
    """

    if not isinstance(crop_size, int):
        raise ValueError("crop_size must be an int")

    x1, y1, x2, y2 = bbox.xyxy
    bw = bbox.w  # Bbox[2]
    bh = bbox.h  # Bbox[3]
    bbox_center = np.array(bbox.center)  # np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])

    if bh > bw:
        x1 = bbox_center[0] - bh / 2
        x2 = bbox_center[0] + bh / 2
    else:
        y1 = bbox_center[1] - bw / 2
        y2 = bbox_center[1] + bw / 2

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if img.ndim > 2:
        roi_img = np.zeros((max(bh, bw), max(bh, bw), img.shape[2]), dtype=img.dtype)
    else:
        roi_img = np.zeros((max(bh, bw), max(bh, bw)), dtype=img.dtype)
    roi_x1 = max((0 - x1), 0)
    x1 = max(x1, 0)
    roi_x2 = roi_x1 + min((img.shape[1] - x1), (x2 - x1))
    roi_y1 = max((0 - y1), 0)
    y1 = max(y1, 0)
    roi_y2 = roi_y1 + min((img.shape[0] - y1), (y2 - y1))
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    roi_img[roi_y1:roi_y2, roi_x1:roi_x2] = img[y1:y2, x1:x2].copy()
    if roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
        raise ValueError("roi_img is None")
    roi_img = cv2.resize(roi_img, (crop_size, crop_size), interpolation=interpolation)
    return roi_img, [x1, y1, x2, y2]


def crop_pad_resize(
    img: np.ndarray, bbox: int, crop_size: int = None, interpolation=None, padding_value=0
) -> np.ndarray:
    x1, y1, x2, y2 = bbox.xyxy
    bbox_center =  np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    bw, bh = bbox.w, bbox.h
    scale = max(bw, bh)
    if img.ndim > 2:
        roi_img = np.zeros((scale, scale, img.shape[2]), dtype=img.dtype)
    else:
        roi_img = np.zeros((scale, scale), dtype=img.dtype)
    #if bw > bh:
    #    x1, x2 = 0, scale
    x1, y1 = max(x1,0), max(y1,0)
    x2, y2 = min(x2,img.shape[1]), min(y2,img.shape[0])
    roi_x1 = max(int(x1 - bbox_center[0] + scale/2),0) 
    roi_x2 = min(int(x2 - bbox_center[0] + scale/2),scale) 
    roi_y1 = max(int(y1 - bbox_center[1] + scale/2),0) 
    roi_y2 = min(int(y2 - bbox_center[1] + scale/2),scale) 

    
    roi_img[roi_y1:roi_y2,roi_x1:roi_x2] = img[y1:y2, x1:x2].copy()
    roi_img = cv2.resize(roi_img, (crop_size, crop_size), interpolation=interpolation)
    return roi_img


def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW", endpoint=False):
    """
    Args:
        width:
        height:
        endpoint: whether to include the endpoint
    Returns:
        xy: (2, height, width)
    """
    # coords values are in [low, high]  [0,1] or [-1,1]
    x = np.linspace(low, high, width, dtype=np.float32, endpoint=endpoint)
    y = np.linspace(low, high, height, dtype=np.float32, endpoint=endpoint)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy


def get_bbox3d_and_center(pts):
    """
    pts: Nx3
    ---
    bb: bb3d+center, 9x3
    """
    bb = []
    minx, maxx = min(pts[:, 0]), max(pts[:, 0])
    miny, maxy = min(pts[:, 1]), max(pts[:, 1])
    minz, maxz = min(pts[:, 2]), max(pts[:, 2])
    avgx = np.average(pts[:, 0])
    avgy = np.average(pts[:, 1])
    avgz = np.average(pts[:, 2])
    bb = np.array(
        [
            [maxx, maxy, maxz],
            [minx, maxy, maxz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, maxy, minz],
            [minx, maxy, minz],
            [minx, miny, minz],
            [maxx, miny, minz],
            [avgx, avgy, avgz],
        ],
        dtype=np.float32,
    )
    return bb


def load_ply(path, vertex_scale=1.0):
    f = open(path, "rb")
    face_n_corners = 3
    n_pts = 0
    n_faces = 0
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False
    texture_file = None

    # Read the header.
    while True:
        # Strip the newline character(s)
        line = f.readline()
        if isinstance(line, str):
            line = line.rstrip("\n").rstrip("\r")
        else:
            line = str(line, "utf-8").rstrip("\n").rstrip("\r")

        if line.startswith("comment TextureFile"):
            texture_file = line.split()[-1]
        elif line.startswith("element vertex"):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith("element"):  # Some other element.
            header_vertex_section = False
            header_face_section = False
        elif line.startswith("property") and header_vertex_section:
            # (name of the property, data type)
            prop_name = line.split()[-1]
            if prop_name == "s":
                prop_name = "texture_u"
            if prop_name == "t":
                prop_name = "texture_v"
            prop_type = line.split()[-2]
            pt_props.append((prop_name, prop_type))
        elif line.startswith("property list") and header_face_section:
            elems = line.split()
            if elems[-1] == "vertex_indices" or elems[-1] == "vertex_index":
                # (name of the property, data type)
                face_props.append(("n_corners", elems[2]))
                for i in range(face_n_corners):
                    face_props.append(("ind_" + str(i), elems[3]))
            elif elems[-1] == "texcoord":
                # (name of the property, data type)
                face_props.append(("texcoord", elems[2]))
                for i in range(face_n_corners * 2):
                    face_props.append(("texcoord_ind_" + str(i), elems[3]))
            else:
                LOGGER.warning("Warning: Not supported face property: " + elems[-1])
        elif line.startswith("format"):
            if "binary" in line:
                is_binary = True
        elif line.startswith("end_header"):
            break

    # Prepare data structures.
    model = {}
    if texture_file is not None:
        model["texture_file"] = texture_file
    model["pts"] = np.zeros((n_pts, 3), np.float32)
    if n_faces > 0:
        model["faces"] = np.zeros((n_faces, face_n_corners), np.float32)

    pt_props_names = [p[0] for p in pt_props]
    face_props_names = [p[0] for p in face_props]

    is_normal = False
    if {"nx", "ny", "nz"}.issubset(set(pt_props_names)):
        is_normal = True
        model["normals"] = np.zeros((n_pts, 3), np.float32)

    is_color = False
    if {"red", "green", "blue"}.issubset(set(pt_props_names)):
        is_color = True
        model["colors"] = np.zeros((n_pts, 3), np.float32)

    is_texture_pt = False
    if {"texture_u", "texture_v"}.issubset(set(pt_props_names)):
        is_texture_pt = True
        model["texture_uv"] = np.zeros((n_pts, 2), np.float32)

    is_texture_face = False
    if {"texcoord"}.issubset(set(face_props_names)):
        is_texture_face = True
        model["texture_uv_face"] = np.zeros((n_faces, 6), np.float32)

    # Formats for the binary case.
    formats = {
        "float": ("f", 4),
        "double": ("d", 8),
        "int": ("i", 4),
        "uchar": ("B", 1),
    }

    # Load vertices.
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "red",
            "green",
            "blue",
            "texture_u",
            "texture_v",
        ]
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                read_data = f.read(format[1])
                val = struct.unpack(format[0], read_data)[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model["pts"][pt_id, 0] = float(prop_vals["x"])
        model["pts"][pt_id, 1] = float(prop_vals["y"])
        model["pts"][pt_id, 2] = float(prop_vals["z"])

        if is_normal:
            model["normals"][pt_id, 0] = float(prop_vals["nx"])
            model["normals"][pt_id, 1] = float(prop_vals["ny"])
            model["normals"][pt_id, 2] = float(prop_vals["nz"])

        if is_color:
            model["colors"][pt_id, 0] = float(prop_vals["red"])
            model["colors"][pt_id, 1] = float(prop_vals["green"])
            model["colors"][pt_id, 2] = float(prop_vals["blue"])

        if is_texture_pt:
            model["texture_uv"][pt_id, 0] = float(prop_vals["texture_u"])
            model["texture_uv"][pt_id, 1] = float(prop_vals["texture_v"])

    # Load faces.
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == "n_corners":
                    if val != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                        # print("Number of face corners: " + str(val))
                        # exit(-1)
                elif prop[0] == "texcoord":
                    if val != face_n_corners * 2:
                        raise ValueError("Wrong number of UV face coordinates.")
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == "n_corners":
                    if int(elems[prop_id]) != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                elif prop[0] == "texcoord":
                    if int(elems[prop_id]) != face_n_corners * 2:
                        raise ValueError("Wrong number of UV face coordinates.")
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model["faces"][face_id, 0] = int(prop_vals["ind_0"])
        model["faces"][face_id, 1] = int(prop_vals["ind_1"])
        model["faces"][face_id, 2] = int(prop_vals["ind_2"])

        if is_texture_face:
            for i in range(6):
                model["texture_uv_face"][face_id, i] = float(
                    prop_vals["texcoord_ind_{}".format(i)]
                )

    f.close()
    model["pts"] *= vertex_scale

    return model


def get_symmetry_transformations(model_info, max_sym_disc_step):
    """Returns a set of symmetry transformations for an object model.

    :param model_info: See files models_info.json provided with the datasets.
    :param max_sym_disc_step: The maximum fraction of the object diameter which
      the vertex that is the furthest from the axis of continuous rotational
      symmetry travels between consecutive discretized rotations.
    :return: The set of symmetry transformations.
    """
    # NOTE: t is in mm, so may need to devide 1000
    # Discrete symmetries.
    trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity.
    if "symmetries_discrete" in model_info:
        for sym in model_info["symmetries_discrete"]:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_disc.append({"R": R, "t": t})

    # Discretized continuous symmetries.
    trans_cont = []
    if "symmetries_continuous" in model_info:
        for sym in model_info["symmetries_continuous"]:
            axis = np.array(sym["axis"])
            offset = np.array(sym["offset"]).reshape((3, 1))

            # (PI * diam.) / (max_sym_disc_step * diam.) = discrete_steps_count
            discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))

            # Discrete step in radians.
            discrete_step = 2.0 * np.pi / discrete_steps_count

            for i in range(1, discrete_steps_count):
                R = rotation_matrix(i * discrete_step, axis)[:3, :3]
                t = -(R.dot(offset)) + offset
                trans_cont.append({"R": R, "t": t})

    # Combine the discrete and the discretized continuous symmetries.
    trans = []
    for tran_disc in trans_disc:
        if len(trans_cont):
            for tran_cont in trans_cont:
                R = tran_cont["R"].dot(tran_disc["R"])
                t = tran_cont["R"].dot(tran_disc["t"]) + tran_cont["t"]
                trans.append({"R": R, "t": t})
        else:
            trans.append(tran_disc)

    return trans
