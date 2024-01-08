import numpy as np
import struct


# Function to load a 3D mesh model from a PLY file.
def (path, vertex_scale=1.0):
    # Initialize variables and constants.
    face_n_corners = 3
    pt_props, face_props = [], []
    header_vertex_section, header_face_section = False, False
    is_binary, is_normal, is_color, is_texture_pt, is_texture_face = (
        False,
        False,
        False,
        False,
        False,
    )
    model, formats = {}, {
        "float": ("f", 4),
        "double": ("d", 8),
        "int": ("i", 4),
        "uchar": ("B", 1),
    }

    # Open PLY file for reading.
    with open(path, "r") as f:
        # Process PLY header.
        texture_file, n_pts, n_faces = process_header(f)

        if texture_file:
            model["texture_file"] = texture_file

        # Prepare model data structures.
        prepare_model_data_structures(model, n_pts, n_faces, pt_props, face_props)

        # Load vertices and faces.
        load_vertices(f, model, n_pts, pt_props, formats, is_binary)
        load_faces(f, model, n_faces, face_props, formats, is_binary)

    # Apply vertex scale to all points.
    model["pts"] *= vertex_scale

    return model


# Function to process the header of the PLY file.
def process_header(f):
    # Local variable declarations.
    texture_file, n_pts, n_faces = None, 0, 0
    pt_props, face_props = [], []
    header_vertex_section, header_face_section = False, False
    is_binary = False

    # Read and process the header line by line.
    for line in f:
        line = line.rstrip("\n").rstrip("\r")
        line_split = line.split()

        if line.startswith("comment TextureFile"):
            texture_file = line_split[-1]
        elif line.startswith("element vertex"):
            n_pts = int(line_split[-1])
            header_vertex_section, header_face_section = True, False
        elif line.startswith("element face"):
            n_faces = int(line_split[-1])
            header_vertex_section, header_face_section = False, True
        elif line.startswith("element"):
            header_vertex_section, header_face_section = False, False
        elif line.startswith("property") and header_vertex_section:
            pt_props.append(process_vertex_property(line))
        elif line.startswith("property list") and header_face_section:
            face_props.append(process_face_property(line))
        elif line.startswith("format"):
            if "binary" in line:
                is_binary = True
        elif line.startswith("end_header"):
            break

    return texture_file, n_pts, n_faces


# Other helper functions go here... (process_vertex_property, process_face_property, prepare_model_data_structures, load_vertices, load_faces)
def process_vertex_property(line):
    """Process a vertex property from the PLY header."""
    prop_name = line.split()[-1]
    if prop_name == "s":
        prop_name = "texture_u"
    if prop_name == "t":
        prop_name = "texture_v"
    prop_type = line.split()[-2]
    return (prop_name, prop_type)


def process_face_property(line):
    """Process a face property from the PLY header."""
    elems = line.split()
    face_props = []
    if elems[-1] == "vertex_indices" or elems[-1] == "vertex_index":
        face_props.append(("n_corners", elems[2]))
        for i in range(face_n_corners):
            face_props.append(("ind_" + str(i), elems[3]))
    elif elems[-1] == "texcoord":
        face_props.append(("texcoord", elems[2]))
        for i in range(face_n_corners * 2):
            face_props.append(("texcoord_ind_" + str(i), elems[3]))
    else:
        LOGGER.warning("Warning: Not supported face property: " + elems[-1])
    return face_props


def prepare_model_data_structures(model, n_pts, n_faces, pt_props, face_props):
    """Prepare the data structures for the 3D model."""
    model["pts"] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model["faces"] = np.zeros((n_faces, face_n_corners), np.float)
    pt_props_names = [p[0] for p in pt_props]
    face_props_names = [p[0] for p in face_props]
    if {"nx", "ny", "nz"}.issubset(set(pt_props_names)):
        model["normals"] = np.zeros((n_pts, 3), np.float)
    if {"red", "green", "blue"}.issubset(set(pt_props_names)):
        model["colors"] = np.zeros((n_pts, 3), np.float)
    if {"texture_u", "texture_v"}.issubset(set(pt_props_names)):
        model["texture_uv"] = np.zeros((n_pts, 2), np.float)
    if {"texcoord"}.issubset(set(face_props_names)):
        model["texture_uv_face"] = np.zeros((n_faces, 6), np.float)


def load_vertices(f, model, n_pts, pt_props, formats, is_binary):
    """Load vertices from the PLY file."""
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
        prop_vals = load_properties(f, pt_props, formats, is_binary, load_props)
        update_model_vertex(model, pt_id, prop_vals)


def load_faces(f, model, n_faces, face_props, formats, is_binary):
    """Load faces from the PLY file."""
    for face_id in range(n_faces):
        prop_vals = {}
        prop_vals = load_properties(f, face_props, formats, is_binary, None)
        update_model_face(model, face_id, prop_vals)


def load_properties(f, props, formats, is_binary, load_props):
    """Load properties from the PLY file."""
    prop_vals = {}
    if is_binary:
        for prop in props:
            format = formats[prop[1]]
            read_data = f.read(format)


import torch
def binary_to_decimal(binary_batch):
    # Convert binary to decimal
    power_of_2 = torch.flip(torch.pow(2, torch.arange(binary_batch.size(1), dtype=torch.float32, device=binary_batch.device)), [0])
    decimal_batch = torch.matmul(binary_batch.float(), power_of_2)
    unique, inverse, counts = torch.unique(decimal_batch, return_inverse=True, return_counts=True)
    # Find indices where duplicates occur
    duplicate_mask = counts[inverse] > 1
    duplicate_indices = torch.where(duplicate_mask)[0]

    return decimal_batch, duplicate_indices