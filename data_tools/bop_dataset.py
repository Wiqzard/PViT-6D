from typing import Any, Optional, Tuple, Union, List, Dict
from pathlib import Path
import json
import dill
import json
import copy

from tqdm import tqdm
import torch
import numpy as np
from pytorch3d.ops import sample_farthest_points
import trimesh

from data_tools.utils.data_utils import (
    get_symmetry_transformations,
    get_bbox3d_and_center,
)
from utils import LOGGER, RANK, colorstr
from utils.flags import Mode, DatasetType, ImageType

from .utils.data_utils import require_dataset, correct_suffix
from .utils.metadata import LMMetaData, LMOMetaData, YCBVMetaData


class BOPDataset:
    def __init__(
        self,
        root_path: str,
        mode: Mode,
        dataset_type: DatasetType = DatasetType.LM,
        set_types: Union[str, List[str]] = "train_pbr",
        use_cache: bool = False,
        single_object: Optional[bool] = False,
        num_points: Optional[int] = 3000,
        window_size: Optional[int] = 1,
        stride: Optional[int] = 1,
        det: Optional[str] = None,
        debug: Optional[bool] = False,
        ablation: Optional[bool] = False,
    ) -> None:
        self.root_path = Path(root_path)
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"Path {self.root_path} does not exist.")
        self.debug = debug
        self.dataset_type = dataset_type

        # set meta data (names, symmetries, diameters, etc.)
        #        match self.dataset_type:
        #            case DatasetType.LM:
        #                self.metadata = LMMetaData()
        #            case DatasetType.LMO:
        #                self.metadata = LMOMetaData()
        #            case DatasetType.YCBV:
        #                self.metadata = YCBVMetaData()

        if self.dataset_type == DatasetType.LM:
            self.metadata = LMMetaData()
        elif self.dataset_type == DatasetType.LMO:
            self.metadata = LMOMetaData()
        elif self.dataset_type == DatasetType.YCBV:
            self.metadata = YCBVMetaData()
        elif self.dataset_type == DatasetType.ABLATION:
            self.metadata = LMMetaData()

        # set dataset name and path
        self.dataset_name = self.dataset_type.name.lower()
        self._path = (
            self.root_path / self.dataset_name
            if not self.dataset_type == DatasetType.CUSTOM
            else self.root_path
        )
        if not self._path.is_dir():
            raise FileNotFoundError(f"Path {self._path} does not exist.")
        self.faulty_labels = []
        self.use_cache = use_cache
        self.mode = mode
        self.ablation = ablation
        self.set_types = set_types if isinstance(set_types, list) else [set_types]
        self.scale_to_meter = 0.001
        self.single_object = single_object
        self.num_points = num_points
        self.window_size = window_size
        self.stride = stride
        self.det = det
        self.sym_infos = {}

        # select folders for training and testing
        self.split_paths = []
        for set_type in self.set_types:
            set_path = self._path / set_type.lower()
            if set_path.is_dir():
                self.split_paths.append(set_path)
        if self.mode in (Mode.TRAIN, Mode.DEBUG):
            self.models_root = self._path / "models"
        else:
            self.models_root = self._path / "models_eval"
        if self.dataset_type == DatasetType.CUSTOM:
            self.models_root = self._path / "models"

        self.model_names = [
            f"{int(file.stem.split('_')[-1]):06d}"
            for file in sorted(self.models_root.glob("obj_*.ply"))
        ]
        store_mode = "multi"
        if self.single_object:
            store_mode = "single"
        elif self.window_size > 1:
            store_mode = "recurrent"
        det = self.det if (self.det and self.mode == Mode.TEST) else "gt"
        debug = "debug" if self.debug else "full"
        mode = (
            self.mode.name.lower()
            if not ablation
            else self.mode.name.lower() + "_ablation"
        )
        # Path.cwd()
        self.cache_path = (
            self._path
            / ".cache"
            / f"dataset_{self.dataset_name}_{mode}_{store_mode}_{det}_{debug}.npy"
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.scene_paths = [
            folder
            for split_path in self.split_paths
            for folder in split_path.iterdir()
            if folder.is_dir() and folder.name.isdigit()
        ]
        self.model_points, self.model_bbox3d = self._load_model_points(
            num_points=num_points
        )
        self.model_symmetries = self._load_model_symmetries()
        self.model_keypoints = self._load_model_keypoints()

        if use_cache and self.cache_path.is_file():
            self._load_cache()
            if bool(self._dataset["single_object"]) != self.single_object:
                raise ValueError(
                    "Dataset was cached with different single_object mode."
                )
        else:
            self._dataset = self._create_dataset()
            if store_mode == "multi":
                self._cache()
            else:
                self._dataset_to_single_view() if store_mode == "single" else self._dataset_mult_to_rec_single_view()
                self._cache()

    def _load_cache(self) -> None:
        """
        Load the dataset from disk.
        """
        LOGGER.info(f"Loading dataset from {self.cache_path}.")
        try:
            import gc

            gc.disable()
            self._dataset, exists = (
                np.load(str(self.cache_path), allow_pickle=True).item(),
                True,
            )  # load dict
            gc.enable()
        except (FileNotFoundError, AssertionError, AttributeError):
            self._dataset = self._create_dataset()
            exists = self._cache(), False

    def _cache(self) -> None:
        """
        Cache the dataset to disk.
        """
        LOGGER.info(f"Caching dataset to {self.cache_path}.")
        if not self.cache_path.parent.is_dir():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(self.cache_path), self._dataset, allow_pickle=True)

    def _load_dicts(self, scene_path: str) -> Tuple[Dict[str, Any]]:
        """
        Load ground truth and camera dictionaries from specified scene path.

        Args:
            scene_path (str): Path to the directory containing the scene_gt.json, scene_gt_info.json, and scene_camera.json files.

        Returns:
            A tuple of three dictionaries:
            - gt_dict: A dictionary containing ground truth pose information about the scene.
            - gt_info_dict: A dictionary containing bounding box information of the ground truth objects.
            - cam_dict: A dictionary containing camera information about the scene.
        """
        fix = "_" + self.det if (self.det and self.mode == Mode.TEST) else ""
        with open(scene_path / f"scene_gt{fix}.json", "r") as f:
            gt_dict = json.load(f)
        with open(scene_path / f"scene_gt_info{fix}.json", "r") as f:
            gt_info_dict = json.load(f)
        with open(scene_path / "scene_camera.json", "r") as f:
            cam_dict = json.load(f)
        return gt_dict, gt_info_dict, cam_dict

    def _load_model_points(self, num_points: int) -> Dict[int, np.ndarray]:
        cache_path = (
            self.cache_path.parent
            / f"models_{self.dataset_name}_{self.mode.name.lower()}_{self.num_points}.pkl"
        )
        if cache_path.exists() and self.use_cache:
            with open(cache_path, "rb") as f:
                model_points, bbox3d = dill.load(f)
                return model_points, bbox3d

        models = {}
        for model_num in self.model_names:
            mesh = h.load_mesh(
                self.models_root / f"obj_{model_num}.ply",
                vertex_scale=0.001,  # self.scale_to_meter,
            )
            model = {}
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            model["pts"] = vertices
            model["bbox3d_and_center"] = get_bbox3d_and_center(vertices)
            models[int(model_num)] = model

        cur_model_points = {}
        cur_model_3dbbox = {}
        num = np.inf
        for obj_id, model_points in models.items():
            model_points = model_points["pts"]
            cur_model_points[obj_id] = model_points
            if model_points.shape[0] < num:
                num = model_points.shape[0]
            cur_model_3dbbox[obj_id] = models[obj_id]["bbox3d_and_center"]
        num = min(num, num_points)
        for i in cur_model_points.keys():
            N = cur_model_points[i].shape[0]
            keep_idx = np.random.choice(N, num, replace=False)
            cur_model_points[i] = cur_model_points[i][keep_idx, :]

        LOGGER.info(f"cache models to {cache_path}")
        with open(cache_path, "wb") as cache_path:
            dill.dump([cur_model_points, cur_model_3dbbox], cache_path)
        return cur_model_points, cur_model_3dbbox

    def _load_model_keypoints(self) -> Dict[int, np.ndarray]:
        cache_path = (
            self.cache_path.parent
            / f"keypoints_{self.dataset_name}_{self.mode.name.lower()}_{self.num_points}.pkl"
        )
        if cache_path.exists() and self.use_cache:
            with open(cache_path, "rb") as f:
                model_keypoints = dill.load(f)
            return model_keypoints

        models_kp = {}
        pts = torch.stack([torch.from_numpy(pts) for pts in self.model_points.values()])
        kpts_dict = {}
        for i in range(12):
            n_kpts = int(4 * 2**i)
            kpts = sample_farthest_points(pts, K=n_kpts)[0]
            kpts_dict[n_kpts] = kpts.numpy()

        models_kp = {}
        for i, n in enumerate(self.model_points.keys()):
            kpts_dict_ = {n_kpts: kpts[i, ...] for n_kpts, kpts in kpts_dict.items()}
            models_kp[n] = kpts_dict_

        LOGGER.info(f"cache keypoints to {cache_path}")
        with open(cache_path, "wb") as cache_path:
            dill.dump(models_kp, cache_path)
        return models_kp

    def _dataset_to_single_view(self) -> None:
        """
        Transforms the dataset dictionary to a single view dataset. I.e. each object
        in an image is treated as a separate image.
        """
        raw_img_dataset = self._dataset["raw_img_dataset"].copy()
        single_raw_img_dataset = []
        for entry in tqdm(raw_img_dataset):
            annotation = entry["annotation"]
            for i in range(len(annotation["obj_id"])):
                entry_c = copy.deepcopy(entry)
                entry_c["annotation"] = {k: [v[i]] for k, v in annotation.items()}
                single_raw_img_dataset.append(entry_c)
        self._dataset["raw_img_dataset"] = single_raw_img_dataset
        self._dataset.update({"single_object": self.single_object})

    def get_ensemble_indices(self, obj_id: int):
        assert self.single_object
        assert obj_id != 0 and obj_id <= len(self.metadata.class_names) + 1
        ensemble_indices = []

        for idx in range(len(self._dataset["raw_img_dataset"])):
            if obj_id in self.get_obj_ids(idx):
                ensemble_indices.append(idx)
        return ensemble_indices

    def _dataset_mult_to_rec_single_view(self) -> None:
        """
        Transforms the dataset dictionary to a recurrent view dataset. I.e. idx carries
        a window size in the same order, without switching scenes.
        """
        raw_img_dataset = self._dataset["raw_img_dataset"].copy()
        start_scene_id = raw_img_dataset[0]["scene_id"]
        start_img_id = raw_img_dataset[0]["img_id"]
        num_entries = len(raw_img_dataset)
        dataset = []
        for i in tqdm(range(0, num_entries, self.stride)):
            img_id = raw_img_dataset[i]["img_id"]
            scene_id = raw_img_dataset[i]["scene_id"]
            cam = raw_img_dataset[i]["cam"]

            if scene_id != start_scene_id:
                start_scene_id = scene_id
                start_img_id = img_id + self.stride
                continue
            if i + self.window_size - 1 >= num_entries:
                continue
            # window_annotations = {
            #    k: [] for k in raw_img_dataset[i]["annotation"]["obj_id"]
            # }
            pl = {k: [] for k in raw_img_dataset[i]["annotation"]}
            window_annotations = {
                k: copy.deepcopy(pl) for k in raw_img_dataset[i]["annotation"]["obj_id"]
            }
            img_paths = []
            img_ids = []
            for j in range(self.window_size):
                entry = raw_img_dataset[i + j]
                cr_img_id = entry["img_id"]
                cur_img_path = entry["img_path"]
                if cr_img_id != img_id + j:
                    continue
                if scene_id != start_scene_id:
                    continue
                annotation = entry["annotation"]
                for k, obj_id in enumerate(annotation["obj_id"]):
                    ann = {key: [v[k]] for key, v in annotation.items()}
                    if obj_id in window_annotations:
                        # window_annotations[obj_id].append(ann)
                        for a, b in window_annotations[obj_id].items():
                            window_annotations[ann["obj_id"][0]][a].append(ann[a][0])

                        # TODO: further improvements such that iou, pose difference, etc. can be computed
                    assert (entry["cam"] == cam).all()
                    assert entry["scene_id"] == scene_id
                img_paths.append(str(cur_img_path))
                img_ids.append(cr_img_id)

            # remove all window_annotations that do not have length window_size
            # window_annotations = {
            #    k: v
            #    for k, v in window_annotations.items()
            #    if len(v) == self.window_size
            # }
            for window_annotation in window_annotations.values():
                if not len(window_annotation["obj_id"]) == self.window_size:
                    continue
                new_annotations = {
                    "img_id": img_ids,
                    "scene_id": scene_id,
                    "cam": entry["cam"],
                    "img_path": img_paths,
                    "annotation": window_annotation,
                }
                dataset.append(new_annotations)
            start_img_id += self.stride

        self._dataset["raw_img_dataset"] = dataset
        self._dataset["recurrent"] = True

    def _create_annotation(
        self,
        scene_root: str,
        str_im_id: str,
        gt_dict: Dict[str, Any],
        gt_info_dict: Dict[str, Any],
    ) -> Dict[str, List[Any]]:
        """
        Create an annotation dictionary containing information about a single image.

        Args:
            scene_root: The path to the directory containing the scene information.
            str_im_id: The ID of the image to create the annotation for.
            gt_dict: A dictionary containing ground truth information about the scene.
            gt_info_dict: A dictionary containing additional information about the ground truth objects.

        Returns:
            A dictionary containing the following keys:
            - obj_id: A list of object IDs in the image.
            - bbox_obj: A list of object bounding boxes in the image.
            - bbox_visib: A list of visible bounding boxes in the image.
            - pose: A list of object poses in the image.
            - mask_path: A list of paths to the masks for the objects in the image.
            - mask_visib_path: A list of paths to the visible masks for the objects in the image.
            - graph_path: A list of paths to the XYZ data for the objects in the image.
        """

        annotations = {
            "obj_id": [],
            "bbox_obj": [],
            "bbox_visib": [],
            "pose": [],
            "mask_path": [],
            "mask_visib_path": [],
        }
        for anno_i, anno in enumerate(gt_dict[str_im_id]):
            obj_id = anno["obj_id"]
            R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
            t = np.array(anno["cam_t_m2c"], dtype="float32")
            pose = np.hstack([R, t.reshape(3, 1)])

            bbox_visib = np.array(
                gt_info_dict[str_im_id][anno_i]["bbox_visib"], dtype=np.int32
            )
            bbox_obj = np.array(
                gt_info_dict[str_im_id][anno_i]["bbox_obj"], dtype=np.int32
            )
            if self.mode == Mode.TEST:
                visib_frac = 1
            else:
                visib_frac = gt_info_dict[str_im_id][anno_i]["visib_fract"]

            mask_path = str(
                scene_root / "mask" / f"{int(str_im_id):06d}_{int(anno_i):06d}.jpg"
            )
            mask_visib_path = str(
                scene_root
                / "mask_visib"
                / f"{int(str_im_id):06d}_{int(anno_i):06d}.jpg"
            )
            if not self.det and not self.dataset_type == DatasetType.CUSTOM:
                mask_path = correct_suffix(mask_path)
                mask_visib_path = correct_suffix(mask_visib_path)

            if obj_id - 1 not in self.metadata.class_names.keys():
                continue
            if (bbox_visib[2] == 0 or bbox_visib[3] == 0) and self.mode == Mode.TEST:
                print(0)
            if (
                not (
                    # np.any(bbox_obj < 0) or
                    np.any(bbox_visib < 0)
                    or bbox_visib[2] == 0
                    or bbox_visib[3] == 0
                )
                and visib_frac > 0.05
            ) or self.mode == Mode.TEST:
                annotations["obj_id"].append(obj_id)
                annotations["bbox_obj"].append(bbox_obj)
                annotations["bbox_visib"].append(bbox_visib)
                annotations["pose"].append(pose)
                annotations["mask_path"].append(str(mask_path))
                annotations["mask_visib_path"].append(str(mask_visib_path))
            # else:
            # print(bbox_obj)
            # print(bbox_visib)
            # print(visib_frac)
        return annotations

    def _create_scene_dataset(self, scene_root: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Create a dataset dictionary from information in a single scene folder.

        Args:
            scene_root: The path to the root directory of the scene.

        Returns:
            A list of dictionaries containing the following keys:
            - scene_id: The ID of the scene.
            - img_id: The ID of the image in the scene.
            - cam: The camera matrix for the image.
            - depth_factor: The depth factor for the image.
            - img_type: The type of image (e.g. real or synthetic).
            - img_path: The path to the image file.
            - annotation: A dictionary containing annotations for the image.
        """
        scene_id = int(str(scene_root.name))
        img_type = str(scene_root.parent.name)
        gt_dict, gt_info_dict, cam_dict = self._load_dicts(scene_root)

        scene_dataset = []
        pbar = gt_dict
        for str_im_id in pbar:
            K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
            img_path = str(scene_root / "rgb" / "{:06d}.jpg".format(int(str_im_id)))
            if self.dataset_type == DatasetType.CUSTOM:
                img_path = str(scene_root / "rgb" / "{:06d}.png".format(int(str_im_id)))

            if not self.dataset_type == DatasetType.CUSTOM:
                img_path = correct_suffix(img_path)
            annotations = self._create_annotation(
                scene_root=scene_root,
                str_im_id=str_im_id,
                gt_dict=gt_dict,
                gt_info_dict=gt_info_dict,
            )
            if not annotations or len(annotations["obj_id"]) == 0:
                self.faulty_labels.append(img_path)
                continue
            # depth_scale = cam_dict[str_im_id]["depth_scale"]
            # gt_graph_path = (
            #    scene_root / "graphs" / "gt" / f"gt_graph_{int(str_im_id):06d}.npy"
            # )
            # init_graph_path = (
            #    scene_root / "graphs" / "init" / f"init_graph_{int(str_im_id):06d}.npy"
            # )
            raw_img_data = {
                "scene_id": scene_id,
                "img_id": int(str_im_id),
                "cam": K,
                "img_type": img_type,
                "img_path": str(img_path),
                "annotation": annotations,
                # "gt_graph_path": str(gt_graph_path),
                # "init_graph_path": str(init_graph_path),
                # "depth_factor": depth_scale,
            }
            scene_dataset.append(raw_img_data)
        return scene_dataset  # or None

    def _create_dataset(self) -> Dict[str, Any]:
        """
        Generate the dataset dictionary for the current dataset.

        Args:
            root: The path to the root directory of the split of the dataset.

        Returns:
            A dictionary containing the following keys:
            - dataset_name: The name of the dataset.
            - models_info: Information about the models used in the dataset.
            - raw_img_dataset: A list of dictionaries containing information about each image in the dataset
                together with annotations.
        """
        dataset = {
            "dataset_name": self.dataset_name,
            "models_info": self.models_info,
            "single_object": self.single_object,
            "faulty_labels": self.faulty_labels,
            "raw_img_dataset": [],
        }
        if RANK in {-1, 0}:
            LOGGER.info(f"Creating {self.dataset_name} dataset.")
            pbar = tqdm(self.scene_paths, postfix=f"{self.dataset_name}")
        else:
            pbar = self.scene_paths
        for i, scene_path in enumerate(pbar):
            if i > 5 and self.debug and self.mode == Mode.TRAIN:
                continue
            if i > 0 and self.debug and self.mode == Mode.TEST:
                continue
            if i != 0 and i % 2 == 0 and self.ablation and self.mode == Mode.TRAIN:
                continue
            if i > 0 and self.ablation and self.mode == Mode.TEST:
                continue

            scene_dataset = self._create_scene_dataset(scene_path)
            if scene_dataset is None:
                continue
            dataset["raw_img_dataset"].extend(scene_dataset)
        LOGGER.info(
            colorstr(
                "green",
                f"Created {self.dataset_name} dataset. Faulty labels: {len(self.faulty_labels)}",
            )
        )
        return dataset

    @property
    def models_info(self) -> Dict[str, Any]:
        """
        Load the models_info.json file for the current dataset.

        Returns:
            A dictionary containing information about the models used in the dataset.
        """
        models_info_path = self.models_root / "models_info.json"
        if not models_info_path.exists():
            raise FileNotFoundError("models_info.json not found in models folder")
        with open(models_info_path, "r") as f:
            models_info = json.load(f)
        return models_info

    def _load_model_symmetries(self) -> dict[int, np.ndarray]:
        sym_infos = {}
        for model_id in self.model_names:
            model_info = self.models_info[str(int(model_id))]
            if "symmetries_discrete" or "symmetris_continous" in model_info:
                sym_transforms = get_symmetry_transformations(
                    model_info, max_sym_disc_step=0.01
                )
                sym_info = np.array(
                    [sym["R"] for sym in sym_transforms], dtype=np.float32
                )
            else:
                sym_info = None
            sym_infos[int(model_id)] = sym_info
        return sym_infos

    @require_dataset
    def __len__(self):
        return len(self._dataset["raw_img_dataset"])

    @require_dataset
    def get_metadata(self, idx: int) -> Dict[str, Any]:
        return {
            "scene_id": self._dataset["raw_img_dataset"][idx]["scene_id"],
            "img_id": self._dataset["raw_img_dataset"][idx]["img_id"],
        }

    @require_dataset
    def length(self):
        return len(self._dataset["raw_img_dataset"])

    @require_dataset
    def get_paths_of(self, folder_name: str, idx: int) -> str:
        paths = [
            Path(mask_visib_path).parent.parent
            / "misc"
            / folder_name
            / Path(mask_visib_path).stem
            for mask_visib_path in self._dataset["raw_img_dataset"][idx]["annotation"][
                "mask_visib_path"
            ]
        ]
        return paths

    @require_dataset
    def get_img_path(self, idx: int) -> str:
        return self._dataset["raw_img_dataset"][idx]["img_path"]

    @require_dataset
    def get_img_id(self, idx: int) -> int:
        return self._dataset["raw_img_dataset"][idx]["img_id"]

    @require_dataset
    def get_scene_id(self, idx: int) -> int:
        return self._dataset["raw_img_dataset"][idx]["scene_id"]

    @require_dataset
    def get_cam(self, idx: int) -> np.ndarray:
        return self._dataset["raw_img_dataset"][idx]["cam"].copy()

    @require_dataset
    def get_depth_factor(self, idx: int) -> float:
        return self._dataset["raw_img_dataset"][idx]["depth_factor"]

    @require_dataset
    def get_img_type(self, idx: int) -> str:
        return self._dataset["raw_img_dataset"][idx]["img_type"]

    @require_dataset
    def get_obj_ids(self, idx: int) -> int:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["obj_id"]

    @require_dataset
    def get_bbox_objs(self, idx: int) -> List[List[int]]:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["bbox_obj"].copy()

    @require_dataset
    def get_bbox_visibs(self, idx: int) -> np.ndarray:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["bbox_visib"].copy()

    @require_dataset
    def get_poses(self, idx: int) -> np.ndarray:
        return copy.deepcopy(
            self._dataset["raw_img_dataset"][idx]["annotation"]["pose"]
        )

    @require_dataset
    def get_mask_paths(self, idx: int) -> str:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["mask_path"]

    @require_dataset
    def get_mask_visib_paths(self, idx: int) -> str:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["mask_visib_path"]

    #    @require_dataset
    #    def get_graph_paths(self, idx: int) -> Graph:
    #        return self._dataset["raw_img_dataset"][idx]["annotation"]["graph_path"]


#    @require_dataset
#    def get_graph_gt_path(self, idx: int) -> Tuple[np.ndarray]:
#        return self._dataset["raw_img_dataset"][idx]["gt_graph_path"]
#
#    @require_dataset
#    def get_graph_init_path(self, idx: int) -> Tuple[np.ndarray]:
#        return self._dataset["raw_img_dataset"][idx]["init_graph_path"]

# @property
# def binary_model_symmetries(self) -> Dict[int, bool]:
#    b_model_symmetries = {
#        k: False for k, v in self.model_symmetries.items() if v.shape[0] == 1
#    }
#    b_model_symmetries.update(
#        {k: True for k, v in self.model_symmetries.items() if v.shape[0] > 1}
#    )
#    return b_model_symmetries


#    def _load_models(self) -> None:
#        """
#        Load the models for the current dataset.
#        Format:
#        {model_num:int : model:o3d.geometry.TriangleMesh}
#        """
#        self.models = {
#            int(model_num): o3d.io.read_triangle_mesh(
#                str(self.models_root / f"obj_{model_num}.ply")
#            )
#            for model_num in self.model_names
#        }
#        for key, model in self.models.items():
#            if not model.has_vertex_normals():
#                self.models[key] = model.compute_vertex_normals()
#
#    @property
#    def diameters(self) -> np.ndarray:
#        """
#        Calculate the diameters (size) of each model in the dataset.
#
#        Returns:
#            An array containing the diameters of each model, in the format [size_x, size_y, size_z].
#        """
#        cur_diameters = {}
#        for k, v in self.models_info.items():
#            size_x = v["size_x"] / 1000
#            size_y = v["size_y"] / 1000
#            size_z = v["size_z"] / 1000
#            cur_diameters[int(k)] = np.array([size_x, size_y, size_z], dtype="float32")
#        return cur_diameters

#    def generate_gt_graphs(
#        self, num_points: int, img_width: int, img_height: int, site: bool = False
#    ):
#        site = True
#        assert (
#            num_points == self.num_points
#        ), "num_points must be the same as the number of points used to generate the point clouds"
#        for entry in tqdm(
#            self._dataset["raw_img_dataset"],
#            total=len(self._dataset["raw_img_dataset"]),
#        ):
#            img_id = entry["img_id"]
#            cam = entry["cam"]
#            annotation = entry["annotation"]
#            obj_ids = annotation["obj_id"]
#            poses = annotation["pose"]
#            graph_paths = annotation["graph_path"]
#            assert (
#                len(obj_ids) == len(poses) == len(graph_paths)
#            ), "obj_ids, poses, and graph_paths must have the same length"
#
#            graph_path = Path(graph_paths[0]).parent / "gt"  # / "graphs"
#            graph_path.mkdir(parents=True, exist_ok=True)
#            graph_path = graph_path / f"gt_graph_{int(img_id):06d}.npy"
#            # models = {obj_id:self.models[obj_id].sample_points_poisson_disk(num_points) for obj_id in obj_ids}
#            pc_models = {obj_id: self.pc_models[obj_id] for obj_id in obj_ids}
#
#            vertices_combined = np.zeros((num_points * len(obj_ids), 3))
#            normals_combined = np.zeros((num_points * len(obj_ids), 3))
#            for idx, (obj_id, pose) in enumerate(zip(obj_ids, poses)):
#                pc = pc_models[obj_id]
#                vertices = np.asarray(pc.points)
#                normals = np.asarray(pc.normals)
#                pose[:3, 3] = pose[:3, 3] / 1000
#                homogenized_pointcloud = np.hstack(
#                    (vertices / 1000, np.ones((vertices.shape[0], 1)))
#                )
#                transformed_pointcloud = np.dot(
#                    homogenized_pointcloud, pose.T
#                )  # [:,:3]
#                transformed_normals = np.dot(normals, pose[:3, :3].T)
#                if site:
#                    transformed_pointcloud = np.dot(transformed_pointcloud, cam.T)
#                    transformed_normals = np.dot(transformed_normals, cam.T)
#                    transformed_pointcloud[:, :2] /= np.array(
#                        (img_width, img_height)
#                    ).reshape(1, 2)
#                    # transformed_normals /= np.linalg.norm(transformed_normals, axis=1).reshape(-1,1)
#                    transformed_normals[:, :2] /= np.array(
#                        (img_width, img_height)
#                    ).reshape(1, 2)
#                    transformed_normals = transformed_normals / np.linalg.norm(
#                        transformed_normals, axis=1
#                    )
#
#                vertices_combined[
#                    idx * num_points : (idx + 1) * num_points, :
#                ] = transformed_pointcloud
#                normals_combined[
#                    idx * num_points : (idx + 1) * num_points, :
#                ] = transformed_normals
#            np.save(str(graph_path), (vertices_combined, normals_combined))
#
#    def generate_initial_graphs(
#        self, img_width: int, img_height: int, bbox: str = "visib"
#    ):
#        vertices = np.asarray(_ico_verts0, dtype=np.float32) * 0.05
#        edges = np.asarray(_ico_edges0, dtype=np.int32)
#        faces = np.asarray(_ico_faces0, dtype=np.int32)
#        num_vertices = vertices.shape[0]
#
#        for i in tqdm(range(len(self._dataset["raw_img_dataset"]))):
#            graph_paths = self.get_graph_paths(i)  # annotation["graph_path"]
#            graph_path = Path(graph_paths[0]).parent / "init"
#            graph_path.mkdir(parents=True, exist_ok=True)
#            img_id = self._dataset["raw_img_dataset"][i]["img_id"]
#            graph_path = graph_path / f"init_graph_{int(img_id):06d}.npy"
#            bbox_objs = self.get_bbox_objs(i)
#            # could be cx cy h w, byt also x y h w (left top)
#            centers = np.array(
#                [
#                    np.asarray([bbox[0] / img_width, bbox[1] / img_height])
#                    for bbox in bbox_objs
#                ]
#            )
#            initial_features = np.tile(vertices, (centers.shape[0], 1))
#            centers_adj = np.repeat(centers, vertices.shape[0], axis=0)
#            initial_features[:, :2] += centers_adj
#            initial_features[:, 2] += 0.8
#
#            initial_edges = np.tile(edges, (centers.shape[0], 1))
#            initial_edges[:, 0] += np.repeat(
#                np.arange(centers.shape[0]) * num_vertices, edges.shape[0]
#            )
#            initial_edges[:, 1] += np.repeat(
#                np.arange(centers.shape[0]) * num_vertices, edges.shape[0]
#            )
#            initial_faces = np.tile(faces, (centers.shape[0], 1))
#            initial_faces[:, 0] += np.repeat(
#                np.arange(centers.shape[0]) * num_vertices, faces.shape[0]
#            )
#            initial_faces[:, 1] += np.repeat(
#                np.arange(centers.shape[0]) * num_vertices, faces.shape[0]
#            )
#            initial_faces[:, 2] += np.repeat(
#                np.arange(centers.shape[0]) * num_vertices, faces.shape[0]
#            )
#            dtype = [
#                ("initial_features", initial_features.dtype, initial_features.shape),
#                ("initial_faces", initial_faces.dtype, initial_faces.shape),
#                ("initial_edges", initial_edges.dtype, initial_edges.shape),
#            ]
#            structured_array = np.array(
#                [(initial_features, initial_faces, initial_edges)], dtype=dtype
#            )
#            np.save(str(graph_path), structured_array)
#
#    def generate_initial_meshes(
#        self, img_width: int, img_height: int, bbox: str = "visib"
#    ):
#        vertices = np.asarray(_ico_verts0, dtype=np.float32) * 0.05
#        faces = np.asarray(_ico_faces0, dtype=np.int32)
#
#        for i in tqdm(range(len(self._dataset["raw_img_dataset"]))):
#            graph_paths = self.get_graph_paths(i)  # annotation["graph_path"]
#            graph_path = Path(graph_paths[0]).parent / "init"
#            graph_path.mkdir(parents=True, exist_ok=True)
#            img_id = self._dataset["raw_img_dataset"][i]["img_id"]
#            graph_path = graph_path / f"init_graph_{int(img_id):06d}.npy"
#            bbox_objs = self.get_bbox_objs(i)
#            # could be cx cy h w, byt also x y h w (left top)
#            centers = np.array(
#                [
#                    np.asarray([bbox[0] / img_width, bbox[1] / img_height])
#                    for bbox in bbox_objs
#                ]
#            )
#            initial_features = np.repeat(vertices[None, ...], centers.shape[0], axis=0)
#            centers_adj = np.repeat(centers[:, None, :], vertices.shape[0], axis=1)
#            initial_features[..., :2] += centers_adj
#            initial_features[..., 2] += 0.8
#
#            initial_faces = np.repeat(faces[None, ...], centers.shape[0], axis=0)
#            mesh = Meshes(
#                verts=torch.from_numpy(initial_features),
#                faces=torch.from_numpy(initial_faces),
#            )
#            initial_faces = mesh.faces_packed().numpy()
#            initial_edges = mesh.edges_packed().numpy()
#
#            dtype = [
#                (
#                    "initial_features",
#                    initial_features.dtype,
#                    initial_features.shape,
#                ),
#                (
#                    "initial_faces",
#                    initial_faces.dtype,
#                    initial_faces.shape,
#                ),
#                (
#                    "initial_edges",
#                    initial_edges.dtype,
#                    initial_edges.shape,
#                ),
#            ]
#            structured_array = np.array(
#                [(initial_features, initial_faces, initial_edges)], dtype=dtype
#            )
#            np.save(str(graph_path), structured_array)
#
#    def generate_gt_meshes(
#        self, num_points: int, img_width: int, img_height: int, site: bool = False
#    ):
#        # transform full graph, make to mesh, calculate normals, sample, save
#        site = True
#        assert (
#            num_points == self.num_points
#        ), "num_points must be the same as the number of points used to generate the point clouds"
#        for entry in tqdm(
#            self._dataset["raw_img_dataset"],
#            total=len(self._dataset["raw_img_dataset"]),
#        ):
#            img_id = entry["img_id"]
#            cam = entry["cam"]
#            annotation = entry["annotation"]
#            obj_ids = annotation["obj_id"]
#            poses = annotation["pose"]
#            graph_paths = annotation["graph_path"]
#            assert (
#                len(obj_ids) == len(poses) == len(graph_paths)
#            ), "obj_ids, poses, and graph_paths must have the same length"
#
#            graph_path = Path(graph_paths[0]).parent / "gt"  # / "graphs"
#            graph_path.mkdir(parents=True, exist_ok=True)
#            graph_path = graph_path / f"gt_graph_{int(img_id):06d}.npy"
#            meshes = []
#            for idx, (obj_id, pose) in enumerate(zip(obj_ids, poses)):
#                model = self.models[obj_id]
#                vertices = np.asarray(model.vertices)
#                faces = np.asarray(model.triangles)
#                pose[:3, 3] = pose[:3, 3] / 1000
#                homogenized_pointcloud = np.hstack(
#                    (vertices / 1000, np.ones((vertices.shape[0], 1)))
#                )
#                transformed_pointcloud = np.dot(
#                    homogenized_pointcloud, pose.T
#                )  # [:,:3]
#                if site:
#                    transformed_pointcloud = np.dot(transformed_pointcloud, cam.T)
#                    transformed_pointcloud[:, :2] /= np.array(
#                        (img_width, img_height)
#                    ).reshape(1, 2)
#
#                meshes.append(
#                    Meshes(
#                        verts=[torch.from_numpy(transformed_pointcloud).float()],
#                        faces=[torch.from_numpy(faces).long()],
#                    )
#                )
#            meshes_combined = join_meshes_as_batch(meshes)
#            verts_sampled, normals_sampled = sample_points_from_meshes(
#                meshes_combined, num_points, return_normals=True
#            )
#            verts_sampled, normals_sampled = (
#                verts_sampled.numpy(),
#                normals_sampled.numpy(),
#            )
#            dtype = [
#                (
#                    "gt_features",
#                    verts_sampled.dtype,
#                    verts_sampled.shape,
#                ),
#                (
#                    "gt_normals",
#                    normals_sampled.dtype,
#                    normals_sampled.shape,
#                ),
#            ]
#            structured_array = np.array([(verts_sampled, normals_sampled)], dtype=dtype)
#            np.save(str(graph_path), structured_array)
#
#    def generate_gt_meshes_single(
#        self,
#        num_points: List[int],
#        img_width: int,
#        img_height: int,
#        site: bool = True,
#        trans=True,
#    ):
#        #        assert (
#        #            num_points == self.num_points
#        #        ), "num_points must be the same as the number of points used to generate the point clouds"
#        if isinstance(num_points, int):
#            num_points = [num_points]
#
#        for entry in tqdm(
#            self._dataset["raw_img_dataset"],
#            total=len(self._dataset["raw_img_dataset"]),
#        ):
#            img_id = entry["img_id"]
#            cam = entry["cam"]
#            annotation = entry["annotation"]
#            obj_ids = annotation["obj_id"]
#            poses = annotation["pose"]
#            graph_paths = annotation["graph_path"]
#            assert (
#                len(obj_ids) == len(poses) == len(graph_paths)
#            ), "obj_ids, poses, and graph_paths must have the same length"
#            for idx, (obj_id, pose) in enumerate(zip(obj_ids, poses)):
#                graph_path = graph_paths[idx]
#                Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
#                model = self.models[obj_id]
#                vertices = np.asarray(model.vertices)
#                faces = np.asarray(model.triangles)
#                pose[:3, 3] = pose[:3, 3] / 1000
#                """
#                CAREFUL HERE!
#                """
#                pose[:3, :3] = np.eye(3)
#                pose[:3, 3] = np.array([0, 0, 0.8])
#
#                if trans:
#                    homogenized_pointcloud = np.hstack(
#                        (vertices / 1000, np.ones((vertices.shape[0], 1)))
#                    )
#                    transformed_pointcloud = np.dot(homogenized_pointcloud, pose.T)
#                else:
#                    # only rotation
#                    transformed_pointcloud = np.dot(vertices / 1000, pose[:3, :3].T)
#                    # shift center to (0,0,0.8)
#                    # transformed_pointcloud += np.array([0, 0, 0.8])
#                if site:
#                    transformed_pointcloud = np.dot(transformed_pointcloud, cam.T)
#                    transformed_pointcloud[:, :2] /= np.array(
#                        (img_width, img_height)
#                    ).reshape(1, 2)
#
#                meshes = Meshes(
#                    verts=[torch.from_numpy(transformed_pointcloud).float()],
#                    faces=[torch.from_numpy(faces).long()],
#                )
#                data = {}
#                for num_point in num_points:
#                    verts_sampled, normals_sampled = sample_points_from_meshes(
#                        meshes, num_point, return_normals=True
#                    )
#                    verts_sampled, normals_sampled = (
#                        verts_sampled.squeeze().numpy(),
#                        normals_sampled.squeeze().numpy(),
#                    )
#                    data[f"gt_features_{num_point}"] = verts_sampled
#                    data[f"gt_normals_{num_point}"] = normals_sampled
#                with open(graph_path, "wb") as f:
#                    pickle.dump(data, f)
#                #                dtype = [
#                #                    (
#                #                        "gt_features",
#                #                        verts_sampled.dtype,
#                #                        verts_sampled.shape,
#                #                    ),
#                #                    (
#                #                        "gt_normals",
#                #                        normals_sampled.dtype,
#                #                        normals_sampled.shape,
#                #                    ),
#                #                ]
#                #                structured_array = np.array(
#                #                    [(verts_sampled, normals_sampled)], dtype=dtype
#                #                )
#                # dt = np.dtype([("data1", object), ("data2", object)])
#
#                # structured_array = np.array([data], dtype=dtype)
#                # np.save(str(graph_path), structured_array)
