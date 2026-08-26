import argparse
import importlib.util
import os
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd


DEFAULT_YOLO_MODEL = "yolo11l-pose.pt"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_METRABS_MODEL_NAME = "metrabs_eff2l_y4_384px_800k_28ds"
DEFAULT_METRABS_MODEL_URL = "https://bit.ly/metrabs_l"
DEFAULT_METRABS_MODEL_PATH = os.environ.get("METRABS_MODEL_PATH")
DEFAULT_METRABS_SKELETON = "coco_19"
DEFAULT_METRABS_DEFAULT_FOV_DEGREES = 55
METRABS_LEG_JOINT_CANDIDATES = {
    "left_knee": ("lkne", "left_knee", "left knee", "l_knee"),
    "right_knee": ("rkne", "right_knee", "right knee", "r_knee"),
    "left_foot": ("lank", "left_ankle", "left ankle", "l_ankle", "lfoot", "left_foot"),
    "right_foot": ("rank", "right_ankle", "right ankle", "r_ankle", "rfoot", "right_foot"),
}


def resolve_yolo_model():
    model_name = DEFAULT_YOLO_MODEL
    script_path = Path(__file__).resolve().parent / model_name
    if script_path.exists():
        return str(script_path)

    cwd_path = Path.cwd() / model_name
    if cwd_path.exists():
        return str(cwd_path)

    try:
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = attempt_download_asset(model_name)
        if downloaded:
            return str(downloaded)
    except Exception:
        pass


    return model_name


def resolve_metrabs_model():
    for base_path in (Path(__file__).resolve().parent, Path.cwd()):
        exact_path = base_path / DEFAULT_METRABS_MODEL_NAME
        if exact_path.exists():
            return str(exact_path)

    return DEFAULT_METRABS_MODEL_URL


def normalize_leg_name(leg):
    if leg is None:
        raise ValueError("Leg must be provided with --leg2track Left or --leg2track Right.")
    leg = leg.lower()
    if leg in ("left", "right"):
        return leg
    raise ValueError("Leg must be Left or Right.")


def output_stem(video_path):
    normalized_path = Path(video_path)
    base_name = normalized_path.stem
    subfolder_name = normalized_path.parent.name
    return f"{subfolder_name}_{base_name}" if subfolder_name else base_name


def is_remote_model_path(path):
    return urlparse(str(path)).scheme in ("http", "https", "gs")


def normalize_joint_name(name):
    if isinstance(name, bytes):
        name = name.decode("utf8")
    return str(name).lower().replace("-", "_").replace(" ", "_")


class la_video_analysis:
    def __init__(self, config, leg=None):
        self.config = config
        self.requested_leg = normalize_leg_name(leg if leg is not None else config.get("leg2track"))
        self.backend = config["backend"].lower()
        self.fps = config.get("fallback_fps", 30.0)
        self.leg_joint_indices = None
        self._setup_backend()

    def _setup_backend(self):
        self.yolo_model = None
        self.metrabs_model = None
        self.metrabs_skeleton = None
        self.metrabs_default_fov = None
        self.metrabs_edges = None

        if self.backend == "yolo":
            from ultralytics import YOLO

            print(f"Loading YOLO model: {self.config['yolo_model_path']}")
            self.yolo_model = YOLO(self.config["yolo_model_path"])
        elif self.backend == "metrabs":
            model_path = self.config.get("metrabs_model_path")
            if not model_path:
                raise ValueError(
                    "Metrabs backend needs a model path. Pass --metrabs_model_path, set "
                    "METRABS_MODEL_PATH, place metrabs_eff2l_y4_384px_800k_28ds next to "
                    "this script, or allow the TFHub URL fallback."
                )
            if not is_remote_model_path(model_path) and not Path(model_path).expanduser().exists():
                raise FileNotFoundError(f"Metrabs model path does not exist: {model_path}")

            print(f"Loading Metrabs model: {model_path}")
            try:
                import tensorflow_hub as tfhub
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "tensorflow-hub is required for the Metrabs backend. Install the "
                    "repo's inference environment or run `pip install tensorflow tensorflow-hub`."
                ) from exc

            self.metrabs_skeleton = self.config.get("metrabs_skeleton", "coco_19")
            self.metrabs_default_fov = self.config.get("metrabs_default_fov_degrees", 55)
            resolved_model_path = (
                model_path if is_remote_model_path(model_path) else str(Path(model_path).expanduser())
            )
            self.metrabs_model = tfhub.load(resolved_model_path)
            self.metrabs_edges = self._get_metrabs_skeleton_tensor("per_skeleton_joint_edges").numpy()
            self.metrabs_joint_names = self._get_metrabs_skeleton_tensor(
                "per_skeleton_joint_names"
            ).numpy()
            self.leg_joint_indices = self._resolve_metrabs_leg_joint_indices()
            print(
                "Metrabs leg joints: "
                + ", ".join(f"{name}={idx}" for name, idx in self.leg_joint_indices.items())
            )
        else:
            raise ValueError("Unsupported backend. Use yolo or metrabs.")

    def _get_metrabs_skeleton_tensor(self, attr_name):
        skeleton_map = getattr(self.metrabs_model, attr_name)
        try:
            return skeleton_map[self.metrabs_skeleton]
        except KeyError as exc:
            try:
                available = ", ".join(str(key) for key in skeleton_map.keys())
            except AttributeError:
                available = "unknown"
            raise ValueError(
                f"Metrabs skeleton '{self.metrabs_skeleton}' is not available. "
                f"Available skeletons: {available}"
            ) from exc

    def _resolve_metrabs_leg_joint_indices(self):
        normalized_to_index = {
            normalize_joint_name(name): idx for idx, name in enumerate(self.metrabs_joint_names)
        }
        resolved = {}
        for required_name, candidates in METRABS_LEG_JOINT_CANDIDATES.items():
            for candidate in candidates:
                normalized_candidate = normalize_joint_name(candidate)
                if normalized_candidate in normalized_to_index:
                    resolved[required_name] = normalized_to_index[normalized_candidate]
                    break
            else:
                joint_names = ", ".join(normalize_joint_name(name) for name in self.metrabs_joint_names)
                raise RuntimeError(
                    f"Could not find required Metrabs joint '{required_name}' in skeleton "
                    f"'{self.metrabs_skeleton}'. Model joints: {joint_names}"
                )
        return resolved

    def _extract_features(self, distances):
        current_dir = Path(__file__).resolve().parent
        feature_extraction_path = current_dir.parent / "feature extraction" / "feature_extraction.py"

        spec = importlib.util.spec_from_file_location(
            "feature_extraction_module", feature_extraction_path
        )
        feature_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_module)

        extractor = feature_module.feature_ext_analysis({"test_type": "la"})
        feature_values, feature_names = extractor._extract_features(np.asarray(distances), self.fps)
        feature_columns = feature_names[6:]

        df = pd.DataFrame([feature_values], columns=feature_columns)
        feature_path = self.SAVE_DIR / "features.csv"
        df.to_csv(feature_path, index=False)
        print(f"Saved features to: {feature_path}")
        return feature_path

    def preprocess_and_display_video(self):
        raw_video_path = self.config["video_path"]
        video_path = str(raw_video_path).replace("/data", "//chansey.umcn.nl")
        frame_stride = self.config.get("frame_stride", 1)
        progress_interval = self.config.get("progress_interval", 30)
        leg_to_track = self.requested_leg
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            self.fps = self.config.get("fallback_fps", 30.0)

        print(f"Opened video: {video_path}")
        print(f"Video size: {width}x{height}, fps: {self.fps:.2f}, frames: {total_frames or 'unknown'}")
        print(f"Tracking leg: {leg_to_track}")

        self.SAVE_DIR = Path(self.config.get("save_path", DEFAULT_RESULTS_DIR)).expanduser()
        self.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(raw_video_path).suffix or ".mp4"
        self.new_filename = f"{output_stem(raw_video_path)}{suffix}"
        visualization_path = self.SAVE_DIR / self.new_filename
        print(f"Writing keypoint overlay video to: {visualization_path}")

        output_fps = max(self.fps / frame_stride, 1.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(visualization_path), fourcc, output_fps, (width, height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not create visualization video: {visualization_path}")

        sequences = []
        frame_num = 0
        all_keypoints = []
        norm_dist = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                if (frame_num - 1) % frame_stride != 0:
                    continue

                if progress_interval and frame_num == 1:
                    print("Running first pose inference. This can take a moment...")

                kpts, annotated_frame = self._predict_keypoints(frame, frame_num)
                if video_writer is not None:
                    video_writer.write(annotated_frame)
                cv2.imshow("Annotated Frame", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if kpts is None or not self._has_required_leg_keypoints(kpts):
                    continue

                left_knee_y, right_knee_y, left_foot_y, right_foot_y = self._get_leg_y_values(kpts)

                if norm_dist is None:
                    if leg_to_track == "right":
                        norm_dist = abs(left_foot_y - left_knee_y)
                    else:
                        norm_dist = abs(right_foot_y - right_knee_y)
                    if norm_dist == 0:
                        continue

                if leg_to_track == "right":
                    foot_dist_y = -(right_foot_y - left_foot_y)
                else:
                    foot_dist_y = -(left_foot_y - right_foot_y)

                sequences.append(foot_dist_y / norm_dist)
                all_keypoints.append(kpts)

                if progress_interval and frame_num % progress_interval == 0:
                    if total_frames > 0:
                        print(f"Processed frame {frame_num}/{total_frames}; signal points: {len(sequences)}")
                    else:
                        print(f"Processed frame {frame_num}; signal points: {len(sequences)}")
        finally:
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()

        distances = np.array(sequences)
        if len(distances) == 0:
            raise RuntimeError("No valid pose detections were found in the video.")

        self.validate_sample_keypoints(distances, all_keypoints, video_path)
        self._extract_features(distances)

        print(f"Saved visualization to: {visualization_path}")
        print(f"Detected frames: {len(distances)}")
        print(f"Backend: {self.backend}")
        print(f"Leg: {leg_to_track}")

    def _predict_keypoints(self, frame_bgr, frame_num):
        if self.backend == "yolo":
            results = self.yolo_model(frame_bgr, verbose=False)
            annotated_frame = results[0].plot()
            if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
                return None, annotated_frame
            return results[0].keypoints.xy[0].cpu().numpy(), annotated_frame

        if self.backend == "metrabs":
            import tensorflow as tf

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pred = self.metrabs_model.detect_poses(
                tf.convert_to_tensor(frame_rgb, dtype=tf.uint8),
                default_fov_degrees=self.metrabs_default_fov,
                skeleton=self.metrabs_skeleton,
                max_detections=1,
            )
            boxes = pred["boxes"].numpy()
            poses2d = pred["poses2d"].numpy()
            if len(poses2d) == 0:
                return None, frame_bgr
            kpts = poses2d[0]
            return kpts, self._draw_metrabs_overlay(frame_bgr, boxes, kpts)

        raise ValueError(f"Unsupported backend: {self.backend}")

    def _draw_metrabs_overlay(self, frame_bgr, boxes, kpts):
        annotated = frame_bgr.copy()
        if boxes is not None and len(boxes) > 0:
            x, y, w, h, _conf = boxes[0]
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(annotated, pt1, pt2, (0, 255, 0), 2)

        for i_start, i_end in self.metrabs_edges:
            if i_start >= len(kpts) or i_end >= len(kpts):
                continue
            p1 = kpts[i_start]
            p2 = kpts[i_end]
            if not np.isfinite(p1).all() or not np.isfinite(p2).all():
                continue
            cv2.line(annotated, tuple(np.int32(p1)), tuple(np.int32(p2)), (0, 255, 255), 2)

        for point in kpts:
            if np.isfinite(point).all():
                cv2.circle(annotated, tuple(np.int32(point)), 2, (255, 0, 0), -1)

        return annotated

    def _has_required_leg_keypoints(self, kpts):
        if self.backend == "metrabs":
            if self.leg_joint_indices is None:
                return False
            kpts = np.asarray(kpts)
            if kpts.ndim != 2 or kpts.shape[1] < 2:
                return False
            required_indices = self.leg_joint_indices.values()
            return all(
                idx < len(kpts) and np.isfinite(kpts[idx, :2]).all()
                for idx in required_indices
            )
        return len(kpts) > 16

    def _get_leg_y_values(self, kpts):
        if self.backend == "metrabs":
            left_knee_idx = self.leg_joint_indices["left_knee"]
            right_knee_idx = self.leg_joint_indices["right_knee"]
            left_foot_idx = self.leg_joint_indices["left_foot"]
            right_foot_idx = self.leg_joint_indices["right_foot"]
        else:
            left_knee_idx, right_knee_idx = 13, 14
            left_foot_idx, right_foot_idx = 15, 16

        return (
            kpts[left_knee_idx][1],
            kpts[right_knee_idx][1],
            kpts[left_foot_idx][1],
            kpts[right_foot_idx][1],
        )

    def validate_sample_keypoints(self, distances, keypoints, video_path):
        if len(distances) != len(keypoints):
            raise RuntimeError(
                f"Signal/keypoint length mismatch for {video_path}: "
                f"{len(distances)} distance values vs {len(keypoints)} keypoint frames"
            )

        if len(keypoints) == 0:
            raise RuntimeError(f"No keypoints saved for {video_path}")

        if self.backend == "metrabs":
            min_keypoints = max(self.leg_joint_indices.values()) + 1
        else:
            min_keypoints = 17
        for frame_idx, frame_keypoints in enumerate(keypoints):
            frame_keypoints = np.asarray(frame_keypoints)
            if frame_keypoints.ndim != 2 or frame_keypoints.shape[0] < min_keypoints or frame_keypoints.shape[1] < 2:
                raise RuntimeError(
                    f"Invalid keypoint shape for {video_path} at detected frame {frame_idx}: "
                    f"{frame_keypoints.shape}"
                )


def build_config(args):
    save_dir = Path(args.save_dir).expanduser()
    config = {
        "video_path": args.video_path,
        "leg2track": args.leg2track,
        "backend": args.backend,
        "save_path": str(save_dir),
        "fallback_fps": 30.0,
    }

    if args.backend == "yolo":
        config["yolo_model_path"] = resolve_yolo_model()
    elif args.backend == "metrabs":
        config["metrabs_model_path"] = (
            args.metrabs_model_path or DEFAULT_METRABS_MODEL_PATH or resolve_metrabs_model()
        )
        config["metrabs_skeleton"] = args.metrabs_skeleton
        config["metrabs_default_fov_degrees"] = args.metrabs_default_fov_degrees
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    return config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Leg Agility Video Analysis")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--leg2track",
        type=str,
        choices=["Left", "Right"],
        required=True,
        help="Leg to track",
    )
    parser.add_argument(
        "--backend",
        choices=["yolo", "metrabs"],
        default="yolo",
        help="Pose-estimation backend. Default: yolo.",
    )
    parser.add_argument(
        "--save_dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Directory for output files. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--metrabs_model_path",
        default=None,
        help=(
            "Local Metrabs TensorFlow SavedModel directory or TensorFlow Hub URL. "
            "If omitted, uses METRABS_MODEL_PATH, then a local "
            "metrabs_eff2l_y4_384px_800k_28ds folder, then https://bit.ly/metrabs_l."
        ),
    )
    parser.add_argument(
        "--metrabs_skeleton",
        default=DEFAULT_METRABS_SKELETON,
        help=f"Metrabs skeleton name. Default: {DEFAULT_METRABS_SKELETON}.",
    )
    parser.add_argument(
        "--metrabs_default_fov_degrees",
        type=float,
        default=DEFAULT_METRABS_DEFAULT_FOV_DEGREES,
        help=f"Metrabs default camera FOV. Default: {DEFAULT_METRABS_DEFAULT_FOV_DEGREES}.",
    )
    args = parser.parse_args()

    leg_to_track = normalize_leg_name(args.leg2track)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    config["leg2track"] = leg_to_track

    analyzer = la_video_analysis(config)
    analyzer.preprocess_and_display_video()
