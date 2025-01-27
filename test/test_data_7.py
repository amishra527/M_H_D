import os, cv2, yaml, random, numpy as np
from PIL import Image
from glob import glob
from matplotlib import pyplot as plt
from torchvision import transforms as T
from ultralytics import YOLO


class Visualization:
    def __init__(self, root, data_types, n_ims, rows, cmap=None):
        self.n_ims = n_ims
        self.rows = rows
        self.cmap = cmap
        self.data_types = data_types
        self.colors = ["firebrick", "darkorange", "blueviolet"]

        # Convert backslashes to forward slashes for consistency
        self.root = os.path.normpath(root)
        print(f"Initialized with root path: {self.root}")
        self.get_cls_names()
        self.get_bboxes()

    def get_cls_names(self):
        yaml_path = os.path.join(self.root, "KIIT-MiTA.yml")
        print(f"Reading YAML from: {yaml_path}")

        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        # Extract class names
        self.class_dict = {index: name for index, name in enumerate(data["names"])}
        print(f"Loaded class dictionary: {self.class_dict}")

    def get_bboxes(self):
        self.vis_datas = {}
        self.analysis_datas = {}
        self.im_paths = {}

        for data_type in self.data_types:
            print(f"\nProcessing {data_type} dataset")
            all_bboxes = []
            all_analysis_datas = {}

            image_dir = os.path.join(self.root, data_type, "images")
            print(f"Looking for images in: {image_dir}")

            im_paths = glob(os.path.join(image_dir, "*"))
            print(f"Found {len(im_paths)} images")

            for idx, im_path in enumerate(im_paths):
                # Get corresponding label path
                base_name = os.path.basename(im_path)
                name_without_ext = os.path.splitext(base_name)[0]
                lbl_path = os.path.join(
                    self.root, data_type, "labels", f"{name_without_ext}.txt"
                )

                if not os.path.isfile(lbl_path):
                    print(f"Warning: No label file for {im_path}")
                    continue

                bboxes = []
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # Ensure we have class + 4 bbox coordinates
                            try:
                                class_idx = int(parts[0])
                                cls_name = self.class_dict[class_idx]
                                bbox = [cls_name] + [float(x) for x in parts[1:5]]
                                bboxes.append(bbox)

                                # Update analysis data
                                if cls_name not in all_analysis_datas:
                                    all_analysis_datas[cls_name] = 1
                                else:
                                    all_analysis_datas[cls_name] += 1

                            except (ValueError, IndexError, KeyError) as e:
                                print(f"Error processing line in {lbl_path}: {e}")
                                continue

                all_bboxes.append(bboxes)

            print(f"Processed {len(all_bboxes)} valid image-label pairs")
            print(f"Class distribution: {all_analysis_datas}")

            self.vis_datas[data_type] = all_bboxes
            self.analysis_datas[data_type] = all_analysis_datas
            self.im_paths[data_type] = im_paths

    def plot(self, rows, cols, count, im_path, bboxes):
        plt.subplot(rows, cols, count)

        try:
            or_im = np.array(Image.open(im_path).convert("RGB"))
            height, width, _ = or_im.shape

            for bbox in bboxes:
                class_id, x_center, y_center, w, h = bbox

                # Convert YOLO format to pixel values
                x_min = int((x_center - w / 2) * width)
                y_min = int((y_center - h / 2) * height)
                x_max = int((x_center + w / 2) * width)
                y_max = int((y_center + h / 2) * height)

                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                cv2.rectangle(
                    img=or_im,
                    pt1=(x_min, y_min),
                    pt2=(x_max, y_max),
                    color=color,
                    thickness=3,
                )

            plt.imshow(or_im)
            plt.axis("off")
            plt.title(f"There is (are) {len(bboxes)} object(s) in the image.")

        except Exception as e:
            print(f"Error plotting image {im_path}: {e}")

        return count + 1

    def vis(self, save_name):
        print(f"\n{save_name.upper()} Data Visualization is in process...")

        if not self.vis_datas.get(save_name):
            print(f"No data found for {save_name}")
            return

        cols = max(1, self.n_ims // self.rows)
        count = 1

        plt.figure(figsize=(25, 20))

        data_len = len(self.vis_datas[save_name])
        indices = [
            random.randint(0, data_len - 1) for _ in range(min(self.n_ims, data_len))
        ]

        for idx, index in enumerate(indices):
            if count == self.n_ims + 1:
                break

            im_path = self.im_paths[save_name][index]
            bboxes = self.vis_datas[save_name][index]

            count = self.plot(self.rows, cols, count, im_path=im_path, bboxes=bboxes)

        plt.tight_layout()
        plt.show()

    def data_analysis(self, save_name, color):
        print(f"\nData analysis for {save_name}...")

        if not self.analysis_datas.get(save_name):
            print(f"No analysis data for {save_name}")
            return

        cls_names = list(self.analysis_datas[save_name].keys())
        counts = list(self.analysis_datas[save_name].values())

        if not cls_names or not counts:
            print(f"No classes or counts found for {save_name}")
            return

        fig, ax = plt.subplots(figsize=(30, 10))
        width = 0.7
        indices = np.arange(len(counts))

        ax.bar(indices, counts, width, color=color)
        ax.set_xlabel("Class Names", color="black", fontsize=12)
        ax.set_ylabel("Data Counts", color="black", fontsize=12)
        ax.set_title(
            f"{save_name.upper()} Dataset Class Imbalance Analysis", fontsize=14
        )

        ax.set_xticks(indices)
        ax.set_xticklabels(cls_names, rotation=45, ha="right")

        # Add value labels on top of bars
        for i, v in enumerate(counts):
            ax.text(i, v + max(counts) * 0.01, str(v), ha="center", va="bottom")

        plt.tight_layout()
        plt.show()

    def visualization(self):
        for save_name in self.data_types:
            self.vis(save_name)

    def analysis(self):
        for save_name, color in zip(self.data_types, self.colors):
            self.data_analysis(save_name, color)


# Usage
root = "KIIT-MiTA"  # Use forward slashes
vis = Visualization(
    root=root, data_types=["train", "valid", "test"], n_ims=20, rows=5, cmap="rgb"
)
print(vis.analysis())

print(vis.visualization())