# modules/lane_detection/preprocess.py

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

class TuSimpleDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir    = root_dir
        self.split       = split
        self.input_h     = config.LANE_INPUT_HEIGHT
        self.input_w     = config.LANE_INPUT_WIDTH
        self.samples     = []
        self._load_annotations()

    def _load_annotations(self):
        # TuSimple has multiple json label files
        label_files = [
            "label_data_0313.json",
            "label_data_0531.json",
            "label_data_0601.json"
        ]

        for label_file in label_files:
            path = os.path.join(self.root_dir, label_file)
            if not os.path.exists(path):
                continue

            with open(path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.samples.append(data)

    def _make_mask(self, lanes, h_samples, orig_h, orig_w):
        # create blank mask same size as original image
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for lane in lanes:
            # pair each x coordinate with its y coordinate
            points = [
                (x, y) for x, y in zip(lane, h_samples) if x != -2
            ]

            # draw lane as thick line on mask
            for i in range(len(points) - 1):
                cv2.line(mask, points[i], points[i+1], 255, thickness=5)

        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        img_path = os.path.join(self.root_dir, sample["raw_file"])
        lanes    = sample["lanes"]
        h_samples = sample["h_samples"]

        # load image
        image = cv2.imread(img_path)
        orig_h, orig_w = image.shape[:2]

        # create ground truth mask from lane annotations
        mask = self._make_mask(lanes, h_samples, orig_h, orig_w)

        # resize both to U-Net input size
        image = cv2.resize(image, (self.input_w, self.input_h))
        mask  = cv2.resize(mask,  (self.input_w, self.input_h))

        # normalize image to [0, 1] and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        # mask to binary tensor
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # add channel dim

        return image, mask


def get_dataloader(root_dir, batch_size=8, split="train"):
    dataset = TuSimpleDataset(root_dir, split)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    return loader