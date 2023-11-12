from abc import ABC, abstractproperty
from functools import cached_property, reduce
from pathlib import Path

from matplotlib import pyplot as plt
from torchvision.io.image import read_image
import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)


def compare_images(image1, image2):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))  # 创建一行两列的子图

    axes[0].imshow(image1, cmap="gray")
    axes[0].set_title("Image 1")
    axes[1].imshow(image2, cmap="gray")
    axes[1].set_title("Image 2")

    for ax in axes:
        ax.axis("off")

    plt.show()


class PretrainedModelHandler(ABC):
    WEIGHTS = None

    @property
    def preprocess(self):
        return self.WEIGHTS.transforms()

    @property
    def categories(self):
        return self.WEIGHTS.meta["categories"]

    @property
    def class_to_idx(self):
        return {cls: idx for (idx, cls) in enumerate(self.categories)}

    @abstractproperty
    @cached_property
    def model(self) -> torch.nn.Module:
        pass

    def predict(
        self, imagePathList: list[Path], categoryList: list[str] = None
    ) -> list[torch.Tensor]:
        if not categoryList:
            # NOTE: categoryList默认使用模型全部的categories
            categoryList = self.categories
        else:
            for category in categoryList:
                if category not in self.categories:
                    raise ValueError("Invalid category")

        images = [read_image(imagePath) for imagePath in imagePathList]

        batch = self.preprocess(torch.stack(images, axis=0))
        prediction = self.model(batch)["out"]
        normalized_masks = prediction.softmax(dim=1)

        masks = [
            reduce(
                lambda x, y: torch.clamp(x + y, max=1),
                [
                    normalized_masks[ind, self.class_to_idx[category]]
                    for category in categoryList
                ],
            ).to(torch.float32)
            for ind in range(len(images))
        ]
        return masks


class DeepLabV3ResNet101ModelHandler(PretrainedModelHandler):
    WEIGHTS = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1

    @cached_property
    def model(self):
        return deeplabv3_resnet101(weights=self.WEIGHTS)


if __name__ == "__main__":
    from torchvision.transforms.functional import to_pil_image

    handler = DeepLabV3ResNet101ModelHandler()
    print(handler.categories)
    mask1, mask2 = handler.predict(
        imagePathList=["./images/汽车1.jpg", "./images/汽车2.jpg"],
        categoryList=["car", "bus", "bicycle"],
    )
    to_pil_image(mask1).show()
    to_pil_image(mask2).show()
