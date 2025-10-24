from typing import Any

import torch


class collate_images_fn:
    def __init__(self, use_condition: bool):
        self.use_condition = use_condition

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images, conditions = [], [] if self.use_condition else None
        for item in batch:
            images.append(item['x'])

            if self.use_condition:
                conditions.append(item['c'])

        x = torch.stack(images)
        c = torch.tensor(conditions, dtype=torch.long) if self.use_condition else None

        result_batch = {'x': x}
        if self.use_condition:
            result_batch.update({'c': c})
        return result_batch
