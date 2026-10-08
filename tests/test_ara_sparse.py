import unittest
from types import SimpleNamespace
from typing import cast

import torch

from heretic.config import Settings as HereticSettings
from heretic.modifiers.ara import ARA, Parameters, Settings, ara_loss
from heretic.plugin import Context


class SparseARATests(unittest.TestCase):
    def test_missing_distribution_leaves_module_untouched(self):
        modifier = ARA(HereticSettings.model_construct(seed=42), Settings())
        modifier.good_module_io = [
            {"mlp.down_proj": {2: (torch.ones(1, 2), torch.ones(1, 2))}}
        ]
        modifier.bad_module_io = [{}]
        # Objects without weights ensure the guard runs before adapter access.
        model = SimpleNamespace(
            get_layer_modules=lambda index: {
                "mlp.down_proj": [object(), object(), object()]
            }
        )
        ctx = cast(Context, SimpleNamespace(get_model=lambda: model))
        modifier.modify_model(ctx, Parameters(0, 1, 1.0, 1.0, 0.5, 15))

    def test_sparse_samples_cap_neighbors_and_keep_gradients_finite(self):
        good = torch.tensor([[1.0, 2.0]])
        bad = torch.tensor([[3.0, 4.0], [2.0, 3.0]])
        changed = bad.clone().requires_grad_()
        loss = ara_loss(good, bad, good, changed, Parameters(0, 1, 1.0, 1.0, 0.5, 15))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(changed.grad).all())


if __name__ == "__main__":
    unittest.main()
