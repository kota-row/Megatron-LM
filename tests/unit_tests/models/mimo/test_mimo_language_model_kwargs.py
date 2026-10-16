# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Extra language arguments remain explicit across MIMO routes and autograd."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from megatron.core.models.mimo.config.role import ModuleLayout
from megatron.core.models.mimo.model.base import MimoModel


class _Language(nn.Module):
    def forward(self, decoder_input, auxiliary=None, scale=1, **kwargs):
        output = decoder_input.sum()
        if auxiliary is not None:
            output = output + scale * auxiliary.square().sum()
        return output


def _model():
    model = MimoModel.__new__(MimoModel)
    nn.Module.__init__(model)
    model.language_model = _Language()
    model.special_token_ids = {}
    model.language_model_input_projections = nn.ModuleDict()
    model.partition_adapter = SimpleNamespace(cfg=SimpleNamespace(use_cp=False))
    model.colocated_comms = {}
    model.modality_submodules = nn.ModuleDict()
    batch = {
        'tokens': None,
        'position_ids': None,
        'labels': None,
        'loss_mask': None,
        'mtp_input_mask': None,
        'decoder_input': torch.ones(2, 1, 4),
    }
    cp_batch = SimpleNamespace(
        get_batch=lambda: batch,
        get_packed_seq_params=lambda: None,
        batches_by_layout={'default': batch},
    )
    model._prepare_language_inputs = MagicMock(return_value=cp_batch)
    model.get_text_embeddings = MagicMock(return_value=torch.ones(2, 4))
    model.align_embeddings_by_token_positions = MagicMock(return_value=batch['decoder_input'])
    return model, cp_batch


@pytest.mark.parametrize('route', ['first_language', 'later_language', 'colocated'])
def test_auxiliary_gradients_across_microbatches(route):
    model, _ = _model()
    model.role = SimpleNamespace(
        is_first_stage=lambda _: route == 'first_language', is_last_stage=lambda _: True
    )
    ids = torch.ones(1, 2, dtype=torch.long)
    first = torch.tensor([2.0], requires_grad=True)
    second = torch.tensor([4.0], requires_grad=True)

    def forward(auxiliary, scale):
        extra = {'auxiliary': auxiliary, 'scale': scale}
        if route == 'colocated':
            result, _ = model._forward_all_modules(
                ids, None, None, None, None, None, language_model_kwargs=extra
            )
        else:
            result, _ = model._forward_language_module(
                ids, None, None, None, None, {}, language_model_kwargs=extra
            )
        assert extra['auxiliary'] is auxiliary
        assert set(extra) == {'auxiliary', 'scale'}
        return result

    first_loss = forward(first, 3)
    second_loss = forward(second, 5)
    second_loss.backward()
    first_loss.backward()
    torch.testing.assert_close(first.grad, torch.tensor([12.0]))
    torch.testing.assert_close(second.grad, torch.tensor([40.0]))


@pytest.mark.parametrize(
    'key',
    [
        'input_ids',
        'position_ids',
        'attention_mask',
        'decoder_input',
        'labels',
        'loss_mask',
        'mtp_input_mask',
        'packed_seq_params',
        'cp_batch',
    ],
)
def test_managed_arguments_cannot_be_overridden(key):
    model, cp_batch = _model()
    with pytest.raises(ValueError, match='MIMO-managed'):
        model._forward_prepared_language_inputs(cp_batch, None, {key: None})


def test_default_arguments_preserve_language_output():
    model, cp_batch = _model()
    torch.testing.assert_close(
        model._forward_prepared_language_inputs(cp_batch, None), torch.tensor(8.0)
    )


def test_auxiliary_features_survive_activation_recompute():
    model, cp_batch = _model()
    auxiliary = torch.tensor([3.0], requires_grad=True)
    output = checkpoint(
        lambda value: model._forward_prepared_language_inputs(cp_batch, None, {'auxiliary': value}),
        auxiliary,
        use_reentrant=False,
    )
    output.backward()
    torch.testing.assert_close(auxiliary.grad, torch.tensor([6.0]))


@pytest.mark.parametrize('route', ['language', 'colocated', 'encoder'])
def test_public_forward_routes_extra_arguments(route):
    model, _ = _model()
    model.role = SimpleNamespace(
        mode=ModuleLayout.COLOCATED if route == 'colocated' else ModuleLayout.NON_COLOCATED,
        has_modality_modules=route == 'encoder',
        has_language_module=route == 'language',
    )
    model._forward_language_module = MagicMock(return_value=(None, None))
    model._forward_all_modules = MagicMock(return_value=(None, None))
    model._forward_encoders = MagicMock(return_value={})
    extra = {'auxiliary': torch.ones(1)}
    model(torch.ones(1, 2, dtype=torch.long), language_model_kwargs=extra)
    if route == 'encoder':
        assert 'language_model_kwargs' not in model._forward_encoders.call_args.kwargs
    else:
        target = (
            model._forward_all_modules if route == 'colocated' else model._forward_language_module
        )
        assert target.call_args.kwargs['language_model_kwargs'] is extra
