"""Offline preservation loss must not supervise follow or padded options."""
import pytest
import torch

from eidolon_laya_train.train import distillation_loss


def test_selected_margins_padding_and_detached_teacher():
    student = torch.tensor([[3., 1., -torch.inf], [4., 0., 2.]], requires_grad=True)
    teacher = torch.tensor([[1., 1., -torch.inf], [0., 3., 1.]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    selected = torch.tensor([True, False])
    value = distillation_loss(student, teacher, mask, selected)
    assert value.item() == pytest.approx(1.)
    value.backward()
    assert torch.equal(student.grad, torch.tensor([[1., -1., 0.], [0., 0., 0.]]))
    assert teacher.grad is None
    shifted = student.detach() + 100
    assert distillation_loss(shifted, teacher, mask, selected).item() == pytest.approx(1.)


def test_no_selected_rows_is_differentiable_zero():
    student = torch.tensor([[2., -torch.inf]], requires_grad=True)
    value = distillation_loss(student, student.detach(), torch.tensor([[True, False]]),
                              torch.tensor([False]))
    assert value.item() == 0
    value.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_incompatible_shapes_rejected():
    with pytest.raises(ValueError, match="incompatible"):
        distillation_loss(torch.ones(2, 3), torch.ones(2, 2),
                          torch.ones(2, 3, dtype=torch.bool), torch.ones(2, dtype=torch.bool))


def test_cli_resolves_both_checkpoints_relative_to_config(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    from eidolon_laya_train.cli import cmd_train

    config = tmp_path / "experiment.yaml"
    config.write_text("init: student\ndistill:\n  checkpoint: teacher\n")
    captured = {}

    def fake_train(cfg, dataset, out, log):
        captured.update(cfg)
        return {"best_epoch": 1, "best_score": 0.}

    monkeypatch.setattr(importlib.import_module("eidolon_laya_train.train"), "train", fake_train)
    args = SimpleNamespace(config=str(config), init=None, device="cpu", epochs=None,
                           batch_size=None, dataset=str(tmp_path), out=str(tmp_path))
    assert cmd_train(args) == 0
    assert captured["init"] == str(tmp_path / "student")
    assert captured["distill"]["checkpoint"] == str(tmp_path / "teacher")


def test_teacher_is_frozen_and_incompatible_tokenizer_is_rejected(tmp_path, monkeypatch):
    import hashlib
    import importlib
    from types import SimpleNamespace

    from eidolon_laya_train.model import Loaded
    from eidolon_laya_train.train import load_distillation_teacher

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.encoder.config = SimpleNamespace(to_dict=lambda: {'hidden_size': 2})
            self.head = torch.nn.Dropout(.5)

    def tokenizer(value):
        return SimpleNamespace(pad_token_id=0,
                               backend_tokenizer=SimpleNamespace(to_str=lambda: value))

    student = Loaded(Tiny(), tokenizer('same'), {}, torch.device('cpu'))
    teacher = Loaded(Tiny(), tokenizer('same'), {}, torch.device('cpu'))
    for name in ('model.safetensors', 'rl_agent_config.json'):
        (tmp_path / name).write_bytes(b'fixture')
    monkeypatch.setattr(importlib.import_module('eidolon_laya_train.train'),
                        'load_checkpoint', lambda *args: teacher)
    spec = {'checkpoint': str(tmp_path)}
    result = load_distillation_teacher(spec, student)
    assert result is teacher and not result.model.training
    assert not any(p.requires_grad for p in result.model.parameters())
    assert all(p.requires_grad for p in student.model.parameters())
    assert spec['teacher_weights_sha256'] == hashlib.sha256(b'fixture').hexdigest()
    teacher.tok = tokenizer('different')
    with pytest.raises(ValueError, match='tokenizer differs'):
        load_distillation_teacher(spec, student)
