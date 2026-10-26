import numpy as np
import pytest
import torch

from car_sim import config
from car_sim.dqn import DoubleDQNAgent, QNetwork, ReplayBuffer, compute_targets


def fill(buffer, n):
    rng = np.random.default_rng(0)
    for i in range(n):
        buffer.push(
            rng.random(config.STATE_DIM, dtype=np.float32),
            int(rng.integers(0, 5)),
            float(rng.normal()),
            rng.random(config.STATE_DIM, dtype=np.float32),
            bool(i % 7 == 0),
        )


def params(net):
    return [p.detach().clone() for p in net.parameters()]


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_output_shape():
    assert QNetwork()(torch.zeros(3, config.STATE_DIM)).shape == (3, config.ACTION_DIM)


def test_buffer_respects_capacity():
    buffer = ReplayBuffer(5)
    fill(buffer, 10)
    assert len(buffer) == 5


def test_sample_shapes_and_dtypes():
    buffer = ReplayBuffer(100)
    fill(buffer, 50)
    s, a, r, s2, d = buffer.sample(16)
    assert s.shape == (16, config.STATE_DIM) and s.dtype == torch.float32
    assert a.shape == (16,) and a.dtype == torch.int64
    assert r.shape == (16,) and r.dtype == torch.float32
    assert s2.shape == (16, config.STATE_DIM) and s2.dtype == torch.float32
    assert d.shape == (16,) and d.dtype == torch.float32


def test_train_step_none_when_small():
    agent = DoubleDQNAgent()
    fill(agent.buffer, config.BATCH_SIZE - 1)
    assert agent.train_step() is None


def test_train_step_changes_online_weights():
    agent = DoubleDQNAgent()
    fill(agent.buffer, config.BATCH_SIZE)
    before = params(agent.online)
    loss = agent.train_step()
    assert isinstance(loss, float)
    assert not same(before, params(agent.online))


def test_target_unchanged_until_update_then_copies():
    agent = DoubleDQNAgent()
    fill(agent.buffer, config.BATCH_SIZE)
    target_before = params(agent.target)
    agent.train_step()
    assert same(target_before, params(agent.target))
    assert not same(params(agent.online), params(agent.target))
    agent.update_target()
    assert same(params(agent.online), params(agent.target))


def test_double_dqn_target():
    online = lambda x: torch.tensor([[0.0, 5.0, 1.0, 0.0, 0.0]])  # picks action 1
    target = lambda x: torch.tensor([[9.0, 2.0, 0.0, 0.0, 0.0]])  # values action 1 at 2.0
    rewards = torch.tensor([1.0])
    next_states = torch.zeros(1, config.STATE_DIM)
    not_done = compute_targets(online, target, rewards, next_states, torch.tensor([0.0]))
    done = compute_targets(online, target, rewards, next_states, torch.tensor([1.0]))
    assert not_done.item() == pytest.approx(1.0 + config.GAMMA * 2.0)  # plain DQN would give 9
    assert done.item() == pytest.approx(1.0)


def test_act_greedy_when_epsilon_zero():
    agent = DoubleDQNAgent()
    state = np.linspace(0, 1, config.STATE_DIM, dtype=np.float32)
    expected = int(agent.online(torch.from_numpy(state).unsqueeze(0)).argmax(1).item())
    assert all(agent.act(state, 0.0) == expected for _ in range(20))


def test_act_random_when_epsilon_one():
    agent = DoubleDQNAgent()
    state = np.zeros(config.STATE_DIM, dtype=np.float32)
    assert len({agent.act(state, 1.0) for _ in range(200)}) > 1


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "model.pt")
    a, b = DoubleDQNAgent(), DoubleDQNAgent()
    a.save(path, {"epsilon": 0.3, "episode": 7})
    meta = b.load(path)
    assert meta == {"epsilon": 0.3, "episode": 7}
    x = torch.rand(4, config.STATE_DIM)
    assert torch.allclose(a.online(x), b.online(x))
    assert same(params(a.target), params(b.target))  # target network is restored too


def test_load_corrupt_file_raises_valueerror(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"this is not a model")
    with pytest.raises(ValueError):
        DoubleDQNAgent().load(str(path))
