"""Tests for GimblAgentVariable and the mouse speed model it draws on.

These are contract tests, not calibration tests: they check the things whose
failure would be SILENT in a generated dataset -- wrong dtype, negative speed,
a name that collides in the cache path, and above all a speed trace that is
identical across trajectories. Whether the numbers match the fitted mouse is
the job of
`prnn_training/analysis/encoder/20260803_validate_variable_agent.py`, which
produces a table and a figure rather than an assertion.

No Unity, no cluster, no data files -- the simulated conditions are data-free
by construction, which is exactly what makes them testable here.
"""

import numpy as np
import pytest

from prnn.utils.agent import (GIMBL_VARIABLE_CONDITIONS, GIMBL_VARIABLE_PRESETS,
                              GimblAgentVariable, create_agent)
from prnn.utils.mouseSpeed import BOUT_KW, MOUSE_FIT

TSTEPS = 200


def actions(agent, tsteps=TSTEPS):
    """The uu/step values the agent would send Unity, as a flat array."""
    return np.array([a[0] for a in agent.generateActionSequence(tsteps)])


def speeds(agent, tsteps=TSTEPS):
    """Those actions converted back to cm/s."""
    return actions(agent, tsteps) * agent.cm_per_uu / agent.dt


@pytest.fixture(params=sorted(GIMBL_VARIABLE_PRESETS))
def preset_key(request):
    return request.param


class TestActionContract:
    """The action sequence must be interchangeable with GimblAgentConstant's."""

    def test_length_and_element_shape(self, preset_key):
        agent = create_agent('GimblCorridor', None, preset_key)
        seq = agent.generateActionSequence(TSTEPS)
        assert len(seq) == TSTEPS
        assert all(a.shape == (1,) for a in seq)

    def test_dtype_is_float32(self, preset_key):
        agent = create_agent('GimblCorridor', None, preset_key)
        assert all(a.dtype == np.float32 for a in agent.generateActionSequence(20))

    def test_never_negative(self, preset_key):
        """Forward-only wheel. Clipped at 0, and never abs()'d."""
        agent = create_agent('GimblCorridor', None, preset_key, seed=3)
        assert actions(agent, 2000).min() >= 0.0

    def test_finite(self, preset_key):
        agent = create_agent('GimblCorridor', None, preset_key)
        assert np.all(np.isfinite(actions(agent, 1000)))


class TestAcrossTrajectoryVariability:
    """The regression that matters most.

    generate_trajectories calls generateActionSequence once per trajectory. If
    the RNG were seeded inside that method, every trajectory in a 10k-trajectory
    dataset would carry the identical speed trace -- and nothing downstream
    would raise. The dataset would train fine and contain no behavioural
    variability, which is the entire reason this agent exists.
    """

    def test_consecutive_calls_differ(self, preset_key):
        agent = create_agent('GimblCorridor', None, preset_key, seed=0)
        first, second = actions(agent), actions(agent)
        assert not np.array_equal(first, second)

    def test_many_calls_are_all_distinct(self, preset_key):
        agent = create_agent('GimblCorridor', None, preset_key, seed=0)
        traces = np.stack([actions(agent) for _ in range(32)])
        # An all-zero trace is a LEGITIMATE duplicate: a short trajectory can
        # fall entirely inside one stop bout (mean stop ~10.5 s against a 20 s
        # trial), and two fully-stopped trials are identical by definition. So
        # the distinctness claim applies to trajectories that actually move.
        moving = traces[traces.any(axis=1)]
        assert len(moving) >= 2, 'no moving trajectories -- nothing was tested'
        assert len(np.unique(moving, axis=0)) == len(moving)

    def test_short_trials_can_be_entirely_stopped(self):
        """Documented consequence, not a bug.

        With mean stop ~10.5 s a 20 s trajectory can contain no movement at
        all: a training sequence with zero optic flow. `min_running_fraction`
        is the knob that rejects these, and panel (h) of the validation figure
        is where you decide whether you need it.
        """
        agent = create_agent('GimblCorridor', None,
                             'GimblAgentVariableSimPauses', seed=0)
        traces = np.stack([actions(agent) for _ in range(64)])
        assert (~traces.any(axis=1)).sum() > 0

    def test_speed_actually_varies_within_a_trajectory(self, preset_key):
        """Guards the other degenerate case: a constant-valued 'variable' agent."""
        assert speeds(create_agent('GimblCorridor', None, preset_key)).std() > 0.5


class TestReproducibility:
    def test_same_seed_same_sequence(self, preset_key):
        a = create_agent('GimblCorridor', None, preset_key, seed=7)
        b = create_agent('GimblCorridor', None, preset_key, seed=7)
        assert np.array_equal(actions(a), actions(b))

    def test_same_seed_reproduces_a_whole_run(self, preset_key):
        """Reproducibility must hold across the sequence of trajectories, not
        just the first one -- that is the property the instance RNG buys."""
        a = create_agent('GimblCorridor', None, preset_key, seed=7)
        b = create_agent('GimblCorridor', None, preset_key, seed=7)
        for _ in range(5):
            assert np.array_equal(actions(a), actions(b))

    def test_different_seed_differs(self, preset_key):
        a = create_agent('GimblCorridor', None, preset_key, seed=1)
        b = create_agent('GimblCorridor', None, preset_key, seed=2)
        assert not np.array_equal(actions(a), actions(b))


class TestCachePathNaming:
    """agent.name is part of the dataset folder name, rebuilt independently in
    three places (generate_gimbl_trajectories.py, trainNet_res_all.py's
    --zscore_latents branch, and create_dataloader). A collision here means two
    conditions silently share one dataset directory."""

    def test_name_is_the_preset_key(self, preset_key):
        assert create_agent('GimblCorridor', None, preset_key).name == preset_key

    def test_conditions_do_not_share_a_name(self):
        names = {create_agent('GimblCorridor', None, k).name
                 for k in GIMBL_VARIABLE_PRESETS}
        assert len(names) == len(GIMBL_VARIABLE_PRESETS)

    def test_name_has_no_folder_separator(self, preset_key):
        """The folder name joins fields with '-', so a '-' or '/' in the agent
        name would corrupt the parse."""
        name = create_agent('GimblCorridor', None, preset_key).name
        assert '-' not in name and '/' not in name

    def test_does_not_collide_with_the_constant_agent(self, preset_key):
        constant = create_agent('GimblCorridor', None, 'GimblAgentConstant')
        assert create_agent('GimblCorridor', None, preset_key).name != constant.name


class TestConditions:
    def test_presets_cover_every_condition(self):
        covered = {p['condition'] for p in GIMBL_VARIABLE_PRESETS.values()}
        assert covered == set(GIMBL_VARIABLE_CONDITIONS)

    def test_fitted_condition_never_stops(self):
        v = speeds(GimblAgentVariable(condition='riab_fitted', seed=0), 5000)
        assert (v <= BOUT_KW['speed_threshold']).mean() < 0.05

    def test_pauses_condition_stops_substantially(self):
        v = speeds(GimblAgentVariable(condition='riab_with_pauses', seed=0), 5000)
        assert 0.2 < (v <= BOUT_KW['speed_threshold']).mean() < 0.7

    def test_pauses_condition_reaches_exactly_zero(self):
        """rest_speed_sd is 0.0, so a stop is a true stop -- the wheel is not
        turning. If this fails, MOUSE_FIT['rest_speed_sd'] has been changed."""
        assert MOUSE_FIT['rest_speed_sd'] == 0.0
        v = speeds(GimblAgentVariable(condition='riab_with_pauses', seed=0), 5000)
        assert (v == 0.0).any()

    def test_unknown_condition_raises(self):
        with pytest.raises(ValueError, match='unknown condition'):
            GimblAgentVariable(condition='riab_nope')


class TestUnitConversion:
    """action[uu/step] = v[cm/s] * dt / cm_per_uu. cm/s is the interchange unit;
    uu/step never ports between environments."""

    def test_rnn_corridor_divides_by_twenty(self):
        agent = GimblAgentVariable(condition='riab_fitted', seed=0)
        assert agent.dt / agent.cm_per_uu == pytest.approx(1 / 20)

    def test_behaviormate_scale_divides_by_thirty(self):
        agent = GimblAgentVariable(condition='riab_fitted', cm_per_uu=3.0, seed=0)
        assert agent.dt / agent.cm_per_uu == pytest.approx(1 / 30)

    def test_scale_changes_actions_but_not_the_speed_model(self):
        """Same seed, different env scale -> same cm/s trace, scaled actions."""
        a = GimblAgentVariable(condition='riab_fitted', seed=5)
        b = GimblAgentVariable(condition='riab_fitted', seed=5, cm_per_uu=3.0)
        np.testing.assert_allclose(speeds(a), speeds(b), rtol=1e-5)
        np.testing.assert_allclose(actions(a) * 2.0, actions(b) * 3.0, rtol=1e-5)


class TestOptionalKnobs:
    def test_max_speed_caps_the_trace(self):
        agent = GimblAgentVariable(condition='riab_fitted', seed=0, max_speed=15.0)
        assert speeds(agent, 5000).max() <= 15.0 + 1e-6

    def test_max_speed_off_by_default_leaves_the_tail(self):
        assert speeds(GimblAgentVariable(condition='riab_fitted', seed=0),
                      20000).max() > 20.0

    def test_min_running_fraction_rejects_quiet_trajectories(self):
        agent = GimblAgentVariable(condition='riab_with_pauses', seed=0,
                                   min_running_fraction=0.4, max_resample=200)
        fracs = [(speeds(agent, 500) > BOUT_KW['speed_threshold']).mean()
                 for _ in range(20)]
        assert min(fracs) >= 0.4

    def test_min_running_fraction_warns_when_unreachable(self):
        agent = GimblAgentVariable(condition='riab_with_pauses', seed=0,
                                   min_running_fraction=1.01, max_resample=2)
        with pytest.warns(UserWarning, match='min_running_fraction'):
            agent.generateActionSequence(200)


class TestCreateAgent:
    def test_kwargs_reach_the_constant_agent(self):
        """create_agent used to call GimblAgentConstant() with no arguments, so
        its speed could not be set through this path at all."""
        assert create_agent('GimblCorridor', None, 'GimblAgentConstant',
                            speed=0.42).speed == 0.42

    def test_kwargs_override_preset_defaults(self):
        agent = create_agent('GimblCorridor', None, 'GimblAgentVariableSimPauses',
                             cm_per_uu=3.0)
        assert agent.cm_per_uu == 3.0 and agent.condition == 'riab_with_pauses'

    def test_unknown_key_raises_valueerror(self):
        """It used to fall through to `return agent` unbound -> UnboundLocalError."""
        with pytest.raises(ValueError, match='unknown agentkey'):
            create_agent('GimblCorridor', None, 'NotAnAgent')
