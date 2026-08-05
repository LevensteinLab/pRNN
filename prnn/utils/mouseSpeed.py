"""
mouseSpeed.py -- a mouse's forward locomotion speed over time, in cm/s.

WHAT THIS IS: a behavioural model of running speed, plus the bout detector and
summary table used to check it. Nothing in here knows about Unity, uu/step, or
trajectories -- it *produces* a speed trace in cm/s and *measures* speed traces
in cm/s. `GimblAgentVariable` (agent.py) is the adapter that turns a trace into
Gimbl actions. Keeping the model env-agnostic is deliberate: the same speed
model could drive a different environment, or generate reference traces for an
analysis figure with no agent involved.

PROVENANCE: `MOUSE_FIT` was fitted against real CA3 treadmill velocity by
`ca3_prnn/analysis/20260803_mouse_riab_speed_comparison.py` (2026-08-03) --
expt 1707, ONE 10-minute session, 6189 frames @ 10.31 Hz. That script is the
*fitter* and imports this module; this module is the single home for the
parameters, the model, and the detector, so the two cannot drift apart.

REFIT when more sessions (or Meghan's trajectory database) land. The duration
fits rest on n=26 run and n=25 stop bouts, where a KS test has little power to
choose between families -- gamma and lognormal are both "not rejected" for
both. The families below were chosen on shape: runs are tight and roughly
Gaussian, stops are heavy-tailed. Related: a single 10-minute trace has sd
0.048 on its own running fraction, so never tune against just one.

THE TWO FACTS THAT DRIVE THE DESIGN (full write-up: `prnn` skill ->
references/riab.md):

 1. `speed_std` in a single OU controls BOTH the width of the running-speed
    distribution AND how often the process wanders down to zero. One knob, two
    behaviours -- so NO single OU can have a mouse's tight running speeds *and*
    its seconds-long stops. Tighten sd to match the running spread and the
    stops vanish (one continuous 600 s bout); widen it for stops and the
    ceiling inflates to 2-3x a real mouse. That is the whole reason
    `simulate_bout_gated` exists: it gates a *tight* OU with separately
    specified bout durations, so the two behaviours get independent knobs.

 2. Clip negative speed at 0; never use abs(). On a real trace abs() collapses
    36 run bouts into 1 (median 600 s), because every zero-crossing becomes a
    speed *peak*, so the trace never dwells near zero and the bout detector
    sees one unbroken run. max(v, 0) makes backwards a pause, which is what it
    physically is -- ~9% of real frames are negative, but 92% of those are
    above -1 cm/s and only 1.6% fall inside a run bout, i.e. encoder jitter at
    rest rather than backward locomotion. Clipping costs 0.4% of path length.
"""

import numpy as np

# --- Gimbl / pRNN unit conventions -------------------------------------------
# The RNN Gimbl corridor is 2 cm/uu (90 uu = 180 cm) and one pRNN step is
# 100 ms, so action[uu/step] = v[cm/s] * STEP_SECONDS / CM_PER_UU = v / 20.
# Meghan's behaviorMate+Gimbl env is 3 cm/uu -> v / 30. The two scales are
# settled and DIFFERENT: keep every trace in cm/s and apply the divisor of the
# env you are driving. uu/step never ports between the two.
CM_PER_UU = 2.0          # RNN corridor: 90 uu = 180 cm
STEP_SECONDS = 0.1       # one pRNN step = 100 ms
TRACK_CM = 180.0         # RNN corridor length, for lap counting

# --- bout-detection defaults, matching 20260728_0pf_analysis_ca3.py ----------
BOUT_KW = dict(stationary_tolerance=1.0, min_duration=0.5, min_mean_speed=2.0,
               min_peak_speed=5.0, preceding_still_time=0.5, speed_threshold=2.0)

# --- fitted mouse parameters, for a DATA-FREE simulated agent -----------------
# Data-free is a requirement, not a convenience: Meghan is building a database
# of real mouse trajectories for the *preloaded* path, so the *simulated* path
# must never depend on an exported .npz at run time.
MOUSE_FIT = dict(
    run_speed_mean=0.1118,          # m/s  -> RiaB speed_mean
    run_speed_std=0.0354,           # m/s  -> RiaB speed_std
    run_coherence_time=0.48,        # s    -> RiaB speed_coherence_time (WITHIN-run)
    run_dur=('gamma', 10.31, 1.215),      # (a, scale) in seconds
    stop_dur=('lognorm', 0.849, 7.333),   # (sigma, exp(mu)) in seconds
    # At-rest jitter during a stop bout, cm/s (zero-mean). Set to 0.0 on
    # 2026-08-03: the wheel genuinely is not turning during a stop, and a
    # SYMMETRIC Gaussian is the wrong shape for real at-rest jitter (which is
    # skewed) -- it gave 19.2% negative frames against the mouse's 8.9%. The
    # previous value was 0.322 cm/s, which is 0.016 uu/step, i.e. 0.03 cm of
    # avatar movement per step: far below anything visible in a 96x96 frame,
    # so this changes no observation the pRNN can resolve. It only affects
    # 'riab_with_pauses' -- the running-only condition never enters a stop.
    rest_speed_sd=0.0,
    frac_running=0.543,
)

# --- VR running wheel, fitted 2026-08-05 --------------------------------------
# Fitted to mc31 2026-07-24 (4m_ctxA, track_length 4000 mm), the session the
# SameVR comparison is against. Same detector and BOUT_KW as MOUSE_FIT above, on
# the behaviorMate position stream resampled to exactly 10 Hz (it streams at
# ~80 Hz, so the pRNN's 100 ms step needs no rescaling).
#
# WHY A SECOND FIT: running GimblAgentVariableSimPauses (the MOUSE_FIT/belt
# calibration) against this session gave 6.27 cm/s where the mouse ran 10.04 --
# a 1.6x shortfall that understates optic flow at every step. The fit below
# removes it, and GimblAgentVariableWheelPauses reproduces this session's mean,
# running speed, immobile fraction and laps-per-500-steps.
#
# ⚠️ BE CAREFUL ABOUT *WHY*, because the obvious explanation is not established.
# It is tempting to call this a rig effect ("mice run faster on a VR wheel than a
# tactile belt" -- Meghan's impression, 2026-08-05). Measured through ONE pipeline
# the rigs overlap heavily: belt (tg127, n=3) mean 10.48, range 6.2-14.4 cm/s;
# wheel (mc31/mc34, n=5) mean 13.37, range 10.0-18.6. Plausible, not demonstrated.
# Most of the apparent gap is instead a MEASUREMENT-PATH difference: MOUSE_FIT was
# built from lab's exported velocity for expt 1707 (6.11 cm/s), whereas this fit
# reads behaviorMate .tdml position resampled to 10 Hz -- and that same tg127
# session reads 10.86 cm/s through the .tdml path. (Unresolved: lab's 1707 export
# covers 600 s, the same-date tdml only 356 s.) The tdml path is internally
# validated -- rectified distance matches laps x track_length within 3% on all
# eight sessions -- but it is NOT interchangeable with lab's.
#
# ⚠️ SO: SESSION-SPECIFIC, DO NOT POOL, AND MATCH THE MEASUREMENT PATH. Session
# means span 6.2-18.6 cm/s across the eight sessions, and within mc31 they rise
# monotonically with training day (7/24 10.04, 7/29 14.63, 7/30 18.61). A pooled
# wheel fit gives 15.83 cm/s while running -- matching no session, and 1.17x this
# one. A fitted agent must be tied to the session it is compared against and
# measured the way that session's position will be analysed. Refit, don't reuse.
MOUSE_FIT_WHEEL = dict(
    run_speed_mean=0.1357,                 # m/s (13.57 cm/s) vs 0.1118 on the belt
    run_speed_std=0.0452,                  # m/s (4.52 cm/s)
    run_coherence_time=0.20,               # s -- twitchier than the belt's 0.48
    run_dur=('gamma', 4.21, 2.183),        # median 7.8 s, n=48 bouts
    stop_dur=('lognorm', 0.471, 2.938),    # median 2.7 s, n=47
    rest_speed_sd=0.0,                     # same reasoning as MOUSE_FIT
    frac_running=0.736,                    # vs 0.543 on the belt
)


# =============================================================================
# bout detection (port of lab.analyses.behavior.running_intervals)
# =============================================================================
def running_intervals_from_velocity(
        vel, period, stationary_tolerance=0.5, min_duration=1,
        min_mean_speed=3, min_peak_speed=5, preceding_still_time=0.5,
        speed_threshold=2):
    """Nx2 array of [start, end] running-frame indices (inclusive).

    Identical logic to lab.analyses.behavior.running_intervals, but taking a raw
    velocity array + frame period instead of an experiment object, so the same
    bout definition can be applied to simulated traces. `end_padding` is dropped
    (unused here).
    """
    vel = np.asarray(vel, dtype=float)
    n_frames = vel.shape[0]

    running_inds = np.where(vel > speed_threshold)[0]
    if running_inds.size == 0:
        return np.zeros((0, 2), dtype=int)

    max_gap_frames = int(stationary_tolerance / period)
    gaps = np.diff(running_inds)
    ends_idx = np.where(gaps > max_gap_frames)[0]
    starts_idx = np.hstack(([0], ends_idx + 1)).astype(int)
    ends_idx = np.hstack((ends_idx, [running_inds.size - 1])).astype(int)

    good = np.ones(len(starts_idx), dtype=bool)
    for i, (s_idx, e_idx) in enumerate(zip(starts_idx, ends_idx)):
        start_frame, end_frame = running_inds[s_idx], running_inds[e_idx]

        if preceding_still_time > 0 and i > 0:
            prev_end = running_inds[ends_idx[i - 1]]
            if (start_frame - prev_end) * period < preceding_still_time:
                good[i] = False

        duration = (end_frame - start_frame + 1) * period
        if duration < min_duration:
            good[i] = False
            continue

        seg = vel[start_frame:end_frame + 1]
        if np.mean(np.abs(seg)) < min_mean_speed or np.max(np.abs(seg)) < min_peak_speed:
            good[i] = False

    starts, ends = running_inds[starts_idx[good]], running_inds[ends_idx[good]]
    if starts.size == 0:
        return np.zeros((0, 2), dtype=int)
    return np.vstack((starts, ends)).T.astype(int)


def bout_durations(vel, period, n_frames=None, **kw):
    """(run_durations, stop_durations) in seconds."""
    iv = running_intervals_from_velocity(vel, period, **kw)
    if iv.shape[0] == 0:
        return np.array([]), np.array([])
    n = n_frames if n_frames is not None else len(vel)
    runs = (iv[:, 1] - iv[:, 0] + 1) * period
    # stops = the gaps between consecutive run bouts (excluding the head/tail,
    # which are censored by the start/end of the recording)
    stops = (iv[1:, 0] - iv[:-1, 1] - 1) * period
    stops = stops[stops > 0]
    return runs, stops


# =============================================================================
# RatInABox 1-D motion model
# =============================================================================
def ornstein_uhlenbeck(dt, x, drift, noise_scale, coherence_time, rng):
    """Verbatim port of ratinabox.utils.ornstein_uhlenbeck (v1.7.1).

    Stationary sd is `noise_scale` and the ACF is *exactly* exponential, so the
    measured tau (1/e crossing of the autocorrelation) IS `coherence_time`.
    Useful self-check: fit tau back out and you should recover what you set.
    """
    x = np.asarray(x, dtype=float)
    sigma = np.sqrt((2 * noise_scale ** 2) / (coherence_time * dt))
    theta = 1 / coherence_time
    return theta * (drift - x) * dt + sigma * rng.normal(size=x.shape, scale=dt)


def simulate_riab_1d(n_steps, dt, speed_mean, speed_std, coherence_time=0.7,
                     seed=0, use_package=True, rng=None, track_cm=None):
    """Signed velocity (cm/s) from RiaB's 1-D motion model.

    speed_mean / speed_std / coherence_time are in RiaB's native units (m/s, s).
    Returns cm/s. The two paths agree because the port copies the update order
    in Agent.update (position first, then the OU step on velocity).

    use_package=True uses the installed ratinabox and is AUTHORITATIVE -- that is
    what the calibration script checks the port against. It is unsuitable for
    data generation, though: RiaB draws from numpy's *global* RNG, so this path
    calls np.random.seed() and would reset global random state on every
    trajectory. Data generation should pass use_package=False plus its own `rng`
    (see GimblAgentVariable), which keeps the draws on a private generator.
    """
    if use_package:
        try:
            import ratinabox
            from ratinabox.Environment import Environment
            from ratinabox.Agent import Agent
            ratinabox.stylize_plots = False
            np.random.seed(seed)          # RiaB uses numpy's global RNG
            Env = Environment(params={'dimensionality': '1D',
                                      'boundary_conditions': 'periodic',
                                      'scale': (track_cm or TRACK_CM) / 100.0})
            Ag = Agent(Env, {'dt': dt, 'speed_mean': speed_mean,
                             'speed_std': speed_std,
                             'speed_coherence_time': coherence_time})
            for _ in range(n_steps):
                Ag.update()
            vel = np.asarray(Ag.history['vel'], dtype=float).squeeze()
            return vel * 100.0, 'ratinabox %s' % getattr(ratinabox, '__version__', '1.7.1')
        except Exception as exc:                                   # noqa: BLE001
            print('  [ratinabox unavailable (%s) -- using the verbatim port]' % exc)

    if rng is None:
        rng = np.random.default_rng(seed)
    v = np.zeros(n_steps)
    v[0] = speed_mean
    for t in range(1, n_steps):
        v[t] = v[t - 1] + ornstein_uhlenbeck(dt, v[t - 1], speed_mean, speed_std,
                                             coherence_time, rng)
    return v * 100.0, 'verbatim port'


def draw_duration(rng, spec, floor=0.5):
    """One bout duration in seconds.

    `spec` is either an array of measured durations (bootstrap -- needs the real
    data) or a tuple ('gamma', a, scale) / ('lognorm', sigma, exp(mu)) (fitted --
    data-free). Clamped at `floor`, which defaults to the bout detector's own
    `min_duration`: a shorter bout would be re-merged by `stationary_tolerance`
    and so could never be measured back out of the generated trace.
    """
    if isinstance(spec, tuple):
        kind, a, scale = spec
        if kind == 'gamma':
            d = rng.gamma(a, scale)
        elif kind == 'lognorm':
            d = scale * np.exp(a * rng.standard_normal())
        else:
            raise ValueError('unknown duration family: %s' % kind)
    else:
        d = float(rng.choice(spec))
    return max(d, floor)


def draw_rest(rng, spec, size):
    """`size` at-rest speeds (cm/s): from measured immobile frames (array) or
    from a zero-mean Gaussian of the given sd (scalar, data-free).

    A scalar 0.0 gives exactly-zero speeds -- the wheel is not turning during a
    stop. See the MOUSE_FIT['rest_speed_sd'] comment for why that is the default.
    """
    if np.isscalar(spec):
        if spec == 0:
            return np.zeros(size)
        return rng.normal(0.0, float(spec), size)
    return rng.choice(spec, size=size)


def simulate_bout_gated(n_steps, period, ou_trace, run_durs, stop_durs,
                        rest_pool, seed=0, rng=None, frac_running=None):
    """Two-state (run / stop) speed trace, in cm/s.

    RiaB's OU supplies the within-run speed; a separate process supplies the
    bout structure. The reason for splitting them: `speed_std` in a single OU
    controls BOTH the width of the running-speed distribution AND how often the
    process wanders down to zero, so no single OU can have a mouse's tight
    running speeds *and* its seconds-long stops -- tightening sd to match the
    former removes the latter. Here the two are independent.

    Bout durations come from `run_durs`/`stop_durs`, which may be measured
    arrays (bootstrap, needs real data) or fitted family tuples (data-free).
    Either way this is "semi-Markov": dwell times come from an arbitrary
    distribution rather than the geometric one a plain 2-state Markov chain
    would force, which matters because a geometric dwell time makes very short
    bouts far too common.

    Reproducing the whole-trace autocorrelation timescale (4.17 s on the real
    trace) WITHOUT touching the speed model is itself the proof that 4.17 s was
    bout structure rather than speed fluctuation -- the within-run timescale is
    0.48 s. Fitting an OU's coherence time to 4.17 s is the trap; don't.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if not isinstance(run_durs, tuple) and len(run_durs) == 0:
        return np.clip(ou_trace[:n_steps], 0, None)

    out = np.empty(n_steps)
    # start in whichever state, weighted by how much time is spent in each
    if isinstance(run_durs, tuple):
        p_run = MOUSE_FIT['frac_running'] if frac_running is None else frac_running
    else:
        p_run = np.mean(run_durs) / (np.mean(run_durs) + np.mean(stop_durs))
    running = rng.random() < p_run
    i, ou_ptr = 0, 0
    while i < n_steps:
        spec = run_durs if running else stop_durs
        d = max(int(round(draw_duration(rng, spec,
                                        floor=BOUT_KW['min_duration']) / period)), 1)
        d = min(d, n_steps - i)
        if running:
            # consecutive samples from the OU, to keep its within-run structure
            out[i:i + d] = np.take(ou_trace, np.arange(ou_ptr, ou_ptr + d),
                                   mode='wrap')
            ou_ptr += d
        else:
            out[i:i + d] = draw_rest(rng, rest_pool, d)
        i += d
        running = not running
    return out


# =============================================================================
# diagnostics
# =============================================================================
def autocorr(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    ac = np.correlate(x, x, mode='full')[len(x) - 1:len(x) - 1 + max_lag + 1]
    return ac / ac[0] if ac[0] != 0 else ac


def acf_timescale(ac, period):
    """Lag (s) at which the ACF first drops below 1/e."""
    below = np.where(ac < 1 / np.e)[0]
    return below[0] * period if below.size else np.nan


def summarize(name, vel, period):
    """One row of the printed comparison table."""
    v = vel[np.isfinite(vel)]
    runs, stops = bout_durations(v, period, **BOUT_KW)
    ac = autocorr(v, int(10 / period))
    n_steps = len(v)
    laps = np.abs(v).sum() * period / TRACK_CM        # forward path length / track
    return dict(
        name=name,
        mean=np.mean(v), sd=np.std(v),
        pct_neg=100 * np.mean(v < 0),
        pct_immobile=100 * np.mean(v <= BOUT_KW['speed_threshold']),
        vmax=np.max(v), vmin=np.min(v),
        tau=acf_timescale(ac, period),
        n_runs=len(runs),
        med_run=np.median(runs) if len(runs) else np.nan,
        med_stop=np.median(stops) if len(stops) else np.nan,
        laps_per_500=500 * laps / n_steps,
        uu_mean=np.mean(v) * STEP_SECONDS / CM_PER_UU,
    )


def print_table(rows):
    hdr = ('%-26s %6s %6s %6s %6s %6s %6s %5s %6s %6s %7s %7s'
           % ('trace', 'mean', 'sd', 'min', 'max', '%neg', '%imm',
              'nrun', 'medrun', 'medstp', 'lap/500', 'uu/step'))
    print('\n' + hdr)
    print('-' * len(hdr))
    for r in rows:
        print('%-26s %6.2f %6.2f %6.2f %6.2f %6.1f %6.1f %5d %6.2f %6.2f %7.2f %7.3f'
              % (r['name'], r['mean'], r['sd'], r['vmin'], r['vmax'], r['pct_neg'],
                 r['pct_immobile'], r['n_runs'], r['med_run'], r['med_stop'],
                 r['laps_per_500'], r['uu_mean']))
    print('\nmean/sd/min/max in cm/s, medrun/medstp in s; %%imm = frames at or below '
          '%.0f cm/s;\nlap/500 = laps of a %.0f cm track per 500 steps; uu/step = mean '
          'Gimbl action.' % (BOUT_KW['speed_threshold'], TRACK_CM))
