# I-Norm RNN hyperparameter sweep on the pRNN task

`INormRNNCell` (`prnn/utils/thetaRNN.py`) is an excitatory RNN cell. Its LayerNorm is
replaced by learned inhibition: a subtractive pathway predicts the mean of the drive
and a divisive pathway predicts its variance. Both pathways are trained by an aux
moment loss with their own SGD optimizer. The code is ported from RNN_EI_Network's
`SimpleEERNN` + `ParametrizedLayerNorm`; see `prnn/utils/inorm.py`.

To train it outside a sweep:

```bash
python examples/trainNet.py --pRNNtype Masked --cell INormRNNCell --k 5
# or the preset: --pRNNtype inormRNN_mask
```

Options specific to I-Norm:

| Option | Default | What it sets |
|---|---|---|
| `--lr_norm_mean` | 3e-3 | Learning rate of the subtractive pathway |
| `--lr_norm_var0` | 1e-1 | Learning rate of the divisive pathway |
| `--aux_loss_weight` | 1 | Weight of the aux loss. Redundant with the two learning rates above, because the aux loss only trains the inhibitory weights with plain SGD. |
| `--ln_feedback {full,center,fa_center,scale,decorrelate}` | `full` | Which LayerNorm-like backward is used |
| `--no-gradient_norm` | off | Plain I-Norm |
| `--n_div` | hidden size | Number of divisive units |

### How the inhibitory learning rates were chosen

These come from short CPU probes at hidden_size=500 (40 trials):

| Pathway | Learning rate | Result |
|---|---|---|
| Subtractive (mean) | ≤ 1e-2 | Stable |
| Subtractive (mean) | ≥ 3e-2 | Aux loss spikes into the hundreds |
| Subtractive (mean) | 1 | NaN within 6 trials |
| Divisive (var) | 1e-1 to 1 | Stable, and tracks LayerNorm best (MSE to true LN about 3e-4) |
| Both (fmnist values) | 1e-5 / 1e-6 | Inhibition falls behind (aux about 0.2) |

These values are much larger than the fmnist ones for two reasons. The aux loss here is averaged over 500 × 6 steps, whereas RNN_EI_Network sums it over 28 steps. And the excitatory weights change quickly under pRNN's RMSprop.

## Running it on Mila

```bash
cd $HOME/pRNN && git checkout inorm-rnn
bash experiments/inorm_sweep/setup_env.sh      # once: creates .venv and checks the install
mkdir -p sbatch_out sbatch_err
source .venv/bin/activate && wandb login

# 1. Generate the trajectory dataset once (CPU job). 10240 trajectories x 500 steps:
#    about 6 GB in $REPO_ROOT/Data. Generating it took about 0.5 s per trajectory on a Mac
#    CPU, so expect roughly 1.5 h. Each training job copies it to $SLURM_TMPDIR.
sbatch experiments/inorm_sweep/prepare_data.sh

# 2. Pilot: 4 full-size runs, 6 epochs each (W&B group "pilot"); start after step 1 finishes
sbatch experiments/inorm_sweep/pilot.sh

# 3. Sweeps; each `wandb sweep` prints a sweep ID
wandb sweep --project prnn-inorm experiments/inorm_sweep/sweep_inorm.yaml
wandb sweep --project prnn-inorm experiments/inorm_sweep/sweep_layernorm_baseline.yaml
SWEEP_ID=<entity>/prnn-inorm/<id> sbatch experiments/inorm_sweep/sweep_agent.sh   # 60 runs, 10 at a time
```

If the repo or venv are somewhere other than `$HOME/pRNN` and `$HOME/pRNN/.venv`, set
`REPO_ROOT=...` and/or `VENV=...` on the `sbatch` line.

### What to check in the pilot before starting the sweep

- **`epoch_loss`:** the I-Norm runs should track the `layernorm` run. On CPU the losses
  were nearly identical over the first 40 trials.
- **`epoch_aux_loss` and `epoch_ln_mse` / `epoch_ln_cos`:** these show how well the
  inhibition keeps up with the weights. They should stay low and flat for `inorm_default`
  and `inorm_fast`; `inorm_slow` is expected to drift.
- **`epoch_total_time_s` × 20 epochs:** use this to set `--time` in `sweep_agent.sh`
  (currently 24 h), with some margin. The sweep uses 20 epochs.

### What gets logged

| W&B key | Meaning |
|---|---|
| `pRNN loss`, `aux_loss`, `ln_mse`, `ln_cos` | Per training trial (batch) |
| `epoch_loss` | Mean predictive loss over an epoch. **This is the sweep objective** (minimized; its last value = final epoch). Diverged runs log 1e3 and stop. |
| `epoch_aux_loss`, `epoch_ln_mse`, `epoch_ln_cos` | Per epoch: how closely I-Norm matches a true LayerNorm. `ln_mse` is the MSE between the normalized drives; `ln_cos` is the cosine between the relu outputs. |
| `epoch_sRSA`, `epoch_mean_SI`, `epoch_derror` | Per epoch: spatial RSA, mean spatial info, decoding error |
| `epoch_train_time_s`, `epoch_total_time_s` | Per epoch: training time, and time including analysis |
| `final_loss`, `final_sRSA`, `final_SI`, `final_derror`, `diverged` | Run summary |

Networks are pickled to `nets/<savefolder>/<pRNNtype>-<name>-s<seed>.pkl`. Sweep runs use
the W&B run id as the name.

The run is designed to fail safely:
- A non-finite loss stops the epoch and the run, and logs `epoch_loss=1e3`.
- If the per-epoch spatial analysis fails (e.g. on a degenerate network), the error is
  printed and training continues, so the sweep still gets its loss.

## After the sweep: runs for the thesis

1. Find the best config, either by sorting the sweep's runs by `epoch_loss` in W&B or
   with `wandb.Api().sweep("<entity>/prnn-inorm/<id>").best_run().config`.
2. Retrain the best I-Norm and LayerNorm configs for the full 50 epochs with 5 seeds
   (`--seed 1..5 --numepochs 50 --savefolder inorm_final/ --wandb`).
3. Report loss, sRSA, SI and decoding error together. A low prediction loss alone does
   not guarantee place-cell-like representations. `epoch_ln_mse` over training shows how
   closely the learned inhibition matches LayerNorm.

## Notes

- Unlike RNN_EI_Network, the aux loss is averaged over timesteps instead of summed.
- The pRNN readout (`Linear` + `Sigmoid`) is left unconstrained. Only the recurrent
  cell follows Dale's law.
- On macOS, run with `--numworkers 0`: DataLoader workers re-run the script there.
  Linux on Mila is unaffected.
