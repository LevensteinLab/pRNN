# I-Norm RNN hyperparameter sweep on the pRNN task

`INormRNNCell` (`prnn/utils/thetaRNN.py`) is an excitatory RNN cell. Its LayerNorm is
replaced by learned inhibition: a subtractive pathway predicts the mean of the drive
and a divisive pathway predicts its variance. Both pathways are trained by an aux
moment loss with their own SGD optimizer. The code is ported from RNN_EI_Network's
`SimpleEERNN` + `ParametrizedLayerNorm`; see `prnn/utils/inorm.py`.

To train it outside a sweep:

```bash
python examples/trainNet.py --pRNNtype Masked --cell INormRNNCell --k 5 \
    --lr 2e-3 --lr_norm_mean 1e-5 --lr_norm_var0 1e-6 --aux_loss_weight 1
# or the preset: --pRNNtype inormRNN_mask
```

Options specific to I-Norm:

| Option | What it sets |
|---|---|
| `--aux_loss_weight` | Weight of the aux moment loss |
| `--lr_norm_mean` | Learning rate of the subtractive pathway |
| `--lr_norm_var0` | Learning rate of the divisive pathway |
| `--ln_feedback {full,center,fa_center,scale,decorrelate}` | Which LayerNorm-like backward is used |
| `--no-gradient_norm` | Plain I-Norm |
| `--n_div` | Number of divisive units |

## Running the sweep (Mila cluster)

```bash
cd $HOME/pRNN            # repo root, on the inorm-rnn branch, with .venv set up
mkdir -p sbatch_out sbatch_err
wandb login

# 1. Generate the trajectory dataset once (CPU job)
sbatch experiments/inorm_sweep/prepare_data.sh

# 2. Create the sweep(s); each command prints a sweep ID
wandb sweep --project prnn-inorm experiments/inorm_sweep/sweep_inorm.yaml
wandb sweep --project prnn-inorm experiments/inorm_sweep/sweep_layernorm_baseline.yaml   # baseline

# 3. Launch agents (60 runs, 10 at a time), once the data job has finished
SWEEP_ID=<entity>/prnn-inorm/<id> sbatch experiments/inorm_sweep/sweep_agent.sh
```

If the repo or venv are somewhere other than `$HOME/pRNN` and `$HOME/pRNN/.venv`, set
`REPO_ROOT=...` and/or `VENV=...` on the `sbatch` line.

### What gets logged

| W&B key | Meaning |
|---|---|
| `pRNN loss`, `aux loss` | Per training trial (batch) |
| `epoch_loss` | Mean predictive loss over an epoch. **This is the sweep objective** (minimized; its last value = final epoch). Diverged runs log 1e3. |
| `epoch_aux_loss`, `epoch_sRSA`, `epoch_mean_SI`, `epoch_derror` | Per epoch: aux loss, spatial RSA, mean spatial info, decoding error |
| `final_loss`, `final_sRSA`, `final_SI`, `final_derror`, `diverged` | Run summary |

Networks are pickled to `nets/inorm_sweep/Masked-<wandb run id>-s8.pkl`.

## After the sweep: runs for the thesis

1. In the W&B sweep view, sort runs by `epoch_loss`. Or use the API:
   ```python
   import wandb
   sweep = wandb.Api().sweep("<entity>/prnn-inorm/<id>")
   best = sweep.best_run()
   print(best.config)
   ```
2. Retrain the best config with several seeds and the full 50 epochs. Do the same for
   the LayerNorm baseline:
   ```bash
   for s in 1 2 3 4 5; do
     python examples/trainNet.py --pRNNtype Masked --cell INormRNNCell --k 5 --numepochs 50 \
       --lr <lr> --bias_lr <bias_lr> --weight_decay <wd> --lr_norm_mean <m> \
       --lr_norm_var0 <v> --aux_loss_weight <a> --seed $s --namext best \
       --savefolder inorm_final/ --wandb
   done
   ```

## Notes

- The sweep uses 20 epochs × 1024 trials per run to save compute. Set `--time` in
  `sweep_agent.sh` from how long the first runs actually take.
- Unlike RNN_EI_Network, the aux loss is averaged over timesteps instead of summed,
  because pRNN sequences are 500 steps × 6 theta steps long.
- The pRNN readout (`Linear` + `Sigmoid`) is left unconstrained. Only the recurrent
  cell follows Dale's law.
