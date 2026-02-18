# test_trainnet_cli.py
"""
End-to-end tests for trainNet.py command-line interface.
Tests the actual user experience - no imports from trainNet.py needed.
"""

import subprocess
import pytest
from pathlib import Path
from prnn.utils.thetaRNN import RNNCell, LayerNormRNNCell

REPO_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = REPO_ROOT / "examples" / "trainNet.py"


@pytest.fixture
def run_trainnet(tmp_path):
    """
    Helper fixture to run trainNet.py via subprocess.
    Returns a function that runs training and returns the saved network.
    """

    def _run(pRNNtype, **cli_args):
        """
        Run trainNet.py with given arguments.

        Args:
            pRNNtype: The RNN architecture type
            **cli_args: Additional CLI arguments as key=value

        Returns:
            Loaded PredictiveNet object
        """
        save_subfolder = "tmp/" + tmp_path.name + "/"
        # Build command
        cmd = [
            "python",
            str(TRAIN_SCRIPT),
            "--pRNNtype",
            pRNNtype,
            "--hidden_size",
            "100",
            "--numepochs",
            "0",  # Minimal for speed
            "--numtrials",
            "0",  # Minimal for speed
            "--savefolder",
            save_subfolder,
            "--namext",
            "test",
            "--test",
            "--noDataLoader",
        ]

        # Add any extra CLI arguments
        for key, value in cli_args.items():
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

        # Run trainNet.py
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,  # Adjust path as needed
            cwd=REPO_ROOT,
        )

        # Check for errors
        assert result.returncode == 0, (
            f"trainNet.py failed\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout: {result.stdout}\n"
            f"Stderr: {result.stderr}"
        )
        seed = cli_args.get("seed", 8)
        savename = f"{pRNNtype}-test-s{seed}"
        pkl_file = REPO_ROOT / "nets" / save_subfolder / (savename + ".pkl")

        if not pkl_file.exists():
            # Helpful diagnostics
            nets_dir = REPO_ROOT / "nets" / save_subfolder
            if nets_dir.exists():
                found = [f.name for f in nets_dir.glob("*")]
                listing = "\n  ".join(found) or "(empty)"
                msg = (
                    f"Expected: {pkl_file}\n"
                    f"Directory exists but contains:\n  {listing}\n"
                    f"Stdout: {result.stdout}\n"
                    f"Stderr: {result.stderr}"
                )
            else:
                # Walk nets/ to see what DID get created
                nets_root = REPO_ROOT / "nets"
                found_any = list(nets_root.rglob("*.pkl")) if nets_root.exists() else []
                msg = (
                    f"Expected: {pkl_file}\n"
                    f"Directory does not exist: {nets_dir}\n"
                    f"All .pkl files under nets/: {found_any}\n"
                    f"Stdout: {result.stdout}\n"
                    f"Stderr: {result.stderr}"
                )
            pytest.fail(msg)

        # loadNet likely expects the same string saveNet received (without "nets/" prefix and without ".pkl")
        load_string = str(
            Path("tmp") / tmp_path.name / savename
        )  # e.g. "test_thRNN_0win_00/thRNN-test-s8"

        try:
            from prnn.utils.predictiveNet import PredictiveNet

            net = PredictiveNet.loadNet(load_string)
            return net
        except Exception as e:
            pytest.fail(
                f"loadNet failed for string '{load_string}'\n"
                f"pkl_file confirmed exists: {pkl_file.exists()}\n"
                f"cwd: {Path.cwd()}\n"
                f"Error: {e}"
            )

    return _run


# ============================================================================
# TEST PARTIAL PRESETS ARE PRESERVED
# ============================================================================


class TestPartialPresetsViaCLI:
    """Test that CLI preserves partial function presets"""

    # Masked
    def test_thRNN_0win_noLN_k_and_cell(self, run_trainnet):
        """Critical: thRNN_0win should preserve k=0, not override with default"""
        net = run_trainnet("thRNN_0win_noLN")
        actual_k = len(net.pRNN.inMask) - 1
        actual_cell = type(net.pRNN.rnn.cell)
        assert actual_k == 0, f"Expected k=0, got {actual_k}"
        assert actual_cell == RNNCell, f"Expected RNNCell, got {actual_cell}"

    def test_thRNN_10win_k_and_cell(self, run_trainnet):
        """Critical: thRNN_0win should preserve k=0, not override with default"""
        net = run_trainnet("thRNN_10win")
        actual_k = len(net.pRNN.inMask) - 1
        actual_cell = type(net.pRNN.rnn.cell)
        assert actual_k == 10, f"Expected k=10, got {actual_k}"
        assert actual_cell == LayerNormRNNCell, f"Expected LayerNormRNNCell, got {actual_cell}"

    def test_thRNN_prevAct_preserves_actOffset(self, run_trainnet):
        """Test _prevAct variants preserve actOffset=1"""
        net = run_trainnet("thRNN_6win_prevAct")
        actual_k = len(net.pRNN.inMask) - 1
        actual_cell = type(net.pRNN.rnn.cell)
        assert actual_k == 6, f"Expected k=6, got {actual_k}"
        assert net.pRNN.actOffset == 1, f"Expected actOffset=1, got {net.pRNN.actOffset}"
        assert actual_cell == LayerNormRNNCell, f"Expected LayerNormRNNCell, got {actual_cell}"

    def test_thRNN_mask_preserves_mask_actions(self, run_trainnet):
        """Test _mask variants preserve mask_actions"""
        net = run_trainnet("thRNN_4win_mask")
        expected_mask = [True, False, False, False, False]
        actual_k = len(net.pRNN.inMask) - 1
        actual_cell = type(net.pRNN.rnn.cell)
        assert actual_k == 4, f"Expected k=4, got {actual_k}"
        assert actual_cell == LayerNormRNNCell, f"Expected LayerNormRNNCell, got {actual_cell}"
        assert list(net.pRNN.actMask) == expected_mask, (
            f"Expected actMask={expected_mask}, got {list(net.pRNN.actMask)}"
        )

    # Rollouts
    def test_rollout_k(self, run_trainnet):
        net = run_trainnet("thcycRNN_3win")
        assert net.pRNN.k == 3, f"Expected k=3, got {net.pRNN.k}"

    def test_rollout_action_and_continuousTheta(self, run_trainnet):
        """Test rollout variants preserve continuousTheta with rollout_action"""
        # holdc = True
        net = run_trainnet("thcycRNN_5win_fullc")
        assert net.pRNN.rnn.continuousTheta, (
            f"Expected continuousTheta=True, got {net.pRNN.rnn.continuousTheta}"
        )
        assert net.pRNN.actionTheta, f"Expected actionTheta=True, got {net.pRNN.actionTheta}"

        # hold = False
        net = run_trainnet("thcycRNN_5win_hold")
        assert not net.pRNN.rnn.continuousTheta, (
            f"Expected continuousTheta=False, got {net.pRNN.rnn.continuousTheta}"
        )
        assert net.pRNN.actionTheta == "hold", (
            f"Expected actionTheta='hold', got {net.pRNN.actionTheta}"
        )

    def test_rollout_action_conttheta_actoffset(self, run_trainnet):
        net = run_trainnet("thcycRNN_5win_firstc_prevAct")
        actual_cell = type(net.pRNN.rnn.cell)
        assert net.pRNN.rnn.continuousTheta, (
            f"Expected continuousTheta=True, got {net.pRNN.rnn.continuousTheta}"
        )
        assert not net.pRNN.actionTheta, f"Expected actionTheta=False, got {net.pRNN.actionTheta}"
        assert net.pRNN.actOffset == 1, f"Expected actOffset=1, got {net.pRNN.actOffset}"
        assert actual_cell == LayerNormRNNCell, f"Expected LayerNormRNNCell, got {actual_cell}"

    # NextStep
    def test_autoencoder_use_FF(self, run_trainnet):
        """Test that Autoencoder variants freeze W for use_FF (fast forward)"""
        net = run_trainnet("AutoencoderFF")
        actual_cell = type(net.pRNN.rnn.cell)
        W_found = False
        for name, param in net.pRNN.named_parameters():
            if name == "W":
                W_found = True
                assert not param.requires_grad, (
                    f"Expected W.requires_grad=False, got {param.requires_grad}"
                )
                break

        assert W_found, "W parameter not found"
        assert net.pRNN.predOffset == 0, f"Expected predOffset=0, got {net.pRNN.predOffset}"
        assert actual_cell == RNNCell, f"Expected RNNCell, got {actual_cell}"
        net = run_trainnet("AutoencoderPred_LN")
        actual_cell = type(net.pRNN.rnn.cell)
        W_found = False
        for name, param in net.pRNN.named_parameters():
            if name == "W":
                W_found = True
                assert param.requires_grad, (
                    f"Expected W.requires_grad=True, got {param.requires_grad}"
                )
                break

        assert W_found, "W parameter not found"
        assert net.pRNN.predOffset == 1, f"Expected predOffset=1, got {net.pRNN.predOffset}"
        assert actual_cell == LayerNormRNNCell, f"Expected LayerNormCell, got {actual_cell}"

    # Test LognRNNs
    def test_lognRNNs(self, run_trainnet):
        net = run_trainnet("lognRNN_mask")
        actual_k = len(net.pRNN.inMask) - 1
        actual_cell = type(net.pRNN.rnn.cell)

        assert actual_k == 5, f"Expected k=5 got {actual_k}"
        assert actual_cell == LayerNormRNNCell, f"Expected RNNCell, got {actual_cell}"
        assert (net.pRNN.W >= 0).all(), (
            "Expected postiive lognormal distribution, but recurrent weights are negative"
        )
        assert (net.pRNN.W_in >= 0).all(), (
            "Expected postiive lognormal distribution, but input weights are negative"
        )
        sparsity_w = (net.pRNN.W == 0).float().mean().item() * 100
        sparsity_w_in = (net.pRNN.W_in == 0).float().mean().item() * 100
        expected_sparsity = 95.0
        tolerance = 5.0

        assert abs(sparsity_w - expected_sparsity) < tolerance, (
            f"Expected ~{expected_sparsity}% zeros (sparsity=0.05), but got {sparsity_w:.2f}% zeros for recurrent weights"
        )
        assert abs(sparsity_w_in - expected_sparsity) < tolerance, (
            f"Expected ~{expected_sparsity}% zeros (sparsity=0.05), but got {sparsity_w_in:.2f}% zeros for input weights"
        )
        net = run_trainnet("lognRNN_rollout")
        actual_k = net.pRNN.k
        actual_cell = type(net.pRNN.rnn.cell)

        assert actual_k == 5, f"Expected k=5 got {actual_k}"
        assert actual_cell == LayerNormRNNCell, f"Expected RNNCell, got {actual_cell}"
        assert (net.pRNN.W >= 0).all(), (
            "Expected postiive lognormal distribution, but recurrent weights are negative"
        )
        assert (net.pRNN.W_in >= 0).all(), (
            "Expected postiive lognormal distribution, but input weights are negative"
        )
        sparsity_w = (net.pRNN.W == 0).float().mean().item() * 100
        sparsity_w_in = (net.pRNN.W_in == 0).float().mean().item() * 100
        expected_sparsity = 95.0
        tolerance = 5.0

        assert abs(sparsity_w - expected_sparsity) < tolerance, (
            f"Expected ~{expected_sparsity}% zeros (sparsity=0.05), but got {sparsity_w:.2f}% zeros for recurrent weights"
        )
        assert abs(sparsity_w_in - expected_sparsity) < tolerance, (
            f"Expected ~{expected_sparsity}% zeros (sparsity=0.05), but got {sparsity_w_in:.2f}% zeros for input weights"
        )
        assert not net.pRNN.rnn.continuousTheta, (
            f"Expected continuousTheta=False, got {net.pRNN.rnn.continuousTheta}"
        )
        assert net.pRNN.actionTheta, f"Expected actionTheta=True, got {net.pRNN.actionTheta}"


# ============================================================================
# TEST CLI OVERRIDES WORK
# ============================================================================


class TestCLIOverrides:
    """Test that command-line arguments override partial presets"""

    def test_k_override(self, run_trainnet):
        """Test --k=10 overrides thRNN_5win's k=5"""
        net = run_trainnet("thRNN_5win", k=10)
        actual_k = len(net.pRNN.inMask) - 1
        assert actual_k == 10, f"Expected k=10 (override), got {actual_k}"

    def test_actOffset_override(self, run_trainnet):
        """Test --actOffset=3 overrides preset"""
        net = run_trainnet("thRNN_5win_prevAct", actOffset=3)
        assert net.pRNN.actOffset == 3, f"Expected actOffset=3 (override), got {net.pRNN.actOffset}"

    def test_continuousTheta_override(self, run_trainnet):
        """Test --continuousTheta overrides preset"""
        # thcycRNN_5win_hold has continuousTheta=False by default
        net = run_trainnet("thcycRNN_5win_hold", continuousTheta=True)
        assert net.pRNN.rnn.continuousTheta == True, (
            f"Expected continuousTheta=True (override), got {net.pRNN.rnn.continuousTheta}"
        )

    def test_cell_override(self, run_trainnet):
        """Test --cell overrides partial preset"""
        # thRNN_0win has LayerNormRNNCell by default
        net = run_trainnet("thRNN_0win", cell="RNNCell")
        assert type(net.pRNN.rnn.cell) == RNNCell, (
            f"Expected RNNCell (override), got {type(net.pRNN.rnn.cell).__name__}"
        )


# ============================================================================
# TEST DIVNORM PARAMETERS
# ============================================================================


# class TestDivNormViaCLI:
#     """Test that DivNorm parameters flow through CLI correctly"""

#     def test_divnorm_parameters_trainable(self, run_trainnet):
#         """Test --train_divnorm makes parameters trainable"""
#         net = run_trainnet(
#             "Masked",
#             cell="DivNormRNNCell",
#             target_mean=0.8,
#             k_div=2.0,
#             sigma=1.5,
#             train_divnorm=True,
#         )

#         cell = net.pRNN.rnn.cell

#         # Check values reached the cell
#         # assert cell.target_mean == 0.8, f"Expected target_mean=0.8, got {cell.target_mean}"
#         assert cell.divnorm.k_div.item() == 2.0, (
#             f"Expected k_div=2.0, got {cell.divnorm.k_div.item()}"
#         )
#         assert cell.divnorm.sigma.item() == 1.5, (
#             f"Expected sigma=1.5, got {cell.divnorm.sigma.item()}"
#         )

#         # Check they're trainable
#         import torch.nn as nn

#         assert isinstance(cell.divnorm.k_div, nn.Parameter), (
#             "k_div should be a Parameter when train_divnorm=True"
#         )
#         assert isinstance(cell.divnorm.sigma, nn.Parameter), (
#             "sigma should be a Parameter when train_divnorm=True"
#         )

#     def test_divnorm_parameters_fixed(self, run_trainnet):
#         """Test that without --train_divnorm, parameters are fixed"""
#         net = run_trainnet(
#             "Masked",
#             cell="DivNormRNNCell",
#             target_mean=0.7,
#             k_div=1.0,
#             sigma=1.0,
#             # train_divnorm=False is default
#         )

#         cell = net.pRNN.rnn.cell

#         # Check they're NOT trainable (buffers, not Parameters)
#         import torch.nn as nn

#         assert not isinstance(cell.divnorm.k_div, nn.Parameter), (
#             "k_div should be a buffer when train_divnorm=False"
#         )
#         assert not isinstance(cell.divnorm.sigma, nn.Parameter), (
#             "sigma should be a buffer when train_divnorm=False"
#         )


# ============================================================================
# TEST DIVNORM PARAMETERS
# ============================================================================


# class TestDivNormViaCLI:
#     """Test that DivNorm parameters flow through CLI correctly"""

#     def test_divnorm_parameters_trainable(self, run_trainnet):
#         """Test --train_divnorm makes parameters trainable"""
#         net = run_trainnet(
#             "Masked",
#             cell="DivNormRNNCell",
#             target_mean=0.8,
#             k_div=2.0,
#             sigma=1.5,
#             train_divnorm=True,
#         )

#         cell = net.pRNN.rnn.cell

#         # Check values reached the cell
#         # assert cell.target_mean == 0.8, f"Expected target_mean=0.8, got {cell.target_mean}"
#         assert cell.divnorm.k_div.item() == 2.0, (
#             f"Expected k_div=2.0, got {cell.divnorm.k_div.item()}"
#         )
#         assert cell.divnorm.sigma.item() == 1.5, (
#             f"Expected sigma=1.5, got {cell.divnorm.sigma.item()}"
#         )

#         # Check they're trainable
#         import torch.nn as nn

#         assert isinstance(cell.divnorm.k_div, nn.Parameter), (
#             "k_div should be a Parameter when train_divnorm=True"
#         )
#         assert isinstance(cell.divnorm.sigma, nn.Parameter), (
#             "sigma should be a Parameter when train_divnorm=True"
#         )

#     def test_divnorm_parameters_fixed(self, run_trainnet):
#         """Test that without --train_divnorm, parameters are fixed"""
#         net = run_trainnet(
#             "Masked",
#             cell="DivNormRNNCell",
#             target_mean=0.7,
#             k_div=1.0,
#             sigma=1.0,
#             # train_divnorm=False is default
#         )

#         cell = net.pRNN.rnn.cell

#         # Check they're NOT trainable (buffers, not Parameters)
#         import torch.nn as nn

#         assert not isinstance(cell.divnorm.k_div, nn.Parameter), (
#             "k_div should be a buffer when train_divnorm=False"
#         )
#         assert not isinstance(cell.divnorm.sigma, nn.Parameter), (
#             "sigma should be a buffer when train_divnorm=False"
#         )

# ============================================================================
# TEST LR OPTIMIZER CONFIGURATION
# ============================================================================


class TestLRCLI:
    """Test that learning rate configurations work correctly via CLI"""

    def test_eg_lr_configured(self, run_trainnet):
        """Test --eg_lr configures EG optimizer with scaled weight decay"""
        net = run_trainnet("lognRNN_mask", eg_lr=1e-3, eg_weight_decay=1e-6)

        # Check that EG parameter groups exist
        has_eg = False
        W_group = None
        # W_out = None
        W_in = None
        for group in net.optimizer.param_groups:
            if group.get("update_alg") == "eg":
                if group["name"] == "RecurrentWeights":
                    W_group = group
                # if group["name"] == "OutputWeights":
                #     W_out = group
                if group["name"] == "InputWeights":
                    W_in = group
                has_eg = True
                assert group["lr"] == 1e-3, f"Expected eg_lr=1e-3, got {group['lr']}"

                # Check weight decay is scaled by lr
                expected_wd = 1e-6 * 1e-3
                assert abs(group["weight_decay"] - expected_wd) < 1e-12, (
                    f"Expected weight_decay={expected_wd}, got {group['weight_decay']}"
                )

        assert W_group is not None, "Recurring weights not properly updated to EG"
        assert W_in is not None, "Input weights not properly updated to EG"
        # assert W_out is not None, "Output weights not properly updated to EG"
        assert has_eg, "No parameter groups configured for EG"

    def test_bias_lr_uses_gd_not_eg(self, run_trainnet):
        """Test that --bias_lr configures bias learning rate and doesn't use EG"""
        net = run_trainnet(
            "thRNN_5win",
            trainBias=True,
            lr=2e-3,
            weight_decay=3e-3,
            bias_lr=2.0,
            eg_lr=1e-3,
            eg_weight_decay=1e-6,
        )

        # Find the bias parameter group
        bias_group = None
        for group in net.optimizer.param_groups:
            if group["name"] == "biases":
                bias_group = group
                break

        # Check bias group exists
        assert bias_group is not None, "Bias parameter group not found in optimizer"

        # Check bias_lr is applied
        expected_bias_lr = 2.0 * 2e-3
        assert bias_group["lr"] == expected_bias_lr, (
            f"Expected bias lr={expected_bias_lr}, got {bias_group['lr']}"
        )

        # Check biases DON'T use EG (should use GD)
        assert bias_group.get("update_alg") != "eg", (
            f"Biases should use GD, not EG. Got update_alg={bias_group.get('update_alg')}"
        )

        # Biases should have standard weight decay (not EG scaled)
        # For GD: weight_decay is scaled by base_lr * bias_lr * weigth_decay
        # Check it's NOT scaled by eg_lr
        assert bias_group["weight_decay"] == 2.0 * 2e-3 * 3e-3, (
            "Biases should not use EG-scaled weight decay"
        )


# def test_divnorm_trainable_uses_eg(self, run_trainnet):
#     """Test that trainable DivNorm parameters (k_div, sigma) use EG with scaled weight decay"""
#     net = run_trainnet(
#         "Masked",
#         cell="DivNormRNNCell",
#         target_mean=0.7,
#         k_div=1.0,
#         sigma=1.0,
#         train_divnorm=True,
#         eg_lr=1e-3,
#         eg_weight_decay=1e-6,
#     )

#     # Find k_div and sigma parameter groups
#     k_div_group = None
#     sigma_group = None

#     for group in net.optimizer.param_groups:
#         if group["name"] == "k_divnorm":
#             k_div_group = group
#         elif group["name"] == "sigma_divnorm":
#             sigma_group = group

#     # Check both groups exist
#     assert k_div_group is not None, "k_divnorm parameter group not found in optimizer"
#     assert sigma_group is not None, "sigma_divnorm parameter group not found in optimizer"

#     # Check k_div uses EG
#     assert k_div_group.get("update_alg") == "eg", (
#         f"k_div should use EG, got update_alg={k_div_group.get('update_alg')}"
#     )

#     # Check sigma uses EG
#     assert sigma_group.get("update_alg") == "eg", (
#         f"sigma should use EG, got update_alg={sigma_group.get('update_alg')}"
#     )

#     # Check k_div has EG lr
#     assert k_div_group["lr"] == 1e-3, f"k_div should use eg_lr=1e-3, got {k_div_group['lr']}"

#     # Check sigma has EG lr
#     assert sigma_group["lr"] == 1e-3, f"sigma should use eg_lr=1e-3, got {sigma_group['lr']}"

#     # Check k_div has scaled weight decay (eg_weight_decay * eg_lr)
#     expected_wd = 1e-6 * 1e-3
#     assert abs(k_div_group["weight_decay"] - expected_wd) < 1e-12, (
#         f"k_div should use scaled weight_decay={expected_wd}, got {k_div_group['weight_decay']}"
#     )

#     # Check sigma has scaled weight decay (eg_weight_decay * eg_lr)
#     assert abs(sigma_group["weight_decay"] - expected_wd) < 1e-12, (
#         f"sigma should use scaled weight_decay={expected_wd}, got {sigma_group['weight_decay']}"
#     )

#     # Verify parameters are actually trainable (Parameters, not buffers)
#     import torch.nn as nn

#     cell = net.pRNN.rnn.cell
#     assert isinstance(cell.divnorm.k_div, nn.Parameter), (
#         "k_div should be a Parameter when train_divnorm=True"
#     )
#     assert isinstance(cell.divnorm.sigma, nn.Parameter), (
#         "sigma should be a Parameter when train_divnorm=True"
#     )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
