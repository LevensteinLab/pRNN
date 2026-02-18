"""
test_architecture_flow.py

Tests that verify:
1. Partial function arguments are correctly passed through
2. Argparse arguments flow correctly to architectures and cells
3. No unintended overrides occur
"""

import torch
import pytest
from types import SimpleNamespace
import sys

# sys.path.append("../")  # Adjust path as needed

from prnn.utils.predictiveNet import PredictiveNet, CELL_TYPES, netOptions
from prnn.utils.env import make_env
from prnn.utils.Architectures import *
from prnn.utils.thetaRNN import *


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_env():
    """Create a simple test environment"""
    env = make_env("LRoom-18x18-v0", "farama-minigrid", "SpeedHD")
    return env


@pytest.fixture
def base_architecture_kwargs():
    """Base kwargs that should work with any architecture"""
    return {
        "dropp": 0.15,
        "neuralTimescale": 2,
        "bptttrunc": 50,
    }


# ============================================================================
# TEST 1: PARTIAL PRESETS ARE PRESERVED (No Argparse Override)
# ============================================================================


class TestPartialPresets:
    """Test that partial function presets are preserved when no override is given"""

    def test_cell_values(self, mock_env, base_architecture_kwargs):
        test_cases = [
            # ("thRNN_0win_noLN", RNNCell),
            ("thRNN_0win", LayerNormRNNCell),
            ("thcycRNN_5win_holdc_adapt", AdaptingLayerNormRNNCell),
            ("AutoencoderFF", RNNCell),
            ("AutoencoderFF_LN", LayerNormRNNCell),
            ("lognRNN_rollout", LayerNormRNNCell),
        ]

        for pRNNtype, expected_cell in test_cases:
            net = PredictiveNet(
                mock_env, pRNNtype=pRNNtype, hidden_size=100, **base_architecture_kwargs
            )
            actual_cell = type(net.pRNN.rnn.cell)

            assert actual_cell == expected_cell, (
                f"{pRNNtype}: Expected cell={expected_cell.__name__}, "
                f"got cell={actual_cell.__name__}"
            )

    # Masked specific presets
    def test_masked_k_values(self, mock_env, base_architecture_kwargs):
        """Test that different masked RNN variants have correct k values"""
        test_cases = [
            ("thRNN_0win", 0),
            ("thRNN_1win", 1),
            ("thRNN_5win", 5),
            ("thRNN_10win", 10),
        ]

        for pRNNtype, expected_k in test_cases:
            net = PredictiveNet(
                mock_env, pRNNtype=pRNNtype, hidden_size=100, **base_architecture_kwargs
            )
            actual_k = len(net.pRNN.inMask) - 1
            assert actual_k == expected_k, f"{pRNNtype}: Expected k={expected_k}, got k={actual_k}"

    def test_actOffset_values(self, mock_env, base_architecture_kwargs):
        """Test that _prevAct variants have correct actOffset"""
        test_cases = [
            ("thRNN_0win_prevAct", 1),
            ("thRNN_5win_prevAct", 1),
            ("thcycRNN_5win_holdc_prevAct", 1),
        ]

        for (
            pRNNtype,
            expected_actOffset,
        ) in test_cases:
            net = PredictiveNet(
                mock_env, pRNNtype=pRNNtype, hidden_size=100, **base_architecture_kwargs
            )
            assert net.pRNN.actOffset == expected_actOffset, (
                f"{pRNNtype}: Expected actOffset={expected_actOffset}, got {net.pRNN.actOffset}"
            )

    def test_masked_mask_actions(self, mock_env, base_architecture_kwargs):
        """Test that _mask variants have mask_actions=True"""
        test_cases = [
            ("thRNN_1win_mask", [True, False], 1),
            ("thRNN_5win_mask", [True, False, False, False, False, False], 5),
        ]

        for pRNNtype, expected_mask, expected_k in test_cases:
            net = PredictiveNet(
                mock_env, pRNNtype=pRNNtype, hidden_size=100, **base_architecture_kwargs
            )
            assert (net.pRNN.actMask == expected_mask).all(), (
                f"{pRNNtype}: Expected actMask={expected_mask}, got {net.pRNN.actMask}"
            )
            actual_k = len(net.pRNN.inMask) - 1
            assert actual_k == expected_k

    # Rollout Specific presets
    def test_rollout_continuousTheta(self, mock_env, base_architecture_kwargs):
        """Test that rollout variants have correct continuousTheta"""
        test_cases = [
            ("thcycRNN_5win_hold", False),
            ("thcycRNN_5win_holdc", True),
            ("thcycRNN_5win_first", False),
            ("thcycRNN_5win_firstc", True),
        ]

        for pRNNtype, expected_continuous in test_cases:
            net = PredictiveNet(
                mock_env, pRNNtype=pRNNtype, hidden_size=100, **base_architecture_kwargs
            )
            assert net.pRNN.rnn.continuousTheta == expected_continuous, (
                f"{pRNNtype}: Expected continuousTheta={expected_continuous}, got {net.pRNN.rnn.continuousTheta}"
            )

    def test_rollout_rollout_action(self, mock_env, base_architecture_kwargs):
        """Test that rollout variants have correct rollout_action"""
        test_cases = [
            ("thcycRNN_5win_hold", "hold"),  # hold = hold
            ("thcycRNN_5win_first", False),  # first = False
            ("thcycRNN_5win_full", True),  # full = True
        ]

        for pRNNtype, expected_action in test_cases:
            net = PredictiveNet(
                mock_env, pRNNtype=pRNNtype, hidden_size=100, **base_architecture_kwargs
            )
            assert net.pRNN.actionTheta == expected_action, (
                f"{pRNNtype}: Expected rollout_action={expected_action}, got {net.pRNN.actionTheta}"
            )

    def test_rollout_k_values(self, mock_env, base_architecture_kwargs):
        """Test that different masked RNN variants have correct k values"""
        test_cases = [("thcycRNN_3win", 3), ("lognRNN_rollout", 5)]

        for pRNNtype, expected_k in test_cases:
            net = PredictiveNet(
                mock_env, pRNNtype=pRNNtype, hidden_size=100, **base_architecture_kwargs
            )
            actual_k = net.pRNN.k
            assert actual_k == expected_k, f"{pRNNtype}: Expected k={expected_k}, got k={actual_k}"

    # lognRNN presets
    def test_lognRNN_init_and_sparsity(self, mock_env, base_architecture_kwargs):
        """Test that lognRNN variants have correct init and sparsity"""
        test_cases = [
            ("lognRNN_mask"),
            ("lognRNN_rollout"),
        ]

        for pRNNtype in test_cases:
            net = PredictiveNet(
                mock_env, pRNNtype=pRNNtype, hidden_size=100, **base_architecture_kwargs
            )
            # Check that log_normal init was used (weights should follow log-normal)
            assert (net.pRNN.W >= 0).all(), (
                "Expected postiive lognormal distribution, but weights are negative"
            )
            sparsity = (net.pRNN.W == 0).float().mean().item() * 100
            expected_sparsity = 95.0
            tolerance = 5.0

            assert abs(sparsity - expected_sparsity) < tolerance, (
                f"Expected ~{expected_sparsity}% zeros (sparsity=0.05), but got {sparsity:.2f}% zeros"
            )

    def test_autoencoder_use_FF(self, mock_env, base_architecture_kwargs):
        """Test that Autoencoder variants have correct use_FF setting"""
        test_cases = [
            ("AutoencoderFF", False),  # use_FF = true , therefor W does not require grad
            ("AutoencoderRec", True),  # use_FF = false , therefor W does require grad
            ("AutoencoderFFPred", False),
        ]

        for pRNNtype, expected_use_FF in test_cases:
            net = PredictiveNet(
                mock_env, pRNNtype=pRNNtype, hidden_size=100, **base_architecture_kwargs
            )
            W_found = False
            for name, param in net.pRNN.named_parameters():
                if name == "W":
                    W_found = True
                    assert param.requires_grad == expected_use_FF, (
                        f"{pRNNtype}: Expected W.requires_grad={expected_use_FF}, "
                        f"got {param.requires_grad}"
                    )
                    break

        assert W_found, f"{pRNNtype}: W parameter not found in named_parameters()"

    def test_predOffset_values(self, mock_env, base_architecture_kwargs):
        """Test that _prevAct variants have correct actOffset"""
        test_cases = [
            ("AutoencoderFF", 0),
            ("AutoencoderPred", 1),
        ]

        for (
            pRNNtype,
            expected_predOffset,
        ) in test_cases:
            net = PredictiveNet(
                mock_env, pRNNtype=pRNNtype, hidden_size=100, **base_architecture_kwargs
            )
            assert net.pRNN.predOffset == expected_predOffset, (
                f"{pRNNtype}: Expected predOffset={expected_predOffset}, got {net.pRNN.predOffset}"
            )


# # ============================================================================
# # TEST 4: CELL-SPECIFIC PARAMETERS FLOW CORRECTLY
# # ============================================================================


# class TestCellParameters:
#     """Test that cell-specific parameters reach the cell correctly"""

#     def test_divnorm_parameters_reach_cell(self, mock_env, base_architecture_kwargs):
#         """Test that DivNorm parameters are correctly passed to cell"""
#         kwargs = {
#             **base_architecture_kwargs,
#             "target_mean": 0.8,
#             "k_div": 2.0,
#             "sigma": 1.5,
#             "train_divnorm": False,
#         }

#         net = PredictiveNet(
#             mock_env, pRNNtype="Masked", hidden_size=100, cell="DivNormRNNCell", **kwargs
#         )

#         cell = net.pRNN.rnn.cell

#         # Check parameters reached the cell
#         assert hasattr(cell, "divnorm"), "Cell should have divnorm module"
#         assert hasattr(cell, "target_mean"), "Cell should have target_mean"
#         assert cell.target_mean == 0.8, f"Expected target_mean=0.8, got {cell.target_mean}"

#         # Check k_div and sigma values
#         assert cell.divnorm.k_div.item() == 2.0, (
#             f"Expected k_div=2.0, got {cell.divnorm.k_div.item()}"
#         )
#         assert cell.divnorm.sigma.item() == 1.5, (
#             f"Expected sigma=1.5, got {cell.divnorm.sigma.item()}"
#         )

#     def test_divnorm_trainable_parameters(self, mock_env, base_architecture_kwargs):
#         """Test that train_divnorm correctly makes parameters trainable"""
#         # Test with train_divnorm=True
#         kwargs_trainable = {
#             **base_architecture_kwargs,
#             "target_mean": 0.7,
#             "k_div": 1.0,
#             "sigma": 1.0,
#             "train_divnorm": True,
#         }

#         net = PredictiveNet(
#             mock_env, pRNNtype="Masked", hidden_size=100, cell="DivNormRNNCell", **kwargs_trainable
#         )

#         cell = net.pRNN.rnn.cell
#         assert isinstance(cell.divnorm.k_div, torch.nn.Parameter), (
#             "k_div should be a Parameter when train_divnorm=True"
#         )
#         assert isinstance(cell.divnorm.sigma, torch.nn.Parameter), (
#             "sigma should be a Parameter when train_divnorm=True"
#         )

#         # Test with train_divnorm=False
#         kwargs_fixed = {
#             **base_architecture_kwargs,
#             "target_mean": 0.7,
#             "k_div": 1.0,
#             "sigma": 1.0,
#             "train_divnorm": False,
#         }

#         net_fixed = PredictiveNet(
#             mock_env, pRNNtype="Masked", hidden_size=100, cell="DivNormRNNCell", **kwargs_fixed
#         )

#         cell_fixed = net_fixed.pRNN.rnn.cell
#         assert not isinstance(cell_fixed.divnorm.k_div, torch.nn.Parameter), (
#             "k_div should be a buffer when train_divnorm=False"
#         )
#         assert not isinstance(cell_fixed.divnorm.sigma, torch.nn.Parameter), (
#             "sigma should be a buffer when train_divnorm=False"
#         )

#     def test_cell_override_works(self, mock_env, base_architecture_kwargs):
#         """Test that cell override argument works"""
#         # Test with LayerNormRNNCell
#         net_ln = PredictiveNet(
#             mock_env,
#             pRNNtype="Masked",
#             hidden_size=100,
#             cell="LayerNormRNNCell",
#             **base_architecture_kwargs,
#         )

#         assert net_ln.pRNN.rnn.cell.__class__.__name__ == "LayerNormRNNCell", (
#             f"Expected LayerNormRNNCell, got {net_ln.pRNN.rnn.cell.__class__.__name__}"
#         )

#         # Test with DivNormRNNCell
#         kwargs_divnorm = {
#             **base_architecture_kwargs,
#             "target_mean": 0.7,
#             "train_divnorm": False,
#         }
#         net_dn = PredictiveNet(
#             mock_env, pRNNtype="Masked", hidden_size=100, cell="DivNormRNNCell", **kwargs_divnorm
#         )

#         assert net_dn.pRNN.rnn.cell.__class__.__name__ == "DivNormRNNCell", (
#             f"Expected DivNormRNNCell, got {net_dn.pRNN.rnn.cell.__class__.__name__}"
#         )

#     def test_sparsity_reaches_layernorm_cell(self, mock_env, base_architecture_kwargs):
#         """Test that sparsity parameter reaches LayerNormRNNCell"""
#         kwargs = {
#             **base_architecture_kwargs,
#             "sparsity": 0.3,
#         }

#         net = PredictiveNet(
#             mock_env,
#             pRNNtype="lognRNN_mask",  # Uses LayerNormRNNCell
#             hidden_size=100,
#             **kwargs,
#         )

#         cell = net.pRNN.rnn.cell
#         assert hasattr(cell, "f"), "LayerNormRNNCell should have sparsity (f)"
#         # sparsity should equal cell.f
#         assert cell.f == 0.3, f"Expected f=0.3, got {cell.f}"


# # ============================================================================
# # TEST 5: OPTIMIZER PARAMETER GROUPS
# # ============================================================================


# class TestOptimizerParameterGroups:
#     """Test that optimizer parameter groups are correctly configured"""

#     def test_divnorm_in_optimizer_when_trainable(self, mock_env, base_architecture_kwargs):
#         """Test that k_div and sigma are in optimizer when train_divnorm=True"""
#         kwargs = {
#             **base_architecture_kwargs,
#             "target_mean": 0.7,
#             "train_divnorm": True,
#         }

#         net = PredictiveNet(
#             mock_env,
#             pRNNtype="Masked",
#             hidden_size=100,
#             cell="DivNormRNNCell",
#             trainBias=False,
#             **kwargs,
#         )

#         # Check optimizer has k_div and sigma groups
#         param_group_names = [g["name"] for g in net.optimizer.param_groups]
#         assert "k_divnorm" in param_group_names, "k_divnorm should be in optimizer parameter groups"
#         assert "sigma_divnorm" in param_group_names, (
#             "sigma_divnorm should be in optimizer parameter groups"
#         )

#     def test_divnorm_not_in_optimizer_when_fixed(self, mock_env, base_architecture_kwargs):
#         """Test that k_div and sigma are NOT in optimizer when train_divnorm=False"""
#         kwargs = {
#             **base_architecture_kwargs,
#             "target_mean": 0.7,
#             "train_divnorm": False,
#         }

#         net = PredictiveNet(
#             mock_env,
#             pRNNtype="Masked",
#             hidden_size=100,
#             cell="DivNormRNNCell",
#             trainBias=False,
#             **kwargs,
#         )

#         # Check optimizer does NOT have k_div and sigma groups
#         param_group_names = [g["name"] for g in net.optimizer.param_groups]
#         assert "k_divnorm" not in param_group_names, (
#             "k_divnorm should NOT be in optimizer when train_divnorm=False"
#         )
#         assert "sigma_divnorm" not in param_group_names, (
#             "sigma_divnorm should NOT be in optimizer when train_divnorm=False"
#         )

#     def test_bias_in_optimizer_when_trainable(self, mock_env, base_architecture_kwargs):
#         """Test that bias is in optimizer when trainBias=True"""
#         net = PredictiveNet(
#             mock_env, pRNNtype="Masked", hidden_size=100, trainBias=True, **base_architecture_kwargs
#         )

#         param_group_names = [g["name"] for g in net.optimizer.param_groups]
#         assert "biases" in param_group_names, (
#             "biases should be in optimizer parameter groups when trainBias=True"
#         )

#     def test_eg_parameter_groups_configured(self, mock_env, base_architecture_kwargs):
#         """Test that EG is correctly configured for parameter groups"""
#         kwargs = {
#             **base_architecture_kwargs,
#             "target_mean": 0.7,
#             "train_divnorm": True,
#         }

#         net = PredictiveNet(
#             mock_env,
#             pRNNtype="Masked",
#             hidden_size=100,
#             cell="DivNormRNNCell",
#             eg_lr=1e-3,
#             eg_weight_decay=1e-6,
#             **kwargs,
#         )

#         # Check that positive parameter groups have update_alg="eg"
#         for group in net.optimizer.param_groups:
#             if group["name"] in ["RecurrentWeights", "k_divnorm", "sigma_divnorm"]:
#                 # These should have EG (assuming they're all positive)
#                 assert group.get("update_alg") == "eg", (
#                     f"{group['name']} should use EG update algorithm"
#                 )
#                 assert group["lr"] == 1e-3, f"{group['name']} should have eg_lr=1e-3"
#                 # Check weight decay scaling
#                 expected_wd = 1e-6 * 1e-3  # eg_weight_decay * eg_lr
#                 assert abs(group["weight_decay"] - expected_wd) < 1e-12, (
#                     f"{group['name']} weight_decay should be {expected_wd}"
#                 )

# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
