"""Regression tests against measurements taken on the reference machine.

These lock the predictor to reality. The inputs are the tensor-derived cost
figures for DeepSeek-V4-Flash-0731 (MXFP4) and the outputs are throughput and
VRAM figures actually recorded on an EPYC 7B12 + RTX 3090, so a change that
improves the model on paper but drifts away from measurement will fail here.

Recorded 2026-09-08, llama.cpp f114f91, 3 repetitions per configuration:

  ncmoe=43 + 10.9 GB draft:  9.89 tok/s, 20570 MiB peak VRAM
  ncmoe=42 + 10.9 GB draft: 10.10 tok/s, 23834 MiB peak VRAM
  ncmoe=37 without a draft:  failed with `unable to allocate CUDA0 buffer`
"""
import pytest

from wirl import predict
from wirl.model import LayerCost, ModelCost

MiB = 1 << 20
GPU_BW = 936e9              # RTX 3090
MEASURED_RAM_BW = 45e9      # streaming, model cached, measured that day
VRAM_TOTAL = 24576 * MiB
CTX = 16384

DRAFT_BYTES = 10_890_804_124
DRAFT_KV = 100_663_296
TARGET_KV = 1_442_840_576


@pytest.fixture
def dsv4():
    """Cost model for the reference model, from its real tensor table."""
    expert = 3_422_552_064
    other = 152_610_264
    layers = [LayerCost(i, expert, int(expert * 6 / 256), other) for i in range(43)]
    return ModelCost(n_layer=43, n_expert=256, n_expert_used=6, layers=layers,
                     non_layer_bytes=1_621_835_796,
                     non_layer_bytes_per_tok=562_774_036,
                     total_bytes=155_971_123_548)


def _vram(dsv4, ncmoe):
    return (dsv4.resident_vram_weights(ncmoe) + TARGET_KV
            + DRAFT_BYTES + DRAFT_KV + predict.CUDA_OVERHEAD)


@pytest.mark.parametrize("ncmoe,measured_mib", [(43, 20570), (42, 23834)])
def test_vram_prediction_matches_nvidia_smi(dsv4, ncmoe, measured_mib):
    predicted = _vram(dsv4, ncmoe)
    measured = measured_mib * MiB
    err = abs(predicted - measured) / measured
    assert err < 0.03, f"predicted {predicted/MiB:.0f} MiB vs measured {measured_mib}"


@pytest.mark.parametrize("ncmoe,measured_tps", [(43, 9.89), (42, 10.10)])
def test_throughput_prediction_matches_measurement(dsv4, ncmoe, measured_tps):
    tps, _, _ = predict.predict_tps(dsv4, ncmoe, MEASURED_RAM_BW, GPU_BW)
    err = abs(tps - measured_tps) / measured_tps
    assert err < 0.10, f"predicted {tps:.2f} tok/s vs measured {measured_tps}"


def test_predicts_the_config_that_actually_failed(dsv4):
    """ncmoe=37 must be reported as not fitting, with no draft model loaded."""
    vram = (dsv4.resident_vram_weights(37) + TARGET_KV + predict.CUDA_OVERHEAD)
    assert vram > VRAM_TOTAL


def test_lower_ncmoe_is_faster_but_needs_more_vram(dsv4):
    """The core trade-off must be monotonic in both directions."""
    prev_tps, prev_vram = 0.0, 0.0
    for n in range(43, 35, -1):
        tps, _, _ = predict.predict_tps(dsv4, n, MEASURED_RAM_BW, GPU_BW)
        vram = _vram(dsv4, n)
        assert tps > prev_tps
        assert vram > prev_vram
        prev_tps, prev_vram = tps, vram


def test_headroom_rejects_the_tightest_fit(dsv4):
    """A config that fits with 0.5 GiB spare must not be recommended.

    On the reference machine the peak-throughput config sat at 97% of VRAM and
    failed to allocate several thousand tokens into a real conversation.
    """
    pts = [predict.Prediction(n, 0, 0, _vram(dsv4, n), _vram(dsv4, n) <= VRAM_TOTAL,
                              predict.predict_tps(dsv4, n, MEASURED_RAM_BW, GPU_BW)[0],
                              0, 0)
           for n in range(43, 39, -1)]
    with_headroom = [p for p in pts if p.vram_bytes <= VRAM_TOTAL - 3e9]
    assert [p.n_cpu_moe for p in with_headroom] == [43]


def test_verdict_says_stop_when_at_the_wall(dsv4):
    best = predict.Prediction(43, 0, 0, 0, True, 9.61, 94.0, 8.0)
    lines = " ".join(predict.verdict(dsv4, best, MEASURED_RAM_BW, measured_tps=9.89))
    assert "nothing left to tune" in lines
    assert "Bandwidth-bound" in lines


def test_verdict_flags_a_machine_performing_far_below_roofline(dsv4):
    best = predict.Prediction(43, 0, 0, 0, True, 10.0, 94.0, 8.0)
    lines = " ".join(predict.verdict(dsv4, best, MEASURED_RAM_BW, measured_tps=4.0))
    assert "Something is wrong" in lines


def test_no_gpu_falls_back_to_pure_cpu_roofline(dsv4):
    tps = predict.cpu_only_tps(dsv4, MEASURED_RAM_BW)
    hybrid, _, _ = predict.predict_tps(dsv4, 43, MEASURED_RAM_BW, GPU_BW)
    assert tps < hybrid


def test_cpu_only_matches_the_measured_penalty(dsv4):
    """CPU-only was measured at 2.21 tok/s against 8.79 for a hybrid split."""
    tps = predict.cpu_only_tps(dsv4, MEASURED_RAM_BW)
    assert 1.8 < tps < 2.8, f"got {tps:.2f} tok/s, measured was 2.21"


def test_cpu_only_prices_attention_bytes_too(dsv4):
    """--n-cpu-moe never moves attention or shared experts off the GPU, so
    'all experts on CPU' is not the same thing as 'no GPU'."""
    no_gpu = predict.cpu_only_tps(dsv4, MEASURED_RAM_BW)
    all_experts_on_cpu, _, _ = predict.predict_tps(dsv4, 43, MEASURED_RAM_BW, GPU_BW)
    assert no_gpu < all_experts_on_cpu


def test_beating_the_roofline_is_reported_as_a_bad_estimate(dsv4):
    """Measured 10.43 tok/s against a 5.17 roofline on the reference machine.

    The old logic said "nothing left to tune" for anything above 90% of the
    roofline, so 202% printed a confident falsehood. Exceeding the roofline
    means the estimate is wrong -- hot experts stay in cache and are not
    re-read from DRAM every token.
    """
    best = predict.Prediction(43, 0, 0, 0, True, 5.17, 96.0, 4.0)
    lines = " ".join(predict.verdict(dsv4, best, 23e9, measured_tps=10.43))
    assert "EXCEEDS" in lines
    assert "estimate is wrong" in lines
    assert "nothing left to tune" not in lines


def test_no_bandwidth_bound_claim_when_measurement_contradicts_it(dsv4):
    """Do not tell someone they are stuck at the memory wall while measuring
    twice the throughput that wall allows."""
    best = predict.Prediction(43, 0, 0, 0, True, 5.17, 96.0, 4.0)
    lines = " ".join(predict.verdict(dsv4, best, 23e9, measured_tps=10.43))
    assert "Bandwidth-bound" not in lines


def test_at_the_wall_still_says_stop(dsv4):
    best = predict.Prediction(43, 0, 0, 0, True, 9.61, 94.0, 8.0)
    lines = " ".join(predict.verdict(dsv4, best, 45e9, measured_tps=9.89))
    assert "nothing left to tune" in lines
    assert "EXCEEDS" not in lines


def test_gpu_bound_does_not_raise_a_false_alarm(dsv4):
    """A dense model fully on the GPU measured 340 tok/s against a 1106 tok/s
    roofline, and the tool announced "something is wrong". Nothing was: at
    2.9 ms/token the cost is kernel launch and attention, not bandwidth.
    """
    best = predict.Prediction(28, 0, 0, 0, True, 1106.0, 1.0, 99.0)
    lines = " ".join(predict.verdict(dsv4, best, 23e9, measured_tps=340.0))
    assert "Something is wrong" not in lines
    assert "GPU-bound" in lines


def test_cpu_bound_underperformance_still_warns(dsv4):
    best = predict.Prediction(43, 0, 0, 0, True, 10.0, 95.0, 5.0)
    lines = " ".join(predict.verdict(dsv4, best, 45e9, measured_tps=4.0))
    assert "Something is wrong" in lines
