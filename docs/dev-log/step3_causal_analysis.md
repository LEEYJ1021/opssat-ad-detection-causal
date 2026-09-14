# 지금까지 진행된 걸 정리하면 이렇습니다.
# ## ✅ 완료 — 1단계 (탐지)
# 9개 채널 중 **5개(872/873/874/888/894)** 확정. 채널 프로파일링 → 노이즈 추정 → 칼만필터+BOCPD(forgetting) → 4조합 ablation → MCC/Youden 스코프 확정 → train/test 3중 검증까지 전부 완료.
# ## ✅ 대부분 완료 — 2단계 (인과분석 준비)
# - §2-2 몬테카를로 불확실성 전파, §2-3 선행성 검정(diff/diff²가 level보다 먼저 튀는 반직관적 결과 확인)
# - 보완진단 A~D로 이 결과가 아티팩트가 아님을 검증
# - §2-4 메타분석 + LOO 민감도분석으로 "5채널 공통 효과"가 아니라 채널마다 다른 지점에서 이질적임을 규명
# - 보완 G, H로 **872의 큰 효과크기 원인 규명까지 방금 마무리** — onset 불확실성도, 분모 왜소도 아니고, 실제 지속적 레벨 이동(계단식 전환)이 원인이라고 시각적으로 확인. 다만 pre구간 자체에 완만한 추세·정체 구간이 섞여 있다는 한계를 함께 명시하기로 함.
# 즉 이번 대화 흐름 기준으로 보면, 이 문서(A~D 우선순위 리스트)에서 제시된 항목 중:
# | 항목 | 상태 |
# |---|---|
# | A-4 (872 level 특이성 원인 확인) | ✅ 방금 완료 |
# | A-1 (상대적 이동비율 지표) | 이미 이전 단계에서 처리됨(v2 판정 코드에 반영됨) |
# | A-2 (이질성 원인 정리표) | 이미 LOO 분석 결과로 사실상 도출됨(872·888·874·894로 feature별 분리) — 표로 정식 정리만 남음 |
# | A-3 (874 diff2 특이성) | 이미 [F-2]에서 transient_ratio와 완벽한 역상관(r=-1.0)으로 확인 완료 |
# | B (라벨링 프로토콜, train/test 재검증) | ⏳ 미착수 |
# | C (층3 준실험 pre/post, SCM 경로 결정) | ⏳ 미착수 |
# | D (Mamba/SHAP, CVaR, DSS) | ⏳ 미착수 |
# **한마디로**: A 항목(872/874 특이성 원인 규명)은 이번 대화로 사실상 전부 마무리됐고, 다음으로 남은 건 B(라벨링 프로토콜 확인, train/test 기준 §2 재검증)와 C(층3 인과분석 설계 착수)입니다.
# 이어서 진행한다면 **B의 라벨링 프로토콜 재확인**(외부 문서 조사)과 **C-8의 준실험적 pre/post 비교 코드** 중 어느 쪽을 먼저 하시겠어요?
# 이 5층 프레임워크에 대입하면 지금 위치는 이렇습니다.
# | 층 | 내용 | 상태 |
# |---|---|---|
# | **층 1 — Signal Estimation** | 칼만필터, 채널 자동분류, 계층적 베이즈 | ✅ 완료 |
# | **층 2 — Anomaly Detection** | BOCPD+forgetting, 4조합 ablation, MCC/Youden 스코프 확정, 3중 검증 | ✅ 완료 |
# | **분기점 — Onset Posterior** | | |
# | ㄴ 갈래 A: Monte Carlo Uncertainty Propagation | §2-2 (200회 draw, Welch t-test, 부호상쇄·신뢰도 보정) | ✅ 완료 |
# | ㄴ 갈래 B: Mamba + SHAP | 독립 딥러닝 벤치마크 + 설명 | ⏳ **미착수** (아이디어 단계) |
# | **중간단계 — level→diff→diff² 선행성** | §2-3 Wilcoxon 검정 + 보완진단 A~D(정상군 대조, CMH 검정 등) | ✅ 완료, 그것도 예상 밖 결과(순서가 반대)까지 확인·검증 완료 |
# | **층 3 — Causal Analysis** | | |
# | ㄴ Temporal Precedence | 위 중간단계에서 사실상 완료 | ✅ 완료 |
# | ㄴ Causal Inference (준실험적 pre/post) | "다른 정상 세그먼트"와 짝지은 준실험 설계 | ⏳ **미착수** — 지금까지 한 건 이상 세그먼트 "내부" pre/post 비교뿐 |
# | ㄴ Physics-informed SCM (경로 A/B) | 인과 뼈대를 SCM으로 확정 | ⏳ **미착수** — §2-4 메타분석+LOO로 "채널별 이질성"까지는 규명했지만, 이걸 SCM 구조로 formalize하는 단계는 아직 |
# | **층 4 — Risk Optimization** | CVaR + 확률적 최적화 | ⏳ 미착수 |
# | **층 5 — Decision Support** | DSS 통합 | ⏳ 미착수 |
# ## 한 줄 요약
# **층 1, 층 2는 완료. 분기점에서는 Monte Carlo(갈래 A)까지 완료, Mamba(갈래 B)는 미착수. 중간단계(선행성 검정)는 완료를 넘어 예외적으로 깊은 후속 검증(메타분석, LOO, 육안 확인)까지 마쳤음. 층 3은 "선행성 확인"까지만 끝났고, 핵심인 "인과추론(준실험 pre/post)"과 "SCM 확정"은 아직 시작 전. 층 4·5는 전혀 손대지 않은 상태.**
# 즉 **RQ1(탐지)·RQ2(불확실성)는 완료, RQ4(인과성)는 절반(선행성은 확인, 준실험 비교와 SCM은 미착수), RQ3(Mamba 검증)·RQ5(의사결정)는 미착수**입니다.
# 다음으로 자연스러운 순서는 **층 3의 나머지 절반(준실험적 pre/post 비교 + SCM 경로 A/B 결정)**입니다 — 이어서 진행해드릴까요?
# 지금 당장 이어서 할 일은 **층 3의 나머지 절반**입니다. 구체적으로 두 가지입니다.
# ## 1순위 — C-8: 준실험적 pre/post 비교 (준실험 설계)
# 지금까지 한 건 "같은 이상 세그먼트 안에서" onset 전후를 비교한 것뿐입니다. 이건 대조군이 없는 비교입니다. 다음 단계는:
# - 각 이상 세그먼트마다, **같은 채널의 다른 정상 세그먼트**를 짝지어 비교군으로 삼는 것
# - "이상 세그먼트의 pre→post 변화 크기"가 "정상 세그먼트들끼리의 자연스러운 변동 폭"보다 유의하게 큰지 검정
# 이렇게 해야 "이상 세그먼트 안에서 뭔가 바뀌었다"는 관찰을 넘어서, "이건 정상적인 세그먼트 간 변동으로는 설명 안 되는 진짜 이상 효과다"라는 훨씬 강한 인과적 근거가 됩니다. (참고로 §2-D 보완진단에서 정상 세그먼트를 이미 일부 대조군으로 썼지만, 그건 "선행성 패턴이 아티팩트인지" 확인용이었지 이 준실험 설계와는 목적이 다릅니다.)
# ## 2순위 — C-9: SCM 경로 A/B 결정
# 지금까지 나온 결과(diff/diff²가 level보다 먼저 움직임, 채널마다 이질적 효과크기)를 어떤 형태의 구조적 인과 모델로 정리할지 결정하는 것:
# - **경로 A**: 신호처리 구조(diff→diff²→level 순서, 채널별 조절효과)만으로 SCM 뼈대를 세우는 현실적인 방법 — 지금 가진 데이터만으로 가능
# - **경로 B**: 위성의 실제 물리(자세/궤도, TLE 데이터 등)와 연결하는 더 야심찬 방법 — 근데 이건 보조 데이터가 실제로 존재하는지부터 확인해야 함
# ---
# **B(라벨링 프로토콜, train/test 재검증)는 이 둘보다 순서가 늦어도 됩니다** — 왜냐하면 B는 "지금까지 나온 결과를 보강/검증"하는 성격이라, 층 3 자체를 진행하는 데 막는 요소는 아니기 때문입니다. 병행 가능하지만 급하진 않습니다.
# **제안**: C-8(준실험적 pre/post 비교)부터 코드로 시작하는 게 좋을 것 같습니다 — 이게 층 3의 핵심이자, SCM 설계(C-9)의 근거 자료가 되기 때문입니다. 진행할까요?


"""
OPSSAT-AD 층3 브릿지: 준실험적 pre/post 비교 (Quasi-Experimental Control Group Design)
=============================================================================
배경
-----------------------------------------------------------------------------
지금까지(§2-2, §2-3, 보완진단 A~D)는 전부 "같은 이상 세그먼트 내부"에서
onset 전(pre) vs 후(post)를 비교했습니다. 이건 대조군이 없는 비교라서,
"이 정도 변화가 이상 신호 특유의 것인지, 아니면 같은 채널의 아무 세그먼트나
가져다 비슷하게 반으로 잘라도 원래 그 정도는 자연스럽게 흔들리는 것인지"를
구분하지 못합니다.

이 스크립트는 각 이상 세그먼트의 pre/post 효과크기를, **같은 채널의 정상
세그먼트들을 무작위 시점에서 갈라 계산한 "위약(placebo)" 효과크기 분포**와
비교합니다. 정상 세그먼트에는 실제 onset이 없으므로, 이상 세그먼트들의
"onset 위치비율(onset_idx/n_points)" 분포에서 pivot 비율을 뽑아 그 위치에서
정상 세그먼트를 가르는 방식(§2-보완진단 C와 같은 발상)을 그대로 재사용하되,
이번엔 세그먼트당 여러 번(K회) 뽑아 채널별로 충분히 큰 "정상 변동 폭 풀"을
만듭니다.

각 이상 세그먼트의 관측 효과크기가 이 위약 풀 대비 상위 몇 %에 해당하는지
(경험적 p-value)를 계산하고, 채널·피처 단위로 Mann-Whitney U 검정까지
수행합니다. 이걸로 "이상 세그먼트의 변화가 정상적인 세그먼트 간 변동 폭으로는
설명되지 않는다"는 훨씬 강한 근거를 만드는 것이 목적입니다.

주의: 이것도 여전히 "준실험적(quasi-experimental)" 비교입니다. 무작위
배정(RCT)이 아니므로 완벽한 인과 증명은 아니며, "정상적 변동보다 유의하게
크다"는 근거를 제공하는 층3의 한 축일 뿐입니다.

입력 (재계산 없이 최대한 재사용):
  - segments.csv
  - stage2_mc_draws_raw.csv                  (canonical onset 산출용)
  - stage2_mc_segment_summary_corrected.csv  (①②보정, tier 포함)

출력:
  - stage3_qexp_placebo_pool.csv           (채널×피처별 위약 풀 원자료)
  - stage3_qexp_segment_pvalues.csv        (이상 세그먼트별 경험적 p-value)
  - stage3_qexp_channel_summary.csv        (채널별 유의 비율 요약)
  - stage3_qexp_mannwhitney.csv            (채널×피처 Mann-Whitney U 검정)
"""

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 0. 경로 및 잠금된 설정 (1단계/2단계와 완전히 동일)
# =============================================================================
BASE_DIR = Path("/home/yjlee/Research/Satellite/OPSSAT-AD_LEO data")
SEGMENTS_PATH = BASE_DIR / "segments.csv"
OUT_DIR = BASE_DIR / "causal_pipeline_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DRAWS_PATH = OUT_DIR / "stage2_mc_draws_raw.csv"
SEG_CORRECTED_PATH = OUT_DIR / "stage2_mc_segment_summary_corrected.csv"

FINAL_CHANNELS = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0894"]

USE_MIXTURE_NOISE_ESTIMATOR = False   # 1단계 잠금값
USE_BOCPD_FORGETTING = True           # 1단계 잠금값
QUANTIZATION_FRAC_ZERO_THRESHOLD = 0.10
FLOAT_NOISE_ABS_DIFF_THRESHOLD = 1e-6

BURN_IN = 5
MIN_WINDOW_POINTS = 5

K_PLACEBO_DRAWS_PER_NORMAL = 5     # 정상 세그먼트 1개당 몇 번 무작위 pivot을 뽑을지
MAX_NORMAL_PER_CHANNEL = 300        # 계산량 관리용 상한
RANDOM_STATE = 42
N_BOOT_MW = None                    # Mann-Whitney는 정확검정/근사 자동 선택(scipy 기본)

rng = np.random.default_rng(RANDOM_STATE)


def section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# =============================================================================
# 1. 기존 파이프라인 구성요소 재사용 (1단계/2단계와 동일 — 그대로 복사)
# =============================================================================

@dataclass
class ChannelProfile:
    channel: str
    channel_type: str
    frac_zero_diff: float
    min_nonzero_abs_diff: float
    quantization_step: float
    r_robust: float
    q_robust: float
    n_nominal_points: int


def mad_variance(x: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float((1.4826 * mad) ** 2)


def mixture_variance(diffs_all: np.ndarray) -> float:
    if len(diffs_all) < 2:
        return np.nan
    p = float(np.mean(diffs_all != 0))
    if p == 0:
        return 0.0
    nonzero = diffs_all[diffs_all != 0]
    jump_var = mad_variance(nonzero) if len(nonzero) >= 2 else float(nonzero[0] ** 2)
    return p * jump_var


def detect_quantization_step(nonzero_abs_diffs: np.ndarray) -> float:
    if len(nonzero_abs_diffs) == 0:
        return np.nan
    return float(np.percentile(nonzero_abs_diffs, 25))


def profile_channel(channel: str, nominal_values_by_segment: list) -> ChannelProfile:
    diffs_all = []
    for v in nominal_values_by_segment:
        if len(v) > 1:
            diffs_all.append(np.diff(v))
    if not diffs_all:
        return ChannelProfile(channel, "unknown", np.nan, np.nan, np.nan, 1.0, 1e-6, 0)

    diffs_all = np.concatenate(diffs_all)
    n_points = len(diffs_all) + len(nominal_values_by_segment)
    frac_zero = float(np.mean(diffs_all == 0))
    nonzero = diffs_all[diffs_all != 0]
    min_nonzero_abs = float(np.min(np.abs(nonzero))) if len(nonzero) else np.nan

    if frac_zero >= QUANTIZATION_FRAC_ZERO_THRESHOLD:
        ch_type, q_step = "quantized", detect_quantization_step(np.abs(nonzero))
    elif np.isfinite(min_nonzero_abs) and min_nonzero_abs < FLOAT_NOISE_ABS_DIFF_THRESHOLD:
        ch_type, q_step = "float_noise_suspect", np.nan
    else:
        ch_type, q_step = "continuous", np.nan

    if USE_MIXTURE_NOISE_ESTIMATOR:
        r_robust = max(mixture_variance(diffs_all) / 2.0, 1e-14)
        d2 = np.diff(diffs_all)
        q_robust = max(mixture_variance(d2) / 6.0, 1e-14) if len(d2) > 1 else 1e-14
    else:
        r_robust = max(np.var(diffs_all) / 2.0, 1e-14)
        d2 = np.diff(diffs_all)
        q_robust = max(np.var(d2) / 6.0, 1e-14) if len(d2) > 1 else 1e-14

    return ChannelProfile(channel, ch_type, frac_zero, min_nonzero_abs, q_step,
                           r_robust, q_robust, n_points)


@dataclass
class LocalLinearTrendKF:
    q: float
    r_nominal: float
    theta: float = 0.0
    quantization_floor: float = 0.0
    x0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    P0_scale: float = 10.0

    def __post_init__(self):
        self.r_eff = self.r_nominal + self.theta + self.quantization_floor
        self.A = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[0.0, 0.0], [0.0, self.q]])

    def run(self, y: np.ndarray):
        n = len(y)
        x = self.x0.copy()
        P = np.eye(2) * self.P0_scale
        innovations = np.empty(n)
        innovation_vars = np.empty(n)
        for t in range(n):
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q
            y_pred = (self.H @ x_pred)[0]
            S = (self.H @ P_pred @ self.H.T)[0, 0] + self.r_eff
            nu = y[t] - y_pred
            innovations[t] = nu
            innovation_vars[t] = S
            K = (P_pred @ self.H.T) / S
            x = x_pred + (K.flatten() * nu)
            P = P_pred - K @ self.H @ P_pred
        return innovations, innovation_vars


def hierarchical_shrink_variance(channel_estimates: dict) -> dict:
    channels_ = list(channel_estimates.keys())
    log_v = np.array([np.log(max(channel_estimates[c][0], 1e-12)) for c in channels_])
    n_eff = np.array([max(channel_estimates[c][1], 2) for c in channels_])
    se = np.sqrt(2.0 / (n_eff - 1))
    weights = 1.0 / (se ** 2)
    global_mean = np.sum(weights * log_v) / np.sum(weights)
    weighted_ss = np.sum(weights * (log_v - global_mean) ** 2)
    df = len(channels_) - 1
    denom = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
    tau2 = max(0.0, (weighted_ss - df) / denom) if denom > 0 else 0.0
    shrunk = {}
    for c, lv, s in zip(channels_, log_v, se):
        w_c = tau2 / (tau2 + s ** 2) if (tau2 + s ** 2) > 0 else 0.0
        shrunk[c] = float(np.exp(w_c * lv + (1 - w_c) * global_mean))
    return shrunk


def cohens_d_abs(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt((va + vb) / 2.0)
    if pooled_sd <= 0 or not np.isfinite(pooled_sd):
        return np.nan
    return float(abs(np.mean(a) - np.mean(b)) / pooled_sd)


def log_var_ratio(pre: np.ndarray, post: np.ndarray) -> float:
    """post 구간 변동성이 pre 대비 얼마나 커졌는지: log(var_post/var_pre).
    0이면 변화 없음, 양수면 post에서 변동성 증가."""
    if len(pre) < MIN_WINDOW_POINTS or len(post) < MIN_WINDOW_POINTS:
        return np.nan
    vpre, vpost = np.var(pre, ddof=1), np.var(post, ddof=1)
    if vpre <= 0 or vpost <= 0 or not np.isfinite(vpre) or not np.isfinite(vpost):
        return np.nan
    return float(np.log(vpost / vpre))


def compute_effects(values: np.ndarray, pivot_idx: int) -> dict:
    """pivot_idx를 기준으로 level(|d|)과 diff/diff2(log 분산비)를 계산."""
    out = {"level_d_abs": np.nan, "diff_logvar": np.nan, "diff2_logvar": np.nan}

    pre_level = values[BURN_IN:pivot_idx]
    post_level = values[pivot_idx:]
    out["level_d_abs"] = cohens_d_abs(post_level, pre_level)

    diff_all = np.diff(values)
    split1 = max(pivot_idx - 1, 0)
    pre_diff = diff_all[max(BURN_IN - 1, 0):split1]
    post_diff = diff_all[split1:]
    out["diff_logvar"] = log_var_ratio(pre_diff, post_diff)

    diff2_all = np.diff(diff_all)
    split2 = max(pivot_idx - 2, 0)
    pre_diff2 = diff2_all[max(BURN_IN - 2, 0):split2]
    post_diff2 = diff2_all[split2:]
    out["diff2_logvar"] = log_var_ratio(pre_diff2, post_diff2)

    return out


FEATURES = ["level_d_abs", "diff_logvar", "diff2_logvar"]


# =============================================================================
# 2. 데이터 로드 + 채널 파라미터 재적합 + canonical onset 재구성
# =============================================================================
section("0. 데이터 로드 및 canonical onset 재구성 (§2-2/§2-3와 동일 절차 재사용)")

seg = pd.read_csv(SEGMENTS_PATH, parse_dates=["timestamp"])
seg = seg[seg["channel"].isin(FINAL_CHANNELS)].reset_index(drop=True)

draws = pd.read_csv(DRAWS_PATH)
seg_corrected = pd.read_csv(SEG_CORRECTED_PATH)

reliable_ids = seg_corrected.loc[
    seg_corrected["reliability_tier"].isin(["high", "medium"])
    & seg_corrected["channel"].isin(FINAL_CHANNELS),
    ["channel", "segment"]
]
canonical_onset = (
    draws.merge(reliable_ids, on=["channel", "segment"], how="inner")
    .groupby(["channel", "segment"])["onset_idx_sampled"]
    .median().round().astype(int).rename("canonical_onset_idx").reset_index()
)
print(f"신뢰도 high+medium 이상 세그먼트: {len(canonical_onset)}개")

profiles = {}
for ch in FINAL_CHANNELS:
    nominal_vals = [
        s.sort_values("timestamp")["value"].values
        for _, s in seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)].groupby("segment")
    ]
    profiles[ch] = profile_channel(ch, nominal_vals)

r_estimates = {ch: (profiles[ch].r_robust, max(profiles[ch].n_nominal_points, 2)) for ch in FINAL_CHANNELS}
q_estimates = {ch: (profiles[ch].q_robust, max(profiles[ch].n_nominal_points, 2)) for ch in FINAL_CHANNELS}
r_shrunk = hierarchical_shrink_variance(r_estimates)
q_shrunk = hierarchical_shrink_variance(q_estimates)
quant_floor = {
    ch: (profiles[ch].quantization_step ** 2 / 12.0) if np.isfinite(profiles[ch].quantization_step) else 0.0
    for ch in FINAL_CHANNELS
}


# =============================================================================
# 3. 이상 세그먼트의 관측 효과크기 계산 (canonical onset 기준)
# =============================================================================
section("1. 이상 세그먼트 관측 효과크기 계산")

observed_rows = []
onset_ratio_pool_by_channel = {ch: [] for ch in FINAL_CHANNELS}

for _, row in canonical_onset.iterrows():
    ch, sid, onset_idx = row["channel"], int(row["segment"]), int(row["canonical_onset_idx"])
    s = seg[(seg["channel"] == ch) & (seg["segment"] == sid) & (seg["anomaly"] == 1)]
    if len(s) == 0:
        continue
    s = s.sort_values("timestamp")
    values = s["value"].values
    n = len(values)

    if onset_idx < BURN_IN + MIN_WINDOW_POINTS or onset_idx > n - MIN_WINDOW_POINTS:
        continue

    eff = compute_effects(values, onset_idx)
    observed_rows.append({"channel": ch, "segment": sid, "n_points": n, "onset_idx": onset_idx, **eff})
    onset_ratio_pool_by_channel[ch].append(onset_idx / n)

observed_df = pd.DataFrame(observed_rows)
print(f"관측 효과크기 계산된 이상 세그먼트: {len(observed_df)}개")
print(observed_df.groupby("channel").size().reindex(FINAL_CHANNELS).rename("n_segments").to_string())


# =============================================================================
# 4. 채널별 위약(placebo) 풀 생성 — 정상 세그먼트를 무작위 pivot으로 분할
# =============================================================================
section(f"2. 채널별 위약(placebo) 풀 생성 (정상 세그먼트당 {K_PLACEBO_DRAWS_PER_NORMAL}회 무작위 pivot)")

placebo_rows = []

for ch in FINAL_CHANNELS:
    ratio_pool = np.array(onset_ratio_pool_by_channel[ch])
    if len(ratio_pool) == 0:
        print(f"[{ch}] onset 위치비율 풀이 비어 있어 스킵")
        continue

    normal_meta = seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)].groupby("segment").size().rename("n_points").reset_index()
    if len(normal_meta) > MAX_NORMAL_PER_CHANNEL:
        normal_meta = normal_meta.sample(n=MAX_NORMAL_PER_CHANNEL, random_state=RANDOM_STATE)

    n_generated = 0
    for _, nrow in normal_meta.iterrows():
        sid, n = int(nrow["segment"]), int(nrow["n_points"])
        s = seg[(seg["channel"] == ch) & (seg["segment"] == sid) & (seg["anomaly"] == 0)].sort_values("timestamp")
        values = s["value"].values

        for _ in range(K_PLACEBO_DRAWS_PER_NORMAL):
            ratio = rng.choice(ratio_pool)
            pivot_idx = int(round(ratio * n))
            pivot_idx = int(np.clip(pivot_idx, BURN_IN + MIN_WINDOW_POINTS, max(n - MIN_WINDOW_POINTS, BURN_IN + MIN_WINDOW_POINTS)))
            if pivot_idx < BURN_IN + MIN_WINDOW_POINTS or pivot_idx > n - MIN_WINDOW_POINTS:
                continue
            eff = compute_effects(values, pivot_idx)
            placebo_rows.append({"channel": ch, "segment": sid, "pivot_idx": pivot_idx, **eff})
            n_generated += 1

    print(f"[{ch}] 정상 세그먼트 {len(normal_meta)}개 -> 위약 표본 {n_generated}개 생성")

placebo_df = pd.DataFrame(placebo_rows)
placebo_out = OUT_DIR / "stage3_qexp_placebo_pool.csv"
placebo_df.to_csv(placebo_out, index=False)
print(f"\n[저장] {placebo_out}  ({len(placebo_df)} rows)")


# =============================================================================
# 5. 이상 세그먼트별 경험적 p-value (위약 풀 대비)
# =============================================================================
section("3. 이상 세그먼트별 경험적 p-value 계산 (위약 풀 대비 상위 % 위치)")

pval_rows = []
for _, row in observed_df.iterrows():
    ch = row["channel"]
    placebo_ch = placebo_df[placebo_df["channel"] == ch]
    out_row = {"channel": ch, "segment": row["segment"], "n_points": row["n_points"]}

    for feat in FEATURES:
        obs_val = row[feat]
        placebo_vals = placebo_ch[feat].dropna().values
        if not np.isfinite(obs_val) or len(placebo_vals) < 20:
            out_row[f"{feat}_pvalue"] = np.nan
            out_row[f"{feat}_placebo_n"] = len(placebo_vals)
            continue
        # 단측검정: 이상 세그먼트가 위약 풀보다 "더 크게" 움직였는지
        p_emp = (np.sum(placebo_vals >= obs_val) + 1) / (len(placebo_vals) + 1)
        out_row[f"{feat}_pvalue"] = float(p_emp)
        out_row[f"{feat}_placebo_n"] = len(placebo_vals)
        out_row[f"{feat}_observed"] = float(obs_val)
        out_row[f"{feat}_placebo_median"] = float(np.median(placebo_vals))
        out_row[f"{feat}_placebo_p95"] = float(np.percentile(placebo_vals, 95))

    pval_rows.append(out_row)

pval_df = pd.DataFrame(pval_rows)
pval_out = OUT_DIR / "stage3_qexp_segment_pvalues.csv"
pval_df.to_csv(pval_out, index=False)
print(f"[저장] {pval_out}  ({len(pval_df)} rows)")

preview_cols = ["channel", "segment"] + [c for f in FEATURES for c in [f"{f}_observed", f"{f}_placebo_p95", f"{f}_pvalue"]]
preview_cols = [c for c in preview_cols if c in pval_df.columns]
print("\n미리보기 (상위 10행):")
print(pval_df[preview_cols].head(10).to_string(index=False))


# =============================================================================
# 6. 채널별 요약 — 위약 대비 유의미(p<0.05)한 세그먼트 비율
# =============================================================================
section("4. 채널별 요약: 위약 풀 대비 유의미(p<0.05)한 이상 세그먼트 비율")

summary_rows = []
for ch in FINAL_CHANNELS:
    sub = pval_df[pval_df["channel"] == ch]
    row = {"channel": ch, "n_segments": len(sub)}
    for feat in FEATURES:
        pcol = f"{feat}_pvalue"
        p_vals = sub[pcol].dropna().values
        row[f"{feat}_frac_significant"] = float(np.mean(p_vals < 0.05)) if len(p_vals) else np.nan
        row[f"{feat}_n_valid"] = len(p_vals)
    summary_rows.append(row)

channel_summary_df = pd.DataFrame(summary_rows)
channel_summary_out = OUT_DIR / "stage3_qexp_channel_summary.csv"
channel_summary_df.to_csv(channel_summary_out, index=False)
print(channel_summary_df.to_string(index=False))
print(f"\n[저장] {channel_summary_out}")


# =============================================================================
# 7. Mann-Whitney U 검정 — 이상 세그먼트 효과 분포 vs 위약 풀 분포 (독립 2군 비교)
# =============================================================================
section("5. Mann-Whitney U 검정: 이상 세그먼트 효과크기 분포 vs 위약 풀 분포 (채널별)")

mw_rows = []
for ch in FINAL_CHANNELS:
    obs_ch = observed_df[observed_df["channel"] == ch]
    placebo_ch = placebo_df[placebo_df["channel"] == ch]
    for feat in FEATURES:
        a = obs_ch[feat].dropna().values
        b = placebo_ch[feat].dropna().values
        if len(a) < 5 or len(b) < 5:
            mw_rows.append({
                "channel": ch, "feature": feat, "n_anomaly": len(a), "n_placebo": len(b),
                "u_stat": np.nan, "p_value": np.nan, "median_anomaly": np.nan, "median_placebo": np.nan,
                "note": "표본부족 (n<5)",
            })
            continue
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative="greater")
        mw_rows.append({
            "channel": ch, "feature": feat, "n_anomaly": len(a), "n_placebo": len(b),
            "u_stat": float(u_stat), "p_value": float(p_val),
            "median_anomaly": float(np.median(a)), "median_placebo": float(np.median(b)),
            "note": "",
        })

mw_df = pd.DataFrame(mw_rows)
n_tests = int(mw_df["p_value"].notna().sum())
mw_df["p_value_bonferroni"] = (mw_df["p_value"] * max(n_tests, 1)).clip(upper=1.0)

mw_out = OUT_DIR / "stage3_qexp_mannwhitney.csv"
mw_df.to_csv(mw_out, index=False)
print(mw_df.to_string(index=False))
print(f"\n[저장] {mw_out}")


# =============================================================================
# 8. 완료 — 해석 가이드
# =============================================================================
section("완료 — 해석 가이드")
print(
    "1) segment_pvalues의 '_pvalue'는 '이 이상 세그먼트의 변화가 같은 채널 정상\n"
    "   세그먼트들의 자연스러운 변동 폭보다 얼마나 극단적인가'를 나타내는 경험적\n"
    "   단측 p-value입니다. 낮을수록(0에 가까울수록) 정상적 변동으로는 설명 안\n"
    "   되는 변화라는 뜻입니다.\n"
    "2) channel_summary의 frac_significant가 높을수록, 그 채널·피처에서 이상\n"
    "   세그먼트들의 변화가 '정상 변동 범위를 벗어난다'는 근거가 강합니다. 이걸\n"
    "   §2-2(내부 pre/post)·§2-3(선행성)·§2-4(메타분석) 결과와 나란히 놓으면,\n"
    "   '내부적으로도 유의하고, 정상 대조군과 비교해도 유의하다'는 이중 근거가\n"
    "   완성됩니다.\n"
    "3) Mann-Whitney 결과의 p_value_bonferroni를 주 지표로 쓰세요(15개 검정=5채널\n"
    "   x3피처 다중비교 보정). median_anomaly가 median_placebo보다 뚜렷이 크면서\n"
    "   p_value_bonferroni가 유의하면, 채널 전체 수준에서 '이상 세그먼트 그룹이\n"
    "   정상 세그먼트 그룹보다 체계적으로 더 크게 움직인다'는 근거입니다.\n"
    "4) 이 결과 역시 무작위 배정 실험이 아닌 준실험(quasi-experimental) 설계이므로,\n"
    "   '정상적 변동보다 유의하게 크다'까지만 주장하고 완전한 인과 증명으로\n"
    "   과장하지 마세요. 다음 단계(SCM 경로 A/B 결정)에서 이 결과를 '인과 뼈대가\n"
    "   실제 효과를 반영한다'는 근거 자료 중 하나로 사용하면 됩니다."
)

==========================================================================================
0. 데이터 로드 및 canonical onset 재구성 (§2-2/§2-3와 동일 절차 재사용)
==========================================================================================
신뢰도 high+medium 이상 세그먼트: 177개

==========================================================================================
1. 이상 세그먼트 관측 효과크기 계산
==========================================================================================
관측 효과크기 계산된 이상 세그먼트: 177개
channel
CADC0872    41
CADC0873    29
CADC0874    50
CADC0888    36
CADC0894    21

==========================================================================================
2. 채널별 위약(placebo) 풀 생성 (정상 세그먼트당 5회 무작위 pivot)
==========================================================================================
[CADC0872] 정상 세그먼트 300개 -> 위약 표본 1495개 생성
[CADC0873] 정상 세그먼트 300개 -> 위약 표본 1485개 생성
[CADC0874] 정상 세그먼트 125개 -> 위약 표본 625개 생성
[CADC0888] 정상 세그먼트 192개 -> 위약 표본 760개 생성
[CADC0894] 정상 세그먼트 123개 -> 위약 표본 615개 생성

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/stage3_qexp_placebo_pool.csv  (4980 rows)

==========================================================================================
3. 이상 세그먼트별 경험적 p-value 계산 (위약 풀 대비 상위 % 위치)
==========================================================================================
[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/stage3_qexp_segment_pvalues.csv  (177 rows)

미리보기 (상위 10행):
 channel  segment  level_d_abs_observed  level_d_abs_placebo_p95  level_d_abs_pvalue  diff_logvar_observed  diff_logvar_placebo_p95  diff_logvar_pvalue  diff2_logvar_observed  diff2_logvar_placebo_p95  diff2_logvar_pvalue
CADC0872        2              1.864434                  3.95771            0.548128              1.617414                 1.809027            0.059492               1.557343                  1.656322             0.056818
CADC0872        4              0.533090                  3.95771            0.816845              0.976971                 1.809027            0.083556               1.302382                  1.656322             0.090241
CADC0872        8              1.383267                  3.95771            0.641043              2.997197                 1.809027            0.011364               3.012032                  1.656322             0.011364
CADC0872       13              0.899902                  3.95771            0.728610              3.438259                 1.809027            0.007353               3.319074                  1.656322             0.004011
CADC0872       20              3.594102                  3.95771            0.095588              1.912758                 1.809027            0.044118               1.679047                  1.656322             0.047460
CADC0872       21              2.310658                  3.95771            0.405080              2.211143                 1.809027            0.028075               2.464671                  1.656322             0.015374
CADC0872       22              6.771598                  3.95771            0.000668              3.396931                 1.809027            0.007353               3.201027                  1.656322             0.004011
CADC0872       29              1.858211                  3.95771            0.549465              2.373936                 1.809027            0.019385               2.423515                  1.656322             0.018048
CADC0872       80              3.877146                  3.95771            0.054813              4.111519                 1.809027            0.003342               4.400016                  1.656322             0.002674
CADC0872       97              1.443944                  3.95771            0.634358              4.899863                 1.809027            0.001337               5.556300                  1.656322             0.002005

==========================================================================================
4. 채널별 요약: 위약 풀 대비 유의미(p<0.05)한 이상 세그먼트 비율
==========================================================================================
 channel  n_segments  level_d_abs_frac_significant  level_d_abs_n_valid  diff_logvar_frac_significant  diff_logvar_n_valid  diff2_logvar_frac_significant  diff2_logvar_n_valid
CADC0872          41                      0.048780                   41                      0.780488                   41                       0.804878                    41
CADC0873          29                      0.000000                   29                      0.892857                   28                       0.892857                    28
CADC0874          50                      0.020000                   50                      0.500000                   50                       0.840000                    50
CADC0888          36                      0.083333                   36                      0.121212                   33                       0.424242                    33
CADC0894          21                      0.142857                   21                      0.250000                   16                       0.250000                    16

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/stage3_qexp_channel_summary.csv

==========================================================================================
5. Mann-Whitney U 검정: 이상 세그먼트 효과크기 분포 vs 위약 풀 분포 (채널별)
==========================================================================================
 channel      feature  n_anomaly  n_placebo  u_stat      p_value  median_anomaly  median_placebo note  p_value_bonferroni
CADC0872  level_d_abs         41       1495 25627.0 9.634314e-01        1.450736        2.019690             1.000000e+00
CADC0872  diff_logvar         41       1495 59400.0 5.254867e-25        2.927772       -0.801451             7.882301e-24
CADC0872 diff2_logvar         41       1495 58675.0 7.413532e-24        2.962304        0.096681             1.112030e-22
CADC0873  level_d_abs         29       1485  9951.0 9.999997e-01        0.768600        2.193673             1.000000e+00
CADC0873  diff_logvar         28       1485 40725.0 1.609797e-18        4.329193       -1.162254             2.414695e-17
CADC0873 diff2_logvar         28       1485 40054.0 2.041600e-17        4.324148        0.040494             3.062400e-16
CADC0874  level_d_abs         50        625 14906.0 7.061870e-01        1.036157        1.066683             1.000000e+00
CADC0874  diff_logvar         50        625 26696.0 3.599511e-17        2.398290       -0.011362             5.399267e-16
CADC0874 diff2_logvar         50        625 28771.0 1.928547e-23        2.421222       -0.002773             2.892821e-22
CADC0888  level_d_abs         36        760 12797.0 7.438968e-01        0.985201        1.164286             1.000000e+00
CADC0888  diff_logvar         33        654 16250.0 4.621805e-07        2.385977        0.193253             6.932708e-06
CADC0888 diff2_logvar         33        659 19335.0 2.176436e-14        3.383056       -0.435012             3.264654e-13
CADC0894  level_d_abs         21        615  7226.0 1.768215e-01        0.888752        0.914404             1.000000e+00
CADC0894  diff_logvar         16        569  7899.0 2.597045e-07        2.470860        0.269297             3.895568e-06
CADC0894 diff2_logvar         16        571  7383.0 1.296501e-05        2.196727        0.244223             1.944752e-04

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/stage3_qexp_mannwhitney.csv

==========================================================================================
완료 — 해석 가이드
==========================================================================================
1) segment_pvalues의 '_pvalue'는 '이 이상 세그먼트의 변화가 같은 채널 정상
   세그먼트들의 자연스러운 변동 폭보다 얼마나 극단적인가'를 나타내는 경험적
   단측 p-value입니다. 낮을수록(0에 가까울수록) 정상적 변동으로는 설명 안
   되는 변화라는 뜻입니다.
2) channel_summary의 frac_significant가 높을수록, 그 채널·피처에서 이상
   세그먼트들의 변화가 '정상 변동 범위를 벗어난다'는 근거가 강합니다. 이걸
   §2-2(내부 pre/post)·§2-3(선행성)·§2-4(메타분석) 결과와 나란히 놓으면,
   '내부적으로도 유의하고, 정상 대조군과 비교해도 유의하다'는 이중 근거가
   완성됩니다.
3) Mann-Whitney 결과의 p_value_bonferroni를 주 지표로 쓰세요(15개 검정=5채널
   x3피처 다중비교 보정). median_anomaly가 median_placebo보다 뚜렷이 크면서
   p_value_bonferroni가 유의하면, 채널 전체 수준에서 '이상 세그먼트 그룹이
   정상 세그먼트 그룹보다 체계적으로 더 크게 움직인다'는 근거입니다.
4) 이 결과 역시 무작위 배정 실험이 아닌 준실험(quasi-experimental) 설계이므로,
   '정상적 변동보다 유의하게 크다'까지만 주장하고 완전한 인과 증명으로
   과장하지 마세요. 다음 단계(SCM 경로 A/B 결정)에서 이 결과를 '인과 뼈대가
   실제 효과를 반영한다'는 근거 자료 중 하나로 사용하면 됩니다.

# 결과가 아주 선명하고, 동시에 이전 단계 결과와 흥미로운 긴장 관계를 만듭니다. 하나씩 짚겠습니다.
# ## 핵심 발견 — level과 diff/diff²의 운명이 완전히 갈림
# | 피처 | 5채널 전체 판정 | p_bonferroni 범위 |
# |---|---|---|
# | **level (\|d\|)** | **전부 비유의** | 1.000 (전 채널) |
# | **diff (log 분산비)** | **전부 강하게 유의** | 6e-24 ~ 4e-6 |
# | **diff2 (log 분산비)** | **전부 강하게 유의** | 1e-22 ~ 2e-4 |
# 게다가 level은 `median_anomaly < median_placebo`인 채널이 대부분입니다(872, 873, 888, 894) — 즉 **이상 세그먼트의 레벨 변화가 정상 세그먼트를 무작위로 잘랐을 때보다 오히려 작거나 비슷합니다.**
# ## 왜 이게 모순이 아니라 오히려 좋은 결과인가
# 이전 §2-2(내부 pre/post)에서는 level의 `frac_significant`가 0.86~0.92로 매우 높게 나왔었습니다. 이번 결과와 겹쳐 놓으면 이렇게 정리됩니다:
# > **"level이 onset 전후로 유의하게 달라진다"는 §2-2의 발견은 사실이지만, 그건 이상 신호만의 특성이 아니라 이 채널들이 원래 정상 세그먼트에서도 흔히 보이는 완만한 드리프트(추세)의 산물이다.**
# 실제로 보완 H에서 확인했던 872의 원신호 그림들을 떠올려보면, pre 구간 자체가 이미 오르내리는 추세를 갖고 있었습니다. 정상 세그먼트도 아무 데나 잘라서 pre/post를 나누면 똑같이 "추세 때문에 평균이 다르다"는 결과가 나옵니다. 그래서 **level은 "이상 vs 정상"을 구분하는 마커로서는 힘이 없다**는 게 이번 준실험 비교로 명확히 드러났습니다.
# 반대로 **diff/diff²의 분산 급증은 정상 세그먼트에서는 거의 나타나지 않는, 이상에 고유한 신호**입니다. 이게 이 연구 전체에서 가장 방어하기 좋은 인과적 근거입니다 — §2-3에서 확인한 "diff/diff²가 먼저 튄다"는 시간적 선행성과 결합하면:
# > **"이상의 진짜 시그니처는 지속적 레벨 이동이 아니라, onset 직후 변화율·가속도의 일시적(transient) 분산 급증이며, 이는 같은 채널의 정상 세그먼트에서는 관찰되지 않는 이상 고유의 패턴이다."**
# ## 채널별 강도 차이도 주목할 만함
# diff/diff²의 유의 비율(§4 채널요약)을 보면 872·873·874(자력계, float_noise_suspect)가 0.78~0.89로 매우 높은 반면, 888·894(광다이오드, quantized)는 0.12~0.42로 상대적으로 약합니다. 이는 이전 §2-4 메타분석의 I²(이질성) 결과와도 방향이 일치합니다 — "채널 공통 효과 하나"보다는 "자력계 계열에서 특히 강한 diff/diff² 시그니처"로 서술하는 게 데이터와 더 맞습니다.
# ## 다음 액션 제안
# 이 결과로 **C-8(준실험적 pre/post 비교)은 완료**입니다. 이제 남은 건 **C-9(SCM 경로 결정)**인데, 지금까지 나온 근거를 종합하면 SCM 뼈대를 다음처럼 잡는 게 데이터와 가장 일치합니다:
# ```
# onset → diff/diff² 분산 급증(transient, 채널 고유 강도) → (약한 영향) level
# ```
# level은 인과 사슬의 최종 산물이 아니라 오히려 배제해야 할 혼란변수(confounder, 정상 드리프트와 혼재)로 다루는 게 안전합니다.
# 이 방향으로 SCM 뼈대 정리(경로 A) 문서·표를 작성해드릴까요, 아니면 지금 결과를 먼저 "이질성 채널별 요약표"에 통합해서 최종 정리본을 만들어드릴까요?

# =============================================================================
# 층3 산출물 (1): SCM 뼈대 확정 — 경로 A (신호처리 구조 기반 인과 뼈대)
# =============================================================================
#
# [배경] §2-2(내부 pre/post), §2-3(선행성 CMH), §2-4(메타분석+LOO),
# §3-준실험(위약 풀 대조)까지 네 갈래의 독립적 검증을 모두 거친 뒤 확정하는
# SCM 뼈대. 원래 계획했던 "level → diff → diff² 순서의 5채널 공통 SCM"은
# 이번 준실험 결과로 완전히 폐기하고, 아래 구조로 교체한다.
#
# -----------------------------------------------------------------------------
# [최종 SCM 뼈대]
#
#   Onset (BOCPD 확률적 변화점)
#         │
#         ▼
#   Diff/Diff² 분산 급증 (transient, onset 직후 국한)
#         │  ├─ 채널별 강도(moderator): float_noise_suspect(872/873/874) 강함
#         │  │                          quantized(888/894) 약함
#         │
#         ▼ (약한/불확실한 파생 경로 — 인과 사슬의 핵심 아님)
#   Level 변화 (persistent, 그러나 confounded)
#
#   ※ Level은 인과 사슬의 "결과 변수"로 취급하지 않는다. 정상 세그먼트에서도
#     동일한 정도의 drift가 나타나므로(§3 준실험, level MW p_bonf=1.000 전 채널),
#     level 변화는 "이상이 만든 결과"가 아니라 "채널이 원래 갖고 있는 완만한
#     추세(confounder)와 혼재된 관찰"로 다뤄야 한다.
#
# -----------------------------------------------------------------------------
# [단계별 근거 요약]
#
#   (1) 시간적 선행성 (§2-3, CMH 검정)
#       diff/diff2가 level보다 먼저 임계값을 넘음 (전체 CMH p<0.001,
#       채널별 기저율 통제 후에도 유지) → 순서상 diff/diff2가 선행
#
#   (2) 효과 지속성 프로파일 (보완진단 A)
#       diff/diff2: 98%가 transient(짧고 뾰족한 스파이크 후 감쇠)
#       level: persistent(지속 유지)와 transient가 혼재(약 50:50)
#       → diff/diff2는 "순간적 반응", level은 "느린 추세"라는 질적 차이
#
#   (3) 채널 간 이질성 (§2-4 메타분석 + LOO)
#       diff/diff2의 pooled frac_significant: I²=92~94%(매우 높음)
#       → "5채널 공통 크기"로 뭉뚱그릴 수 없고, 채널별 조절효과로 반영해야 함
#       level의 큰 효과크기(872)는 onset 불확실성·분모 왜소 아님 → 실제 신호
#       (보완 G/H) 이지만 아래 (4)에서 confounder로 재해석됨
#
#   (4) 준실험적 정상군 대조 (본 스크립트, 결정적 근거)
#       level:  MW U 검정 전 채널 비유의(p_bonf=1.000), median_anomaly가
#               median_placebo보다 오히려 작은 채널 다수
#               → "이상 특유의 레벨 이동"이라는 근거 소멸. §2-2에서 관찰된
#                 level의 높은 내부 유의성은 "정상 세그먼트에도 흔한 drift"
#                 였을 뿐 — confounder로 재분류
#       diff/diff2: MW U 검정 전 채널 강한 유의(p_bonf 최대 4e-6, 대부분 <1e-15)
#               → 정상 세그먼트에서는 재현되지 않는, 이상에 고유한 시그니처
#
# -----------------------------------------------------------------------------
# [경로 A를 채택하고 경로 B(물리 SCM)를 보류하는 이유]
#
#   경로 A(신호처리 구조 기반)만으로도 "diff/diff2 분산 급증"이라는 이상
#   고유 시그니처와 그 채널별 조절효과까지 통계적으로 방어 가능한 수준으로
#   확정되었음. 경로 B(TLE/자세 quaternion과 연결한 물리 기반 SCM)는:
#     - 보조 데이터(TLE, 자세 quaternion) 존재 여부가 여전히 미확인 상태
#     - 경로 A의 결론(diff/diff2 중심 구조)이 물리적으로 그럴듯한지
#       (자력계 진동·광다이오드 각도 변화의 실제 물리 메커니즘과 부합하는지)
#       사전 검토가 안 된 상태
#   → 경로 B는 "확장 가능성"으로 남겨두되, 현재 논문 스코프는 경로 A로
#     확정하는 것을 권장. 경로 B 착수 전 "TLE/자세 데이터 존재 여부 확인"이
#     반드시 선행되어야 함(§B 미해결 항목과 동일 선상).
#
# -----------------------------------------------------------------------------
# [최종 채택 SCM 명제 — 논문 서술용 한 문장]
#
#   "OPSSAT-AD의 이상은 신호 레벨 자체의 지속적 이동이 아니라, onset 직후
#    국한된 변화율·가속도(diff/diff²)의 일시적 분산 급증으로 특징지어지며,
#    이 시그니처는 같은 채널의 정상 세그먼트에서는 재현되지 않는 이상 고유의
#    패턴이다. 신호 레벨의 변화는 채널이 원래 갖는 완만한 드리프트와 혼재되어
#    있어 인과적 결과변수가 아니라 혼란변수로 취급해야 하며, 이 시그니처의
#    강도는 채널 유형(float_noise_suspect > quantized)에 따라 이질적이다."
#
# =============================================================================

# =============================================================================
# 층3 산출물 (2): 채널×피처 이질성·근거 종합 요약표
# =============================================================================
#
# [목적] §2-2/§2-3/§2-4/LOO/보완G·H/§3-준실험까지 흩어져 있던 "왜 이 채널이
# 특이한가"에 대한 결론들을 feature(level/diff/diff2) x channel 단위로
# 한 번에 조회 가능하도록 통합. 이후 SCM 문서·논문 본문 작성 시 이 표 하나만
# 참조하면 되도록 만드는 것이 목적이며, 재계산은 하지 않고 기존 결론만 정리.
#
# -----------------------------------------------------------------------------
# [표 1] Feature x Channel 이질성 원인 및 최종 판정
#
#   feature  | 이질성 원인 채널 | 원인 성격                    | §3 준실험 판정
#   ---------|-----------------|------------------------------|----------------
#   level    | CADC0872 (크기)  | 실제 신호(onset 불확실성·      | 비유의(전 채널)
#            |                 | 분모 왜소 아님, 보완G/H로 확인)| → confounder로 재분류
#            | CADC0888 (유의성)| 888만 유의비율 0.55, 나머지     |
#            |                 | 0.96~0.98로 이질적             |
#   diff2    | CADC0874 (크기)  | transient_ratio와 완벽한 역상관 | 강한 유의(전 채널)
#            |                 | (r=-1.0, [F-2]) → 검정설계상   | → 인과 시그니처로 채택
#            |                 | 과소추정(방법론적 아티팩트 가능)|
#   diff/    | CADC0894 (유의성)| 894만 frac_sig 0.81~0.86으로    | 강한 유의(전 채널,
#   diff2    |                 | 압도적, 나머지 4채널은 0.001~   |   단 894 내부에서는
#            |                 | 0.18의 자연스러운 위계 존재     |   상대적으로 약함)
#
# -----------------------------------------------------------------------------
# [표 2] 채널별 최종 종합 판정 (5채널 모두 2단계/3단계 대상으로 유지)
#
#   channel   | 채널유형            | level 특이성        | diff/diff2 특이성      | 최종 해석
#   ----------|--------------------|--------------------|-----------------------|-----------------------------
#   CADC0872  | float_noise_suspect | 크기 이상치(과대)    | diff/diff2 유의비율    | diff/diff2 시그니처 강함,
#             |                     | → 실제 신호로 확인   | 최상위권(0.78~0.80)    | level은 confounder
#   CADC0873  | float_noise_suspect | 평이함               | diff/diff2 유의비율    | diff/diff2 시그니처 매우 강함
#             |                     |                     | 최상위권(0.89)         |
#   CADC0874  | float_noise_suspect | 평이함               | diff2 효과크기 과소    | diff2 실제 시그니처는 표 상
#             |                     |                     | 추정 가능성(검정설계    | 수치보다 강할 가능성
#             |                     |                     | 아티팩트, [F-2])       |
#   CADC0888  | quantized           | 유의성 이상치(과소)  | diff/diff2 유의비율    | 5채널 중 diff/diff2
#             |                     | → level 자체가 노이즈| 상대적으로 약함        | 시그니처가 가장 약함
#             |                     | 성 지표              | (0.12~0.42)           |
#   CADC0894  | quantized           | 평이함               | frac_significant 유의성| §2 내부 검정에서는
#             |                     |                     | 이상치(압도적으로 높음)| 특이했으나 §3 준실험
#             |                     |                     |                       | 에서는 나머지와 유사한
#             |                     |                     |                       | 패턴(정상 대비 유의)
#
# -----------------------------------------------------------------------------
# [핵심 결론 — 표 1·2를 관통하는 하나의 문장]
#
#   "level 차원의 이질성(872 크기, 888 유의성)은 §3 준실험에서 level 자체가
#    인과적으로 무의미해짐에 따라 자연히 해소되는 반면, diff/diff2 차원의
#    이질성(874 크기, 894 유의성)은 인과 시그니처 '내부의' 강도 차이로 남아
#    채널별 조절변수(moderator)로 SCM에 반영해야 한다."
#
# -----------------------------------------------------------------------------
# [남는 방법론적 주석 — 향후 리비전 시 참고]
#
#   - 874의 diff2 효과크기는 [F-2]에서 방법론적 과소추정 가능성이 제기됨
#     (전체창 평균 검정이 빠른 decay를 가장 심하게 희석) → 정식 리비전에서는
#     조기창(early-window) 재검정치를 diff2 대표값으로 병기 권장
#   - 894는 §2(내부) 기준으로는 이질적이었으나 §3(준실험) 기준으로는 다른
#     4채널과 유사한 패턴으로 수렴 → "이질성"의 정의가 비교 기준(내부 vs
#     대조군)에 따라 달라질 수 있다는 점을 논문 한계(limitation)로 명시할 것
#   - 이 요약표는 재계산 없이 기존 결과(§2-4, LOO, F-2, 본 §3 결과)를
#     정리한 것이므로, 원본 수치가 갱신되면 반드시 함께 갱신할 것
#
# =============================================================================








