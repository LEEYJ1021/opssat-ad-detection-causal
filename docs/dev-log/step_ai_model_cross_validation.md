"""
OPSSAT-AD 부록: 모델군(Model-Class) 교차검증 — 다중 아키텍처 특징귀속 일치도 분석
=============================================================================
[이 버전에서 수정된 사항 — 3군데]

1) compute_shap_importance() 함수:
   기존 코드는 shap_values() 반환값이 "list"인 경우만 처리했다(tree/kernel
   경로 일부에만 있었고 linear 경로엔 아예 없었음). 그런데 최신 SHAP
   버전에서는 이진분류 predict_proba를 explainer에 넘기면(GaussianNB, kNN,
   SVM_RBF, MLP_shallow처럼 kind="kernel"인 모델들, 그리고 일부 트리 모델도)
   list가 아니라 (n_samples, n_features, n_classes) 형태의 3차원 ndarray를
   반환한다. 이 경우 importance.shape이 (3,2)가 되어버려 이후
   `float(imp)`에서 "only 0-dimensional arrays can be converted to Python
   scalars" TypeError로 죽는다(GaussianNB에서 실제로 발생).
   -> list/3D ndarray 두 케이스를 모두 (n_features,) 1D로 정규화하고, 그래도
      shape이 안 맞으면 None을 반환해 permutation importance로 안전하게
      폴백하도록 수정.

2) XGBClassifier(...) 생성 시 `use_label_encoder=False` 파라미터 제거:
   이 파라미터는 최신 xgboost(2.x)에서 완전히 삭제되어
   "TypeError: __init__() got an unexpected keyword argument
   'use_label_encoder'"로 모델 zoo 자체가 죽는다. 최신 xgboost는 이
   옵션 없이도 동일하게 동작하므로 안전하게 제거.

3) [신규] 표 모델 zoo에서 MLP_shallow 제외:
   실행 결과 MLP_shallow는 AUC=0.403(무작위 0.5보다 낮음)으로 나와, 이
   모델이 diff/diff2보다 level을 중요하게 본 것이 "귀납편향에 따른 흥미로운
   예외"가 아니라 단순 학습 실패(수렴 실패/불안정)로 판단된다. 학습이 안 된
   모델의 특징중요도는 무의미한 잡음이므로, 이를 17개 모델 합의도(Kendall's
   W, 이항검정) 계산에 포함시키면 결론을 왜곡할 위험이 있다. 따라서 표 모델
   zoo에서 MLP_shallow를 제외하고 총 16개 모델(표 10개 + 시퀀스 6개)로
   재구성했다. 학습 실패의 원인(조기종료 시점, 스케일링, 클래스 불균형 등)을
   추가로 진단하고 싶다면 별도로 재도입해 디버깅할 것을 권장한다.

4) [신규] 시퀀스 모델(BiLSTM/BiGRU) saliency 계산 시 cuDNN RNN backward 오류:
   eval_model()로 model.eval() 상태에 둔 뒤, 같은 모델에 대해 Integrated
   Gradients(캡텀 경로든 수동 구현 경로든)로 backward를 호출하면
   "RuntimeError: cudnn RNN backward can only be called in training mode"
   로 죽는다. 이는 cuDNN의 LSTM/GRU 커널이 eval 모드에서는 backward에
   필요한 중간 텐서를 보존하지 않기 때문이며(계산값이 아니라 커널
   최적화 경로의 문제), CNN/TCN/Transformer는 nn.LSTM/nn.GRU를 쓰지
   않으므로 이 문제가 발생하지 않았다.
   -> attribution(saliency) 계산 구간만 `torch.backends.cudnn.flags(
      enabled=False)`로 감싸, 그 호출에 한해 cuDNN이 아닌 일반(generic)
      RNN 구현을 쓰도록 강제했다. 이는 수치 결과에는 영향을 주지 않고
      (같은 연산을 더 느린 커널로 수행할 뿐) backward를 학습 모드
      전환 없이도 가능하게 한다. captum 경로와 수동 IG 폴백 경로 모두에
      동일하게 적용.

[참고] 아래 본문의 "17개 모델" 서술은 원래 설계 의도(선형~SSM까지 17개
독립 모델을 비교한다는 취지)를 설명하는 부분이며, 실제 실행 시에는 위
PATCH 3에 따라 MLP_shallow가 제외되어 최종 집계·통계량(Kendall's W,
이항검정, family_summary 등)은 전부 16개 모델(표 10개 + 시퀀스 6개)
기준으로 계산된다. 코드 내 print/report 문구는 모델 수를 데이터에서
동적으로 세어 표시하므로 자동으로 16으로 반영된다.

그 외 로직(부트스트랩, 합의도 분석 등)은 원본과 동일하다.
=============================================================================
목적
-----------------------------------------------------------------------------
지금까지(층1~4) 확립한 핵심 인과 시그니처는 "onset 직후 diff/diff²의 일시적
분산 급증이 이상 고유 패턴이며, level(신호 레벨 자체)은 정상 세그먼트에도
흔한 드리프트와 혼재된 confounder"라는 것이었다. 이 결론은 지금까지 전부
"같은 통계·확률모델 계열"(칼만필터 + BOCPD + CVaR) 안에서만 검증되었다 —
즉 train/test 3중 검증, 부트스트랩은 "같은 파이프라인이 과적합되지 않았는가"
는 보여주지만 "이 시그니처가 방법론 자체의 산물이 아니라 데이터에 실재하는
패턴인가"는 답하지 못한다.

이 스크립트는 원래 계획(Mamba+SHAP 단일 벤치마크) 대신, 훨씬 강건한 설계인
"다중 독립 모델군 비교(model-class robustness analysis)"를 수행한다:
서로 다른 귀납편향(inductive bias)을 가진 17개 모델 — 선형, 확률적, 사례기반,
커널, 트리 앙상블(배깅×2 + 부스팅×3), 얕은 신경망, CNN, RNN(양방향×2),
Transformer, 경량 Mamba식 선택적 상태공간모델(SSM), TCN — 전부에 대해
"level/diff/diff² 중 무엇이 이상 판별에 가장 중요한가"를 독립적으로 묻고,
그 답이 모델군에 관계없이 일관되는지를 정량적으로(Kendall's W, 부호검정,
부트스트랩 안정성) 검증한다.

설계 원칙 — "왜 Mamba 하나가 아니라 17개인가"
-----------------------------------------------------------------------------
단일 딥러닝 벤치마크(예: Mamba 하나)로 결과가 어긋나면, 그게 "결론이 틀려서"
인지 "그 특정 아키텍처의 우연한 한계" 때문인지 구분할 수 없다. 여러 독립적
모델 계열에서 diff/diff² 중요도가 반복 확인된다면, 이는 "특정 모델의 우연한
산물"이 아니라 "모델 선택과 무관하게 견고한 신호"라는 훨씬 강한 근거가 된다.
이는 층4에서 C_MISS·C_DELAY를 여러 자릿수로 흔들어 결론의 견고성을 검증했던
것과 동일한 철학 — "결론에 영향을 줄 수 있는 축을 찾아 직접 흔들어본다" —을
모델 선택이라는 축에 적용한 것이다.

두 갈래 데이터 표현과 그 이유
-----------------------------------------------------------------------------
[표 형태 — 11개 모델] §3 준실험(위약 대조) 스크립트에서 이미 확립한 세
공학적 피처(level_d_abs, diff_logvar, diff2_logvar)를 그대로 재사용한다.
이 피처들은 "관측 이상 세그먼트(canonical onset 기준)" 대 "정상 세그먼트를
무작위 시점에서 가른 위약(placebo)"의 이진분류 문제로 프레이밍된다 — §3과
완전히 동일한 라벨링·표본추출 로직이며, 재계산 없이 그대로 이어받는다.

[시퀀스 형태 — 6개 모델] 표 형태 피처는 이미 사람이 설계한 요약통계량이므로,
"애초에 요약통계량 설계 자체가 diff/diff²를 편애한 것 아닌가"라는 의심을
배제할 수 없다. 이를 막기 위해, 시퀀스 모델에는 원시 신호(level)와 그 1차·
2차 차분(diff, diff²)을 각각 "채널"로 하는 3채널 원시 시계열 윈도우(pivot
전후 SEQ_WINDOW_HALF틱)를 그대로 입력하고, 모델이 학습 후 어느 채널에 더
큰 saliency(적분 그래디언트)를 두는지를 본다 — 즉 사람이 요약하지 않은 원시
데이터에서 모델이 "스스로" 어느 채널이 더 정보량이 많다고 판단하는지를
측정한다. 이렇게 하면 표 형태 결과와 시퀀스 형태 결과가 "같은 결론"에
도달하는지가 훨씬 엄격한 교차검증이 된다.

두 표현 모두 최종적으로 {level, diff, diff2} 세 항목에 대한 "상대적 중요도"
벡터로 환원되므로, 17개 모델 전부를 하나의 공통 좌표계에서 직접 비교할 수
있다.

라이브러리 의존성 — 모두 소프트 의존성(soft dependency)
-----------------------------------------------------------------------------
xgboost, lightgbm, shap, torch가 설치되지 않은 환경에서도 스크립트가 죽지
않고, 해당 모델/기능만 건너뛰며 안내 메시지를 출력한다. mamba_ssm(공식
CUDA 커널 패키지)이 없으면 순수 PyTorch로 구현한 경량 선택적 SSM 블록으로
자동 대체된다(성능·속도가 공식 구현과 동일하지 않음을 코드 내에 명시).

입력 (재계산 없이 재사용):
  - segments.csv
  - stage2_mc_draws_raw.csv, stage2_mc_segment_summary_corrected.csv
  - stage3_qexp_placebo_pool.csv (있으면 로드, 없으면 재계산)

출력 (OUT_DIR):
  - modelclass_dataset_tabular.csv          (17개 모델이 공유하는 라벨셋 원자료)
  - modelclass_feature_importance.csv       (모델 x 피처 중요도 원자료)
  - modelclass_performance_leaderboard.csv  (모델별 AUC/recall/FA, BOCPD 대조)
  - modelclass_agreement_summary.csv        (Kendall's W, 부호검정, 모델군별 일치도)
  - modelclass_bootstrap_stability.csv      (표 모델 부트스트랩 안정성)
  - modelclass_final_report.txt             (해석 가이드 텍스트)
"""

from pathlib import Path
from dataclasses import dataclass
import warnings
import json
import math
import random

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# 0. 경로 및 잠금 설정 (1~4단계와 완전히 동일한 규약)
# =============================================================================
BASE_DIR = Path("/home/yjlee/Research/Satellite/OPSSAT-AD_LEO data")
SEGMENTS_PATH = BASE_DIR / "segments.csv"
OUT_DIR = BASE_DIR / "causal_pipeline_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DRAWS_PATH = OUT_DIR / "stage2_mc_draws_raw.csv"
SEG_CORRECTED_PATH = OUT_DIR / "stage2_mc_segment_summary_corrected.csv"
PLACEBO_CACHE_PATH = OUT_DIR / "stage3_qexp_placebo_pool.csv"

FINAL_CHANNELS = ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0894"]

BURN_IN = 5
MIN_WINDOW_POINTS = 5
K_PLACEBO_DRAWS_PER_NORMAL = 5
MAX_NORMAL_PER_CHANNEL = 300
RANDOM_STATE = 42

SEQ_WINDOW_HALF = 15                     # 시퀀스 모델 입력 윈도우 반폭(틱)
SEQ_MIN_SEGMENT_LEN = 2 * SEQ_WINDOW_HALF + BURN_IN

N_BOOT_TABULAR = 200      # 표 모델 부트스트랩 반복 (계산비용 낮음 -> 크게)
N_BOOT_DEEP = 20          # 딥러닝 부트스트랩 반복 (계산비용 높음 -> 작게, 의도적 비대칭)
EXPLAIN_SAMPLE_SIZE = 400  # SHAP/saliency 계산 시 표본 상한 (3피처라 원래도 저렴)

# --- 층4 v7.8에서 확정된 BOCPD/tau*_Youden 성능 (대조 기준선으로 재사용) ---
REFERENCE_BOCPD_PERFORMANCE = {
    "CADC0872": {"recall": 0.587786, "fa_rate": 0.012077},
    "CADC0873": {"recall": 0.663462, "fa_rate": 0.004132},
    "CADC0874": {"recall": 0.840580, "fa_rate": 0.064000},
    "CADC0888": {"recall": 0.473684, "fa_rate": 0.296053},
    "CADC0894": {"recall": 0.857143, "fa_rate": 0.528455},
}

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)


def section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# =============================================================================
# 0-b. 소프트 의존성 점검
# =============================================================================
section("0. 소프트 의존성(라이브러리) 점검")

HAVE_SKLEARN = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler
except ImportError:
    HAVE_SKLEARN = False
    print("[경고] scikit-learn 미설치 — 표 형태 모델 전체를 건너뜁니다. "
          "설치: pip install scikit-learn")

HAVE_XGB = True
try:
    from xgboost import XGBClassifier
except ImportError:
    HAVE_XGB = False
    print("[안내] xgboost 미설치 — XGBoost 모델 건너뜀. 설치: pip install xgboost")

HAVE_LGBM = True
try:
    from lightgbm import LGBMClassifier
except ImportError:
    HAVE_LGBM = False
    print("[안내] lightgbm 미설치 — LightGBM 모델 건너뜀. 설치: pip install lightgbm")

HAVE_SHAP = True
try:
    import shap
except ImportError:
    HAVE_SHAP = False
    print("[안내] shap 미설치 — SHAP 중요도 대신 permutation importance만 사용. "
          "설치: pip install shap")

HAVE_TORCH = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    HAVE_TORCH = False
    print("[경고] torch 미설치 — 시퀀스(딥러닝) 모델 6종 전체를 건너뜁니다. "
          "설치: pip install torch")

HAVE_CAPTUM = False
if HAVE_TORCH:
    try:
        from captum.attr import IntegratedGradients
        HAVE_CAPTUM = True
    except ImportError:
        print("[안내] captum 미설치 — Integrated Gradients 대신 단순 "
              "Gradient×Input saliency로 대체. 설치: pip install captum")

if HAVE_TORCH:
    torch.manual_seed(RANDOM_STATE)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[안내] torch 사용 디바이스: {DEVICE}")


# =============================================================================
# 1. §3 준실험 스크립트에서 그대로 재사용하는 핵심 함수 (완전히 동일)
# =============================================================================

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
    if len(pre) < MIN_WINDOW_POINTS or len(post) < MIN_WINDOW_POINTS:
        return np.nan
    vpre, vpost = np.var(pre, ddof=1), np.var(post, ddof=1)
    if vpre <= 0 or vpost <= 0 or not np.isfinite(vpre) or not np.isfinite(vpost):
        return np.nan
    return float(np.log(vpost / vpre))


def compute_effects(values: np.ndarray, pivot_idx: int) -> dict:
    """§3 스크립트와 완전히 동일 — level(|d|)과 diff/diff2(log 분산비)."""
    out = {"level": np.nan, "diff": np.nan, "diff2": np.nan}

    pre_level = values[BURN_IN:pivot_idx]
    post_level = values[pivot_idx:]
    out["level"] = cohens_d_abs(post_level, pre_level)

    diff_all = np.diff(values)
    split1 = max(pivot_idx - 1, 0)
    pre_diff = diff_all[max(BURN_IN - 1, 0):split1]
    post_diff = diff_all[split1:]
    out["diff"] = log_var_ratio(pre_diff, post_diff)

    diff2_all = np.diff(diff_all)
    split2 = max(pivot_idx - 2, 0)
    pre_diff2 = diff2_all[max(BURN_IN - 2, 0):split2]
    post_diff2 = diff2_all[split2:]
    out["diff2"] = log_var_ratio(pre_diff2, post_diff2)

    return out


FEATURES = ["level", "diff", "diff2"]


def build_3channel_window(values: np.ndarray, pivot_idx: int, half: int):
    """pivot 전후 half틱씩, level/diff/diff2를 3채널로 쌓은 (3, 2*half) 윈도우.
    각 채널은 윈도우 내부 자체 평균/표준편차로 z정규화한다(세그먼트/채널 간
    스케일 차이를 제거해, 모델이 '절대 크기'가 아니라 '상대적 변화 패턴'에
    집중하도록 함 — level/diff/diff2 세 채널 사이의 상대적 중요도를 묻는 이
    분석의 목적과 직접 부합)."""
    lo, hi = pivot_idx - half, pivot_idx + half
    if lo - 2 < 0 or hi > len(values):
        return None

    def znorm(x):
        mu, sd = np.mean(x), np.std(x)
        return (x - mu) / (sd + 1e-8)

    level_w = values[lo:hi]
    diff_all = np.diff(values)
    diff_w = diff_all[lo - 1: hi - 1]
    diff2_all = np.diff(diff_all)
    diff2_w = diff2_all[lo - 2: hi - 2]

    if len(level_w) != 2 * half or len(diff_w) != 2 * half or len(diff2_w) != 2 * half:
        return None

    return np.stack([znorm(level_w), znorm(diff_w), znorm(diff2_w)], axis=0).astype(np.float32)


# =============================================================================
# 2. 데이터 로드 + canonical onset 재구성 (§3와 동일)
# =============================================================================
section("1. 데이터 로드 및 라벨셋 구성 (§3 준실험 로직 재사용)")

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

# --- 채널별 정상 세그먼트의 onset 위치비율 풀 (§3와 동일 발상: 위약 pivot 추출용) ---
onset_ratio_pool_by_channel = {ch: [] for ch in FINAL_CHANNELS}

tab_rows = []      # 표 형태 데이터셋 (라벨 + level/diff/diff2)
seq_rows = []       # 시퀀스 데이터셋 (라벨 + (3, 2*half) 윈도우)
seq_group = []      # 시퀀스용 group id (channel_segment)
tab_group = []      # 표 형태용 group id

# 1) 이상 세그먼트(canonical onset 기준) — 라벨 1
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
    tab_rows.append({"channel": ch, "segment": sid, "label": 1, **eff})
    tab_group.append(f"{ch}_{sid}")
    onset_ratio_pool_by_channel[ch].append(onset_idx / n)

    win = build_3channel_window(values, onset_idx, SEQ_WINDOW_HALF)
    if win is not None:
        seq_rows.append({"channel": ch, "segment": sid, "label": 1, "window": win})
        seq_group.append(f"{ch}_{sid}")

# 2) 정상 세그먼트 — 무작위 pivot에서 위약 표본 K개씩, 라벨 0
for ch in FINAL_CHANNELS:
    ratio_pool = np.array(onset_ratio_pool_by_channel[ch])
    if len(ratio_pool) == 0:
        continue

    normal_meta = (
        seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)]
        .groupby("segment").size().rename("n_points").reset_index()
    )
    if len(normal_meta) > MAX_NORMAL_PER_CHANNEL:
        normal_meta = normal_meta.sample(n=MAX_NORMAL_PER_CHANNEL, random_state=RANDOM_STATE)

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
            tab_rows.append({"channel": ch, "segment": sid, "label": 0, **eff})
            tab_group.append(f"{ch}_{sid}")

            win = build_3channel_window(values, pivot_idx, SEQ_WINDOW_HALF)
            if win is not None:
                seq_rows.append({"channel": ch, "segment": sid, "label": 0, "window": win})
                seq_group.append(f"{ch}_{sid}")

tab_df = pd.DataFrame(tab_rows).dropna(subset=FEATURES).reset_index(drop=True)
# dropna로 인덱스가 바뀌므로 group도 동일 필터를 적용
_valid_mask = pd.DataFrame(tab_rows)[FEATURES].notna().all(axis=1).values
tab_group = np.array(tab_group)[_valid_mask]

print(f"\n표 형태 데이터셋: {len(tab_df)}행 (양성 {int((tab_df['label']==1).sum())} / "
      f"음성 {int((tab_df['label']==0).sum())})")
print(f"시퀀스 데이터셋: {len(seq_rows)}행")

tab_out = OUT_DIR / "modelclass_dataset_tabular.csv"
tab_df.to_csv(tab_out, index=False)
print(f"[저장] {tab_out}")


# =============================================================================
# 3. [표 형태] 11개 모델 정의 + 학습 + 중요도 추출
# =============================================================================
section("2. [표 형태] 11개 독립 모델 학습 및 특징중요도(SHAP + permutation) 추출")

tabular_results = []          # per-model importance rows
tabular_perf_rows = []        # per-model performance rows
tabular_fold_predictions = {} # model_name -> (y_true, y_prob) out-of-fold


def build_tabular_model_zoo():
    """family 태그와 함께 (이름, family, sklearn추정기, shap_kind) 튜플 리스트 반환.
    shap_kind: 'linear' | 'tree' | 'kernel' — SHAP explainer 선택에 사용."""
    zoo = []
    zoo.append(("LogReg_L2", "Linear",
                LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000), "linear"))
    zoo.append(("LogReg_L1", "Linear(sparse)",
                LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000), "linear"))
    zoo.append(("GaussianNB", "Probabilistic", GaussianNB(), "kernel"))
    zoo.append(("kNN", "Instance-based", KNeighborsClassifier(n_neighbors=15), "kernel"))
    zoo.append(("SVM_RBF", "Kernel",
                SVC(kernel="rbf", probability=True, class_weight="balanced"), "kernel"))
    zoo.append(("RandomForest", "Tree ensemble(bagging)",
                RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced",
                                        random_state=RANDOM_STATE, n_jobs=-1), "tree"))
    zoo.append(("ExtraTrees", "Tree ensemble(bagging,extra-random)",
                ExtraTreesClassifier(n_estimators=300, max_depth=6, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1), "tree"))
    zoo.append(("GradBoost_sklearn", "Tree ensemble(boosting)",
                GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE), "tree"))
    if HAVE_XGB:
        # [PATCH 2] use_label_encoder=False 제거 — 최신 xgboost(2.x)에서는
        # 이 파라미터가 완전히 삭제되어 TypeError를 유발한다. 최신 xgboost는
        # 이 옵션 없이도 동일하게 동작한다.
        zoo.append(("XGBoost", "Tree ensemble(boosting)",
                    XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  eval_metric="logloss", random_state=RANDOM_STATE,
                                  verbosity=0), "tree"))
    if HAVE_LGBM:
        zoo.append(("LightGBM", "Tree ensemble(boosting)",
                    LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   random_state=RANDOM_STATE, verbosity=-1), "tree"))
    # [PATCH 3] MLP_shallow 제외 — 실행 결과 AUC=0.403(무작위 이하)으로
    # 학습 실패가 확인되어, 학습되지 않은 모델의 특징중요도를 17개 모델
    # 합의도 분석에 포함시키면 결론을 왜곡할 위험이 있다. 이제 표 모델은
    # 총 10개, 전체 모델은 16개(표 10 + 시퀀스 6)로 재구성한다.
    # zoo.append(("MLP_shallow", "Neural net(shallow)",
    #             MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
    #                           random_state=RANDOM_STATE, early_stopping=True), "kernel"))
    return zoo


def _normalize_shap_values(sv, n_features: int):
    """[PATCH 1] SHAP 버전 간 shap_values() 반환 형태 차이를 흡수한다.

    - 구버전 SHAP: 이진분류에서 [class0_array, class1_array] 형태의 list를
      반환하는 explainer가 있었다(주로 TreeExplainer/KernelExplainer).
    - 신버전 SHAP: 다수의 explainer가 list 대신
      (n_samples, n_features, n_classes) 형태의 단일 3D ndarray를 반환한다.

    기존 코드는 list 케이스만(그것도 tree/kernel 경로에서만) 처리했기 때문에,
    신버전에서 3D ndarray가 그대로 통과되어 importance.shape이
    (n_features, n_classes)가 되고, 이후 float() 변환에서 크래시가 났다
    (GaussianNB에서 실제 재현됨: kind="kernel" 경로).

    이 함수는 list/2D ndarray/3D ndarray 세 가지 입력을 모두 받아, 항상
    (n_samples, n_features) 형태의 2D ndarray로 정규화해 반환한다. 정규화가
    불가능한(피처 수가 안 맞는 등) 경우 None을 반환한다.
    """
    if isinstance(sv, list):
        # 구버전: [class0, class1, ...] — 양성 클래스(1) 기준 사용
        sv = sv[1] if len(sv) > 1 else sv[0]

    sv = np.asarray(sv)

    if sv.ndim == 3:
        # 신버전: (n_samples, n_features, n_classes) — 양성 클래스(마지막 축의
        # index 1) 기준 사용. 클래스 축이 1개뿐이면 그대로 사용.
        sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
    elif sv.ndim == 1:
        # 극단적 edge case(단일 샘플 등) 방지
        sv = sv.reshape(1, -1)

    if sv.ndim != 2 or sv.shape[1] != n_features:
        return None

    return sv


def compute_shap_importance(model, kind: str, X_train: np.ndarray, X_explain: np.ndarray):
    """3피처뿐이라 어떤 explainer를 써도 비용이 낮다. kind에 따라 가장 적합한
    explainer를 선택하고, 실패 시 None을 반환(호출부에서 permutation으로 대체).
    [PATCH 1] shap_values() 반환 형태 정규화를 모든 경로(tree/linear/kernel)에
    공통 적용하도록 수정."""
    if not HAVE_SHAP:
        return None
    try:
        if kind == "tree":
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_explain)
        elif kind == "linear":
            explainer = shap.LinearExplainer(model, X_train)
            sv = explainer.shap_values(X_explain)
        else:
            background = shap.kmeans(X_train, min(20, len(X_train)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            sv = explainer.shap_values(X_explain, nsamples=100)

        n_features = X_explain.shape[1]
        sv = _normalize_shap_values(sv, n_features)
        if sv is None:
            print(f"    [SHAP 반환 shape 정규화 실패 -> permutation으로 대체]")
            return None

        importance = np.abs(sv).mean(axis=0)          # (n_features,)
        importance = np.asarray(importance).reshape(-1)
        if importance.shape[0] != n_features:
            print(f"    [SHAP importance shape 불일치({importance.shape}) -> permutation으로 대체]")
            return None

        total = importance.sum()
        return (importance / total) if total > 0 else None
    except Exception as e:
        print(f"    [SHAP 실패 -> permutation으로 대체] {e}")
        return None


if HAVE_SKLEARN and len(tab_df) > 50:
    X_all = tab_df[FEATURES].values.astype(float)
    y_all = tab_df["label"].values.astype(int)
    groups_all = tab_group

    gkf = GroupKFold(n_splits=5)
    model_zoo = build_tabular_model_zoo()
    print(f"등록된 표 형태 모델: {len(model_zoo)}개 -> {[m[0] for m in model_zoo]}")

    for name, family, estimator_template, shap_kind in model_zoo:
        print(f"\n[{name}] ({family}) 학습 중...")
        oof_prob = np.full(len(y_all), np.nan)
        fold_importances = []

        for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_all, groups_all)):
            X_tr, X_te = X_all[tr_idx], X_all[te_idx]
            y_tr, y_te = y_all[tr_idx], y_all[te_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

            model = estimator_template.__class__(**estimator_template.get_params())
            try:
                model.fit(X_tr_s, y_tr)
            except Exception as e:
                print(f"    fold {fold_i} 학습 실패: {e}")
                continue

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_te_s)[:, 1]
            else:
                prob = model.decision_function(X_te_s)
            oof_prob[te_idx] = prob

            explain_idx = np.random.choice(len(X_te_s), size=min(EXPLAIN_SAMPLE_SIZE, len(X_te_s)), replace=False)
            shap_imp = compute_shap_importance(model, shap_kind, X_tr_s, X_te_s[explain_idx])

            if shap_imp is None:
                # SHAP 불가 시 permutation importance로 대체 (모델-불가지론적, 항상 가능)
                perm = permutation_importance(model, X_te_s, y_te, n_repeats=20,
                                               random_state=RANDOM_STATE, scoring="roc_auc")
                imp = np.clip(perm.importances_mean, 0, None)
                total = imp.sum()
                shap_imp = (imp / total) if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3])

            fold_importances.append(shap_imp)

        if len(fold_importances) == 0:
            print(f"    [{name}] 모든 fold 실패 — 건너뜀")
            continue

        mean_importance = np.mean(fold_importances, axis=0)
        for feat, imp in zip(FEATURES, mean_importance):
            tabular_results.append({"model": name, "family": family, "feature": feat,
                                     "importance_share": float(imp)})

        valid = ~np.isnan(oof_prob)
        if valid.sum() > 10 and len(np.unique(y_all[valid])) == 2:
            auc = roc_auc_score(y_all[valid], oof_prob[valid])
            fpr, tpr, thr = roc_curve(y_all[valid], oof_prob[valid])
            youden = tpr - fpr
            best_i = int(np.argmax(youden))
            tabular_perf_rows.append({
                "model": name, "family": family, "auc": float(auc),
                "recall_at_youden": float(tpr[best_i]), "fa_rate_at_youden": float(fpr[best_i]),
            })
            tabular_fold_predictions[name] = (y_all[valid].copy(), oof_prob[valid].copy())
            print(f"    AUC={auc:.3f}  중요도(level/diff/diff2)="
                  f"{mean_importance[0]:.3f}/{mean_importance[1]:.3f}/{mean_importance[2]:.3f}")
else:
    print("[건너뜀] scikit-learn 미설치 또는 데이터 부족")

tabular_importance_df = pd.DataFrame(tabular_results)
tabular_perf_df = pd.DataFrame(tabular_perf_rows)


# =============================================================================
# 4. [시퀀스 형태] 6개 딥러닝 모델 정의 + 학습 + saliency 추출
# =============================================================================
section("3. [시퀀스 형태] 6개 독립 아키텍처 학습 및 채널별 saliency 추출")

deep_results = []
deep_perf_rows = []

if HAVE_TORCH and len(seq_rows) > 50:

    class WindowDataset(Dataset):
        def __init__(self, rows):
            self.X = np.stack([r["window"] for r in rows]).astype(np.float32)  # (N, 3, T)
            self.y = np.array([r["label"] for r in rows], dtype=np.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

    T = 2 * SEQ_WINDOW_HALF

    # --- 6개 아키텍처 정의 (전부 입력 (B, 3, T) -> 출력 (B,) 로짓) ---

    class CNN1D(nn.Module):
        def __init__(self, c_in=3, hidden=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c_in, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.net(x).squeeze(-1)
            return self.head(h).squeeze(-1)

    class TCNBlock(nn.Module):
        """확장 인과 컨볼루션(dilated causal conv) 잔차 블록."""
        def __init__(self, c_in, c_out, dilation):
            super().__init__()
            pad = (5 - 1) * dilation
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=5, padding=pad, dilation=dilation)
            self.pad = pad
            self.relu = nn.ReLU()
            self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x):
            out = self.conv(x)[:, :, :-self.pad] if self.pad > 0 else self.conv(x)
            return self.relu(out + self.res(x))

    class TCN(nn.Module):
        def __init__(self, c_in=3, hidden=32):
            super().__init__()
            self.blocks = nn.Sequential(
                TCNBlock(c_in, hidden, dilation=1),
                TCNBlock(hidden, hidden, dilation=2),
                TCNBlock(hidden, hidden, dilation=4),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.blocks(x)
            h = self.pool(h).squeeze(-1)
            return self.head(h).squeeze(-1)

    class BiLSTM(nn.Module):
        def __init__(self, c_in=3, hidden=32):
            super().__init__()
            self.rnn = nn.LSTM(c_in, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Linear(hidden * 2, 1)

        def forward(self, x):
            xt = x.permute(0, 2, 1)  # (B, T, 3)
            out, _ = self.rnn(xt)
            pooled = out.mean(dim=1)
            return self.head(pooled).squeeze(-1)

    class BiGRU(nn.Module):
        def __init__(self, c_in=3, hidden=32):
            super().__init__()
            self.rnn = nn.GRU(c_in, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Linear(hidden * 2, 1)

        def forward(self, x):
            xt = x.permute(0, 2, 1)
            out, _ = self.rnn(xt)
            pooled = out.mean(dim=1)
            return self.head(pooled).squeeze(-1)

    class TinyTransformer(nn.Module):
        def __init__(self, c_in=3, d_model=32, nhead=4, num_layers=2, max_len=200):
            super().__init__()
            self.proj = nn.Linear(c_in, d_model)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=64, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            xt = x.permute(0, 2, 1)               # (B, T, 3)
            h = self.proj(xt) + self.pe[:, :xt.size(1), :]
            h = self.encoder(h)
            pooled = h.mean(dim=1)
            return self.head(pooled).squeeze(-1)

    class LightweightMambaBlock(nn.Module):
        """공식 mamba_ssm 패키지(CUDA 커널 필요)가 없는 환경에서도 동작하도록,
        '입력에 따라 상태전이가 동적으로 변한다(selective)'는 Mamba의 핵심
        아이디어만을 순수 PyTorch 순차 스캔으로 단순화 구현한 대체 블록이다.
        공식 구현의 속도·용량을 재현하지 못하며, 이 분석에서는 오직 'SSM 계열
        아키텍처가 다른 아키텍처(CNN/RNN/Transformer)와 비교해 diff/diff²에
        대해 유사한 특징귀속 패턴을 보이는가'를 확인하는 최소 작동 버전으로만
        사용한다. mamba_ssm이 설치된 환경에서는 이 클래스를 공식 Mamba 블록으로
        교체해 재실행할 수 있다."""
        def __init__(self, c_in=3, d_model=32, d_state=16):
            super().__init__()
            self.proj_in = nn.Linear(c_in, d_model)
            self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.1)   # -exp(A_log) < 0 로 안정화
            self.B_proj = nn.Linear(d_model, d_state)
            self.C_proj = nn.Linear(d_model, d_state)
            self.delta_proj = nn.Linear(d_model, d_model)
            self.head = nn.Linear(d_model, 1)
            self.d_model, self.d_state = d_model, d_state

        def forward(self, x):
            xt = x.permute(0, 2, 1)                        # (B, T, 3)
            u = self.proj_in(xt)                            # (B, T, d_model)
            B_, T_, D = u.shape
            delta = torch.nn.functional.softplus(self.delta_proj(u))  # (B,T,D) 입력의존 시간간격
            A = -torch.exp(self.A_log)                      # (D, d_state), 항상 음수(안정성)
            Bc = self.B_proj(u)                              # (B,T,d_state) 입력의존 B
            Cc = self.C_proj(u)                              # (B,T,d_state) 입력의존 C

            h = torch.zeros(B_, D, self.d_state, device=u.device)
            ys = []
            for t in range(T_):
                dA = torch.exp(delta[:, t, :].unsqueeze(-1) * A.unsqueeze(0))       # (B,D,d_state)
                dBu = delta[:, t, :].unsqueeze(-1) * Bc[:, t, :].unsqueeze(1) * u[:, t, :].unsqueeze(-1)
                h = dA * h + dBu                                                    # 선택적 상태 갱신
                y_t = (h * Cc[:, t, :].unsqueeze(1)).sum(-1)                        # (B,D)
                ys.append(y_t)
            y = torch.stack(ys, dim=1)                       # (B,T,D)
            pooled = y.mean(dim=1)
            return self.head(pooled).squeeze(-1)

    def build_deep_model_zoo():
        return [
            ("CNN1D", "Convolutional", CNN1D),
            ("TCN", "Convolutional(dilated causal)", TCN),
            ("BiLSTM", "Recurrent", BiLSTM),
            ("BiGRU", "Recurrent", BiGRU),
            ("TinyTransformer", "Attention", TinyTransformer),
            ("LightMamba", "State-space(SSM)", LightweightMambaBlock),
        ]

    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def eval_model(model, loader, criterion):
        model.eval()
        total_loss, probs, ys = 0.0, [], []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(yb.cpu().numpy())
        return total_loss / len(loader.dataset), np.concatenate(probs), np.concatenate(ys)

    def integrated_gradients_saliency(model, x, baseline=None, steps=32):
        """captum이 없을 때 쓰는 수동 Integrated Gradients 구현.
        baseline=0(전 채널 평균 신호에 해당, z정규화 후이므로 0이 자연스러운 기준선).

        [PATCH 3] 이 함수는 model.eval() 상태의 모델에 대해 backward를
        호출한다. RNN(nn.LSTM/nn.GRU) 기반 모델을 cuDNN 백엔드(GPU)로 돌릴
        경우, cuDNN의 RNN 커널은 eval 모드에서 backward에 필요한 중간
        텐서를 보존하지 않아 "cudnn RNN backward can only be called in
        training mode" 오류가 난다. 호출부(아래 학습 루프)에서
        torch.backends.cudnn.flags(enabled=False) 컨텍스트 안에서 이
        함수를 호출하도록 수정했으므로, 이 함수 자체는 변경할 필요가 없다
        (cudnn 비활성화는 호출 시점의 전역 백엔드 설정이라 여기서 별도
        처리가 필요 없음)."""
        model.eval()
        if baseline is None:
            baseline = torch.zeros_like(x)
        total_grad = torch.zeros_like(x)
        for alpha in np.linspace(0, 1, steps):
            xi = (baseline + alpha * (x - baseline)).clone().requires_grad_(True)
            out = model(xi).sum()
            grad, = torch.autograd.grad(out, xi, retain_graph=False)
            total_grad += grad
        avg_grad = total_grad / steps
        return (avg_grad * (x - baseline)).detach()

    all_group = np.array(seq_group)
    unique_groups = np.unique(all_group)
    n_folds_deep = 5
    fold_assign = {g: i % n_folds_deep for i, g in enumerate(rng.permutation(unique_groups))}
    fold_of_row = np.array([fold_assign[g] for g in all_group])

    model_zoo_deep = build_deep_model_zoo()
    print(f"등록된 시퀀스 모델: {len(model_zoo_deep)}개 -> {[m[0] for m in model_zoo_deep]}")

    y_seq = np.array([r["label"] for r in seq_rows], dtype=np.float32)
    n_pos, n_neg = int(y_seq.sum()), int((1 - y_seq).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    for name, family, ModelClass in model_zoo_deep:
        print(f"\n[{name}] ({family}) 학습 중...")
        fold_saliency = []
        oof_prob = np.full(len(y_seq), np.nan)

        for fold_i in range(n_folds_deep):
            te_idx = np.where(fold_of_row == fold_i)[0]
            tr_idx = np.where(fold_of_row != fold_i)[0]
            if len(te_idx) < 5 or len(tr_idx) < 20:
                continue

            train_rows = [seq_rows[i] for i in tr_idx]
            test_rows = [seq_rows[i] for i in te_idx]
            train_ds, test_ds = WindowDataset(train_rows), WindowDataset(test_rows)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

            model = ModelClass().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            best_val_loss, patience, patience_ctr = np.inf, 6, 0
            best_state = None
            for epoch in range(60):
                train_one_epoch(model, train_loader, optimizer, criterion)
                val_loss, _, _ = eval_model(model, test_loader, criterion)
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss, patience_ctr = val_loss, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)

            _, probs, ys = eval_model(model, test_loader, criterion)
            oof_prob[te_idx] = probs

            explain_n = min(EXPLAIN_SAMPLE_SIZE, len(test_ds))
            explain_idx = np.random.choice(len(test_ds), size=explain_n, replace=False)
            X_explain = torch.from_numpy(np.stack([test_ds.X[i] for i in explain_idx])).to(DEVICE)

            # [PATCH 3] cuDNN RNN backward 오류 수정:
            # eval() 상태의 nn.LSTM/nn.GRU에 대해 cuDNN 백엔드로 backward를
            # 호출하면 "cudnn RNN backward can only be called in training
            # mode" RuntimeError가 발생한다(BiLSTM/BiGRU에서 재현됨). 이는
            # cuDNN 커널이 eval 모드에서 backward용 중간 텐서를 보존하지
            # 않기 때문이며, model.train()으로 전환하면 dropout/batchnorm이
            # 있는 모델의 경우 결과가 오염될 위험이 있다(이 모델들엔 없지만
            # 일반성을 위해 피함). 대신 attribution 계산 구간에서만
            # cuDNN을 비활성화해 일반(generic) RNN backward 구현을 쓰도록
            # 강제한다 — 수치 결과는 동일하고 속도만 느려진다(설명 샘플
            # 최대 EXPLAIN_SAMPLE_SIZE=400개뿐이라 비용 미미).
            with torch.backends.cudnn.flags(enabled=False):
                if HAVE_CAPTUM:
                    ig = IntegratedGradients(model)
                    attributions = ig.attribute(X_explain, target=None, n_steps=32)
                else:
                    attributions = integrated_gradients_saliency(model, X_explain)

            chan_importance = attributions.abs().sum(dim=2).mean(dim=0).detach().cpu().numpy()  # (3,)
            total = chan_importance.sum()
            chan_importance = (chan_importance / total) if total > 0 else np.array([1/3, 1/3, 1/3])
            fold_saliency.append(chan_importance)

        if len(fold_saliency) == 0:
            print(f"    [{name}] 모든 fold 실패 — 건너뜀")
            continue

        mean_saliency = np.mean(fold_saliency, axis=0)
        for feat, imp in zip(FEATURES, mean_saliency):
            deep_results.append({"model": name, "family": family, "feature": feat,
                                  "importance_share": float(imp)})

        valid = ~np.isnan(oof_prob)
        if valid.sum() > 10 and len(np.unique(y_seq[valid])) == 2:
            auc = roc_auc_score(y_seq[valid], oof_prob[valid])
            fpr, tpr, thr = roc_curve(y_seq[valid], oof_prob[valid])
            youden = tpr - fpr
            best_i = int(np.argmax(youden))
            deep_perf_rows.append({
                "model": name, "family": family, "auc": float(auc),
                "recall_at_youden": float(tpr[best_i]), "fa_rate_at_youden": float(fpr[best_i]),
            })
            print(f"    AUC={auc:.3f}  saliency(level/diff/diff2)="
                  f"{mean_saliency[0]:.3f}/{mean_saliency[1]:.3f}/{mean_saliency[2]:.3f}")
else:
    print("[건너뜀] torch 미설치 또는 시퀀스 데이터 부족")

deep_importance_df = pd.DataFrame(deep_results)
deep_perf_df = pd.DataFrame(deep_perf_rows)


# =============================================================================
# 5. 표+시퀀스 통합 + 성능 리더보드 (BOCPD 대조 포함) 저장
# =============================================================================
section("4. 16개 모델 통합 결과 + BOCPD(층4) 대비 성능 리더보드 (MLP_shallow 제외)")

importance_df = pd.concat([tabular_importance_df, deep_importance_df], ignore_index=True)
importance_out = OUT_DIR / "modelclass_feature_importance.csv"
importance_df.to_csv(importance_out, index=False)
print(f"[저장] {importance_out}  ({importance_df['model'].nunique()}개 모델)")

perf_df = pd.concat([tabular_perf_df, deep_perf_df], ignore_index=True)
bocpd_pooled_recall = np.mean([v["recall"] for v in REFERENCE_BOCPD_PERFORMANCE.values()])
bocpd_pooled_fa = np.mean([v["fa_rate"] for v in REFERENCE_BOCPD_PERFORMANCE.values()])
perf_df = pd.concat([perf_df, pd.DataFrame([{
    "model": "BOCPD_tau_Youden(층4 참조값)", "family": "Reference(확률모델,비교대상)",
    "auc": np.nan, "recall_at_youden": bocpd_pooled_recall, "fa_rate_at_youden": bocpd_pooled_fa,
}])], ignore_index=True)

perf_out = OUT_DIR / "modelclass_performance_leaderboard.csv"
perf_df.sort_values("auc", ascending=False).to_csv(perf_out, index=False)
print(perf_df.sort_values("auc", ascending=False).to_string(index=False))
print(f"\n[저장] {perf_out}")


# =============================================================================
# 6. 교차모델 합의도 분석 — Kendall's W, 부호검정, 모델군별 일치도
# =============================================================================
section("5. 교차모델 합의도(agreement) 분석")

if len(importance_df) > 0:
    pivot = importance_df.pivot_table(index=["model", "family"], columns="feature",
                                       values="importance_share").reset_index()
    pivot = pivot.reindex(columns=["model", "family"] + FEATURES)

    # 모델별 순위 (1=가장 중요)
    rank_cols = []
    for f in FEATURES:
        pivot[f"rank_{f}"] = pivot[f].rank(ascending=False, axis=0)
    # 위 rank는 열방향(모델 간) 순위라 의미가 다름 -> 행방향(모델 내부 3피처 간) 순위로 재계산
    rank_matrix = pivot[FEATURES].rank(axis=1, ascending=False, method="average").values  # (n_model, 3)

    n_models, n_items = rank_matrix.shape

    # --- Kendall's W (일치도 계수) ---
    rank_sums = rank_matrix.sum(axis=0)  # (3,) 각 피처가 받은 순위 합
    S = np.sum((rank_sums - rank_sums.mean()) ** 2)
    W = (12 * S) / (n_models ** 2 * (n_items ** 3 - n_items)) if n_models > 0 else np.nan
    chi2_stat = n_models * (n_items - 1) * W
    chi2_p = 1 - stats.chi2.cdf(chi2_stat, df=n_items - 1)

    print(f"Kendall's W(일치도 계수) = {W:.4f}  (0=완전 무작위, 1=완전 일치)")
    print(f"카이제곱 근사 검정: chi2={chi2_stat:.3f}, df={n_items-1}, p={chi2_p:.6f}")

    # --- 부호검정: level이 3개 피처 중 '가장 중요'로 뽑히지 않은 모델의 비율 ---
    is_level_top = (pivot["level"] >= pivot[["diff", "diff2"]].max(axis=1))
    n_level_not_top = int((~is_level_top).sum())
    binom_result = stats.binomtest(n_level_not_top, n=n_models, p=2 / 3, alternative="greater")

    print(f"\nlevel이 최고 중요 피처가 아닌 모델: {n_level_not_top}/{n_models}개")
    print(f"이항검정(귀무가설: 무작위 순위에서 기대비율 2/3): p={binom_result.pvalue:.6f}")

    # --- 모델군(family)별 일치도 ---
    family_summary = pivot.groupby("family").agg(
        n_models=("model", "count"),
        mean_level=("level", "mean"), mean_diff=("diff", "mean"), mean_diff2=("diff2", "mean"),
    ).reset_index()
    family_summary["diff_plus_diff2_beats_level"] = (
        (family_summary["mean_diff"] + family_summary["mean_diff2"]) > family_summary["mean_level"]
    )

    print("\n모델군(family)별 평균 중요도:")
    print(family_summary.to_string(index=False))

    agreement_summary = pd.DataFrame([{
        "n_models": n_models,
        "kendalls_W": W,
        "chi2_stat": chi2_stat, "chi2_p_value": chi2_p,
        "n_models_level_not_top": n_level_not_top,
        "binomial_p_value(level_not_top_gt_chance)": binom_result.pvalue,
    }])
    agreement_out = OUT_DIR / "modelclass_agreement_summary.csv"
    agreement_summary.to_csv(agreement_out, index=False)
    family_summary.to_csv(OUT_DIR / "modelclass_family_summary.csv", index=False)
    print(f"\n[저장] {agreement_out}")
else:
    print("[건너뜀] 중요도 데이터 없음")
    pivot = pd.DataFrame()


# =============================================================================
# 7. 표 모델 부트스트랩 안정성 (계층적 리샘플, N_BOOT_TABULAR회)
# =============================================================================
section(f"6. 표 모델 부트스트랩 안정성 (N_BOOT_TABULAR={N_BOOT_TABULAR})")

boot_rows = []
if HAVE_SKLEARN and len(tab_df) > 50:
    X_all = tab_df[FEATURES].values.astype(float)
    y_all = tab_df["label"].values.astype(int)
    pos_idx_pool = np.where(y_all == 1)[0]
    neg_idx_pool = np.where(y_all == 0)[0]

    # 부트스트랩은 계산비용을 낮추기 위해 트리 계열(RF) 하나를 대표로,
    # 그리고 선형 계열(LogReg) 하나를 대표로 삼아 두 서로 다른 귀납편향에서
    # 안정성을 확인한다 (11개 전부를 200회 재학습하는 것은 과도한 비용).
    representative_models = [
        ("RandomForest_boot", RandomForestClassifier(n_estimators=200, max_depth=6,
                                                       class_weight="balanced",
                                                       random_state=RANDOM_STATE, n_jobs=-1)),
        ("LogReg_boot", LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000)),
    ]

    for name, estimator_template in representative_models:
        diff_wins = np.zeros(N_BOOT_TABULAR, dtype=bool)
        for b in range(N_BOOT_TABULAR):
            boot_pos = rng.choice(pos_idx_pool, size=len(pos_idx_pool), replace=True)
            boot_neg = rng.choice(neg_idx_pool, size=len(neg_idx_pool), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])

            X_b, y_b = X_all[boot_idx], y_all[boot_idx]
            scaler = StandardScaler().fit(X_b)
            X_bs = scaler.transform(X_b)

            model = estimator_template.__class__(**estimator_template.get_params())
            try:
                model.fit(X_bs, y_b)
            except Exception:
                continue

            perm = permutation_importance(model, X_bs, y_b, n_repeats=10,
                                           random_state=b, scoring="roc_auc")
            imp = np.clip(perm.importances_mean, 0, None)
            diff_wins[b] = (imp[1] + imp[2]) > imp[0]  # diff+diff2 > level

        frac_diff_wins = float(np.mean(diff_wins))
        boot_rows.append({"model": name, "n_boot": N_BOOT_TABULAR,
                           "frac_diff_or_diff2_beats_level": frac_diff_wins})
        print(f"[{name}] {N_BOOT_TABULAR}회 중 diff/diff2가 level을 능가한 비율: {frac_diff_wins:.1%}")

boot_df = pd.DataFrame(boot_rows)
boot_out = OUT_DIR / "modelclass_bootstrap_stability.csv"
boot_df.to_csv(boot_out, index=False)
print(f"\n[저장] {boot_out}")


# =============================================================================
# 8. 최종 요약 리포트
# =============================================================================
section("7. 최종 요약 리포트")

report_lines = []
report_lines.append("=" * 90)
report_lines.append("모델군 교차검증(Model-Class Robustness Analysis) — 최종 요약")
report_lines.append("=" * 90)
report_lines.append("")
report_lines.append(f"총 {importance_df['model'].nunique() if len(importance_df) else 0}개 독립 모델 "
                     f"(표 형태 {tabular_importance_df['model'].nunique() if len(tabular_importance_df) else 0}개 "
                     f"+ 시퀀스 형태 {deep_importance_df['model'].nunique() if len(deep_importance_df) else 0}개)")
report_lines.append("")

if len(pivot) > 0:
    report_lines.append(f"Kendall's W = {W:.4f} (0=무작위, 1=완전 일치)")
    report_lines.append(f"  -> 카이제곱 근사 검정 p = {chi2_p:.6f}")
    report_lines.append(f"level이 최고 중요 피처가 아닌 모델: {n_level_not_top}/{n_models}개 "
                         f"(이항검정 p = {binom_result.pvalue:.6f})")
    report_lines.append("")
    report_lines.append("모델군(family)별 diff+diff2 vs level 우세 여부:")
    for _, r in family_summary.iterrows():
        verdict = "diff/diff2 우세" if r["diff_plus_diff2_beats_level"] else "level 우세(예외)"
        report_lines.append(f"  - {r['family']}: {verdict} "
                             f"(level={r['mean_level']:.3f}, diff={r['mean_diff']:.3f}, diff2={r['mean_diff2']:.3f})")

report_lines.append("")
report_lines.append("[해석 가이드]")
report_lines.append(
    "1) Kendall's W가 높고(예: >0.6) p값이 유의하면, 16개(MLP_shallow는 학습\n"
    "   실패로 제외)의 서로 다른 귀납편향을\n"
    "   가진 모델들이 우연이라고 보기 어려운 수준으로 같은 순위(diff/diff2 > level)에\n"
    "   합의하고 있다는 뜻이다. 이는 층3의 '준실험적 pre/post 비교' 결론을 완전히\n"
    "   독립적인 방법론으로 재현한 것이다."
)
report_lines.append(
    "2) 모델군별 표에서 예외(level 우세)가 있는 모델군이 있다면, 그 모델군의 귀납편향\n"
    "   (예: 거리기반 모델은 스케일에 민감, 커널모델은 국소성에 민감 등)과 연결지어\n"
    "   설명해야 한다 — 무시하지 말고 논문에 그 원인 가설을 명시할 것."
)
report_lines.append(
    "3) 표 형태(사람이 설계한 요약통계량)와 시퀀스 형태(원시 3채널 시계열, saliency)가\n"
    "   같은 결론에 도달했다면, 이는 '요약통계량 설계가 결론을 편향시켰다'는 우려를\n"
    "   배제하는 가장 강력한 근거다 — 두 표현 방식의 family 열을 나란히 비교해 확인할 것."
)
report_lines.append(
    "4) 부트스트랩 안정성(frac_diff_or_diff2_beats_level)이 두 대표 모델(RF, LogReg)\n"
    "   모두에서 90% 이상이면, 이 결론이 특정 표본 구성에 우연히 의존한 것이 아니라는\n"
    "   근거로 추가할 수 있다."
)
report_lines.append(
    "5) 성능 리더보드(AUC)에서 BOCPD_tau_Youden(층4 참조값)과 비교했을 때, 만약 일부\n"
    "   독립 모델이 BOCPD보다 더 높은 AUC를 보인다면, 이는 'BOCPD가 최적'이라는 주장을\n"
    "   약화시키는 것이 아니라 '실시간 온라인 확률적 변화점 탐지(BOCPD)'와 '사후적\n"
    "   배치 분류(표/시퀀스 모델)'는 애초에 다른 문제를 푸는 것이라는 점을 논문에\n"
    "   명시해야 한다 — 이 비교의 목적은 '누가 이기는가'가 아니라 '같은 신호를\n"
    "   본다고 하는가'이다."
)

report_text = "\n".join(report_lines)
report_out = OUT_DIR / "modelclass_final_report.txt"
report_out.write_text(report_text, encoding="utf-8")
print(report_text)
print(f"\n[저장] {report_out}")

section("완료")
print(
    "모든 산출물은 OUT_DIR(causal_pipeline_outputs) 아래에 저장되었습니다.\n"
    "논문 서술 시 modelclass_agreement_summary.csv(Kendall's W, 이항검정)와\n"
    "modelclass_family_summary.csv(모델군별 breakdown)를 주 결과표로,\n"
    "modelclass_bootstrap_stability.csv를 견고성 근거로 사용하는 것을 권장합니다."
)

nohup: ignoring input
Background dataset has 3996 samples but max_samples=100. Subsampling to 100 samples for SHAP value computation. To use all samples, set max_samples=3996 when initializing the masker.
Background dataset has 3997 samples but max_samples=100. Subsampling to 100 samples for SHAP value computation. To use all samples, set max_samples=3997 when initializing the masker.
Background dataset has 3997 samples but max_samples=100. Subsampling to 100 samples for SHAP value computation. To use all samples, set max_samples=3997 when initializing the masker.
Background dataset has 3997 samples but max_samples=100. Subsampling to 100 samples for SHAP value computation. To use all samples, set max_samples=3997 when initializing the masker.
Background dataset has 3997 samples but max_samples=100. Subsampling to 100 samples for SHAP value computation. To use all samples, set max_samples=3997 when initializing the masker.
Background dataset has 3996 samples but max_samples=100. Subsampling to 100 samples for SHAP value computation. To use all samples, set max_samples=3996 when initializing the masker.
Background dataset has 3997 samples but max_samples=100. Subsampling to 100 samples for SHAP value computation. To use all samples, set max_samples=3997 when initializing the masker.
Background dataset has 3997 samples but max_samples=100. Subsampling to 100 samples for SHAP value computation. To use all samples, set max_samples=3997 when initializing the masker.
Background dataset has 3997 samples but max_samples=100. Subsampling to 100 samples for SHAP value computation. To use all samples, set max_samples=3997 when initializing the masker.
Background dataset has 3997 samples but max_samples=100. Subsampling to 100 samples for SHAP value computation. To use all samples, set max_samples=3997 when initializing the masker.

==========================================================================================
0. 소프트 의존성(라이브러리) 점검
==========================================================================================
[안내] torch 사용 디바이스: cuda

==========================================================================================
1. 데이터 로드 및 라벨셋 구성 (§3 준실험 로직 재사용)
==========================================================================================
신뢰도 high+medium 이상 세그먼트: 177개

표 형태 데이터셋: 4996행 (양성 168 / 음성 4828)
시퀀스 데이터셋: 2733행
[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/modelclass_dataset_tabular.csv

==========================================================================================
2. [표 형태] 11개 독립 모델 학습 및 특징중요도(SHAP + permutation) 추출
==========================================================================================
등록된 표 형태 모델: 10개 -> ['LogReg_L2', 'LogReg_L1', 'GaussianNB', 'kNN', 'SVM_RBF', 'RandomForest', 'ExtraTrees', 'GradBoost_sklearn', 'XGBoost', 'LightGBM']

[LogReg_L2] (Linear) 학습 중...
    AUC=0.920  중요도(level/diff/diff2)=0.060/0.407/0.534

[LogReg_L1] (Linear(sparse)) 학습 중...
    AUC=0.920  중요도(level/diff/diff2)=0.057/0.412/0.531

[GaussianNB] (Probabilistic) 학습 중...

  0%|          | 0/400 [00:00<?, ?it/s]
  8%|▊         | 32/400 [00:00<00:01, 313.12it/s]
 16%|█▌        | 64/400 [00:00<00:01, 315.47it/s]
 24%|██▍       | 96/400 [00:00<00:00, 310.95it/s]
 32%|███▏      | 128/400 [00:00<00:00, 314.44it/s]
 40%|████      | 160/400 [00:00<00:00, 316.24it/s]
 48%|████▊     | 192/400 [00:00<00:00, 312.71it/s]
 56%|█████▌    | 224/400 [00:00<00:00, 314.58it/s]
 64%|██████▍   | 256/400 [00:00<00:00, 315.60it/s]
 72%|███████▏  | 288/400 [00:00<00:00, 314.36it/s]
 80%|████████  | 320/400 [00:01<00:00, 313.48it/s]
 88%|████████▊ | 352/400 [00:01<00:00, 314.85it/s]
 96%|█████████▌| 384/400 [00:01<00:00, 315.63it/s]
100%|██████████| 400/400 [00:01<00:00, 314.21it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  8%|▊         | 32/400 [00:00<00:01, 314.64it/s]
 16%|█▌        | 64/400 [00:00<00:01, 299.13it/s]
 24%|██▍       | 96/400 [00:00<00:00, 307.95it/s]
 32%|███▏      | 127/400 [00:00<00:00, 308.04it/s]
 40%|███▉      | 158/400 [00:00<00:00, 303.38it/s]
 48%|████▊     | 191/400 [00:00<00:00, 309.19it/s]
 56%|█████▌    | 223/400 [00:00<00:00, 310.18it/s]
 64%|██████▍   | 255/400 [00:00<00:00, 307.99it/s]
 72%|███████▏  | 286/400 [00:00<00:00, 306.37it/s]
 80%|███████▉  | 318/400 [00:01<00:00, 307.58it/s]
 88%|████████▊ | 350/400 [00:01<00:00, 308.63it/s]
 95%|█████████▌| 381/400 [00:01<00:00, 306.80it/s]
100%|██████████| 400/400 [00:01<00:00, 307.13it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  8%|▊         | 31/400 [00:00<00:01, 303.37it/s]
 16%|█▌        | 62/400 [00:00<00:01, 301.20it/s]
 24%|██▎       | 94/400 [00:00<00:00, 307.54it/s]
 32%|███▏      | 126/400 [00:00<00:00, 309.77it/s]
 40%|███▉      | 158/400 [00:00<00:00, 311.19it/s]
 48%|████▊     | 190/400 [00:00<00:00, 307.16it/s]
 56%|█████▌    | 222/400 [00:00<00:00, 308.99it/s]
 64%|██████▎   | 254/400 [00:00<00:00, 309.60it/s]
 71%|███████▏  | 285/400 [00:00<00:00, 306.66it/s]
 79%|███████▉  | 317/400 [00:01<00:00, 308.43it/s]
 87%|████████▋ | 349/400 [00:01<00:00, 309.28it/s]
 95%|█████████▌| 380/400 [00:01<00:00, 309.45it/s]
100%|██████████| 400/400 [00:01<00:00, 307.30it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  8%|▊         | 32/400 [00:00<00:01, 313.67it/s]
 16%|█▌        | 64/400 [00:00<00:01, 316.74it/s]
 24%|██▍       | 96/400 [00:00<00:00, 317.21it/s]
 32%|███▏      | 128/400 [00:00<00:00, 318.23it/s]
 40%|████      | 160/400 [00:00<00:00, 316.85it/s]
 48%|████▊     | 192/400 [00:00<00:00, 317.61it/s]
 56%|█████▌    | 224/400 [00:00<00:00, 317.99it/s]
 64%|██████▍   | 256/400 [00:00<00:00, 318.09it/s]
 72%|███████▏  | 288/400 [00:00<00:00, 317.68it/s]
 80%|████████  | 320/400 [00:01<00:00, 317.43it/s]
 88%|████████▊ | 352/400 [00:01<00:00, 317.01it/s]
 96%|█████████▌| 384/400 [00:01<00:00, 317.00it/s]
100%|██████████| 400/400 [00:01<00:00, 317.29it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  8%|▊         | 32/400 [00:00<00:01, 312.85it/s]
 16%|█▌        | 64/400 [00:00<00:01, 315.60it/s]
 24%|██▍       | 96/400 [00:00<00:00, 316.93it/s]
 32%|███▏      | 128/400 [00:00<00:00, 317.46it/s]
 40%|████      | 160/400 [00:00<00:00, 317.98it/s]
 48%|████▊     | 192/400 [00:00<00:00, 318.50it/s]
 56%|█████▌    | 224/400 [00:00<00:00, 317.81it/s]
 64%|██████▍   | 256/400 [00:00<00:00, 317.77it/s]
 72%|███████▏  | 288/400 [00:00<00:00, 317.02it/s]
 80%|████████  | 320/400 [00:01<00:00, 316.89it/s]
 88%|████████▊ | 352/400 [00:01<00:00, 317.07it/s]
 96%|█████████▌| 384/400 [00:01<00:00, 316.24it/s]
100%|██████████| 400/400 [00:01<00:00, 316.94it/s]
    AUC=0.900  중요도(level/diff/diff2)=0.051/0.446/0.503

[kNN] (Instance-based) 학습 중...

  0%|          | 0/400 [00:00<?, ?it/s]
  5%|▌         | 20/400 [00:00<00:01, 199.64it/s]
 10%|█         | 41/400 [00:00<00:01, 205.12it/s]
 16%|█▌        | 62/400 [00:00<00:01, 206.03it/s]
 21%|██        | 83/400 [00:00<00:01, 204.34it/s]
 26%|██▌       | 104/400 [00:00<00:01, 205.52it/s]
 31%|███▏      | 125/400 [00:00<00:01, 206.76it/s]
 36%|███▋      | 146/400 [00:00<00:01, 205.09it/s]
 42%|████▏     | 167/400 [00:00<00:01, 206.47it/s]
 47%|████▋     | 188/400 [00:00<00:01, 207.19it/s]
 52%|█████▏    | 209/400 [00:01<00:00, 206.60it/s]
 57%|█████▊    | 230/400 [00:01<00:00, 206.70it/s]
 63%|██████▎   | 251/400 [00:01<00:00, 207.51it/s]
 68%|██████▊   | 272/400 [00:01<00:00, 207.61it/s]
 73%|███████▎  | 293/400 [00:01<00:00, 206.50it/s]
 78%|███████▊  | 314/400 [00:01<00:00, 207.26it/s]
 84%|████████▍ | 335/400 [00:01<00:00, 207.72it/s]
 89%|████████▉ | 356/400 [00:01<00:00, 207.60it/s]
 94%|█████████▍| 377/400 [00:01<00:00, 208.22it/s]
100%|█████████▉| 398/400 [00:01<00:00, 208.33it/s]
100%|██████████| 400/400 [00:01<00:00, 206.86it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  5%|▌         | 21/400 [00:00<00:01, 204.53it/s]
 10%|█         | 42/400 [00:00<00:01, 205.37it/s]
 16%|█▌        | 63/400 [00:00<00:01, 204.27it/s]
 21%|██        | 84/400 [00:00<00:01, 203.43it/s]
 26%|██▋       | 105/400 [00:00<00:01, 204.74it/s]
 32%|███▏      | 126/400 [00:00<00:01, 203.73it/s]
 37%|███▋      | 147/400 [00:00<00:01, 203.22it/s]
 42%|████▏     | 168/400 [00:00<00:01, 204.87it/s]
 47%|████▋     | 189/400 [00:00<00:01, 204.60it/s]
 52%|█████▎    | 210/400 [00:01<00:00, 203.18it/s]
 58%|█████▊    | 231/400 [00:01<00:00, 205.13it/s]
 63%|██████▎   | 252/400 [00:01<00:00, 204.47it/s]
 68%|██████▊   | 273/400 [00:01<00:00, 203.82it/s]
 74%|███████▎  | 294/400 [00:01<00:00, 204.54it/s]
 79%|███████▉  | 315/400 [00:01<00:00, 204.30it/s]
 84%|████████▍ | 336/400 [00:01<00:00, 204.01it/s]
 89%|████████▉ | 357/400 [00:01<00:00, 202.34it/s]
 94%|█████████▍| 378/400 [00:01<00:00, 203.97it/s]
100%|█████████▉| 399/400 [00:01<00:00, 204.02it/s]
100%|██████████| 400/400 [00:01<00:00, 204.07it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  5%|▌         | 20/400 [00:00<00:01, 194.33it/s]
 10%|█         | 42/400 [00:00<00:01, 204.11it/s]
 16%|█▌        | 63/400 [00:00<00:01, 206.62it/s]
 21%|██        | 84/400 [00:00<00:01, 203.28it/s]
 26%|██▋       | 105/400 [00:00<00:01, 205.49it/s]
 32%|███▏      | 127/400 [00:00<00:01, 207.15it/s]
 37%|███▋      | 149/400 [00:00<00:01, 206.45it/s]
 42%|████▎     | 170/400 [00:00<00:01, 204.82it/s]
 48%|████▊     | 191/400 [00:00<00:01, 206.28it/s]
 53%|█████▎    | 212/400 [00:01<00:00, 206.97it/s]
 58%|█████▊    | 233/400 [00:01<00:00, 203.37it/s]
 64%|██████▎   | 254/400 [00:01<00:00, 205.04it/s]
 69%|██████▉   | 275/400 [00:01<00:00, 206.06it/s]
 74%|███████▍  | 296/400 [00:01<00:00, 203.85it/s]
 79%|███████▉  | 317/400 [00:01<00:00, 204.51it/s]
 84%|████████▍ | 338/400 [00:01<00:00, 203.77it/s]
 90%|████████▉ | 359/400 [00:01<00:00, 205.22it/s]
 95%|█████████▌| 380/400 [00:01<00:00, 203.58it/s]
100%|██████████| 400/400 [00:01<00:00, 204.58it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  5%|▌         | 20/400 [00:00<00:01, 198.27it/s]
 10%|█         | 41/400 [00:00<00:01, 201.20it/s]
 16%|█▌        | 62/400 [00:00<00:01, 204.10it/s]
 21%|██        | 83/400 [00:00<00:01, 204.58it/s]
 26%|██▌       | 104/400 [00:00<00:01, 202.19it/s]
 31%|███▏      | 125/400 [00:00<00:01, 204.47it/s]
 37%|███▋      | 147/400 [00:00<00:01, 206.36it/s]
 42%|████▏     | 168/400 [00:00<00:01, 204.93it/s]
 47%|████▋     | 189/400 [00:00<00:01, 206.40it/s]
 52%|█████▎    | 210/400 [00:01<00:00, 207.41it/s]
 58%|█████▊    | 231/400 [00:01<00:00, 208.08it/s]
 63%|██████▎   | 252/400 [00:01<00:00, 205.94it/s]
 68%|██████▊   | 273/400 [00:01<00:00, 206.23it/s]
 74%|███████▎  | 294/400 [00:01<00:00, 207.17it/s]
 79%|███████▉  | 315/400 [00:01<00:00, 205.26it/s]
 84%|████████▍ | 336/400 [00:01<00:00, 205.81it/s]
 89%|████████▉ | 357/400 [00:01<00:00, 206.42it/s]
 94%|█████████▍| 378/400 [00:01<00:00, 207.28it/s]
100%|█████████▉| 399/400 [00:01<00:00, 205.37it/s]
100%|██████████| 400/400 [00:01<00:00, 205.53it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  5%|▌         | 21/400 [00:00<00:01, 200.73it/s]
 10%|█         | 42/400 [00:00<00:01, 200.29it/s]
 16%|█▌        | 63/400 [00:00<00:01, 203.60it/s]
 21%|██        | 84/400 [00:00<00:01, 203.91it/s]
 26%|██▋       | 105/400 [00:00<00:01, 203.32it/s]
 32%|███▏      | 126/400 [00:00<00:01, 204.67it/s]
 37%|███▋      | 147/400 [00:00<00:01, 202.83it/s]
 42%|████▏     | 168/400 [00:00<00:01, 204.72it/s]
 47%|████▋     | 189/400 [00:00<00:01, 203.80it/s]
 52%|█████▎    | 210/400 [00:01<00:00, 204.85it/s]
 58%|█████▊    | 231/400 [00:01<00:00, 204.12it/s]
 63%|██████▎   | 252/400 [00:01<00:00, 202.45it/s]
 68%|██████▊   | 273/400 [00:01<00:00, 203.86it/s]
 74%|███████▎  | 294/400 [00:01<00:00, 204.08it/s]
 79%|███████▉  | 315/400 [00:01<00:00, 205.49it/s]
 84%|████████▍ | 336/400 [00:01<00:00, 204.32it/s]
 89%|████████▉ | 357/400 [00:01<00:00, 205.67it/s]
 94%|█████████▍| 378/400 [00:01<00:00, 205.04it/s]
100%|█████████▉| 399/400 [00:01<00:00, 204.17it/s]
100%|██████████| 400/400 [00:01<00:00, 204.02it/s]
    AUC=0.918  중요도(level/diff/diff2)=0.129/0.391/0.480

[SVM_RBF] (Kernel) 학습 중...

  0%|          | 0/400 [00:00<?, ?it/s]
  3%|▎         | 12/400 [00:00<00:03, 119.70it/s]
  6%|▋         | 25/400 [00:00<00:03, 121.77it/s]
 10%|▉         | 38/400 [00:00<00:02, 122.21it/s]
 13%|█▎        | 51/400 [00:00<00:02, 122.07it/s]
 16%|█▌        | 64/400 [00:00<00:02, 122.57it/s]
 19%|█▉        | 77/400 [00:00<00:02, 122.58it/s]
 22%|██▎       | 90/400 [00:00<00:02, 122.78it/s]
 26%|██▌       | 103/400 [00:00<00:02, 123.01it/s]
 29%|██▉       | 116/400 [00:00<00:02, 123.22it/s]
 32%|███▏      | 129/400 [00:01<00:02, 122.78it/s]
 36%|███▌      | 142/400 [00:01<00:02, 123.21it/s]
 39%|███▉      | 155/400 [00:01<00:02, 122.12it/s]
 42%|████▏     | 168/400 [00:01<00:01, 123.33it/s]
 45%|████▌     | 181/400 [00:01<00:01, 124.36it/s]
 48%|████▊     | 194/400 [00:01<00:01, 122.19it/s]
 52%|█████▏    | 207/400 [00:01<00:01, 123.47it/s]
 55%|█████▌    | 220/400 [00:01<00:01, 124.43it/s]
 58%|█████▊    | 233/400 [00:01<00:01, 122.99it/s]
 62%|██████▏   | 246/400 [00:02<00:01, 123.23it/s]
 65%|██████▍   | 259/400 [00:02<00:01, 124.19it/s]
 68%|██████▊   | 272/400 [00:02<00:01, 123.86it/s]
 71%|███████▏  | 285/400 [00:02<00:00, 121.22it/s]
 74%|███████▍  | 298/400 [00:02<00:00, 123.06it/s]
 78%|███████▊  | 311/400 [00:02<00:00, 124.23it/s]
 81%|████████  | 324/400 [00:02<00:00, 124.45it/s]
 84%|████████▍ | 337/400 [00:02<00:00, 122.16it/s]
 88%|████████▊ | 350/400 [00:02<00:00, 123.62it/s]
 91%|█████████ | 363/400 [00:02<00:00, 123.50it/s]
 94%|█████████▍| 376/400 [00:03<00:00, 123.89it/s]
 97%|█████████▋| 389/400 [00:03<00:00, 123.36it/s]
100%|██████████| 400/400 [00:03<00:00, 123.23it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  3%|▎         | 12/400 [00:00<00:03, 117.71it/s]
  6%|▋         | 25/400 [00:00<00:03, 121.30it/s]
 10%|▉         | 38/400 [00:00<00:02, 122.45it/s]
 13%|█▎        | 51/400 [00:00<00:02, 121.31it/s]
 16%|█▌        | 64/400 [00:00<00:02, 120.50it/s]
 19%|█▉        | 77/400 [00:00<00:02, 121.74it/s]
 22%|██▎       | 90/400 [00:00<00:02, 121.69it/s]
 26%|██▌       | 103/400 [00:00<00:02, 120.83it/s]
 29%|██▉       | 116/400 [00:00<00:02, 121.57it/s]
 32%|███▏      | 129/400 [00:01<00:02, 121.15it/s]
 36%|███▌      | 142/400 [00:01<00:02, 120.73it/s]
 39%|███▉      | 155/400 [00:01<00:02, 121.51it/s]
 42%|████▏     | 168/400 [00:01<00:01, 120.78it/s]
 45%|████▌     | 181/400 [00:01<00:01, 121.66it/s]
 48%|████▊     | 194/400 [00:01<00:01, 121.10it/s]
 52%|█████▏    | 207/400 [00:01<00:01, 121.82it/s]
 55%|█████▌    | 220/400 [00:01<00:01, 121.66it/s]
 58%|█████▊    | 233/400 [00:01<00:01, 121.06it/s]
 62%|██████▏   | 246/400 [00:02<00:01, 121.48it/s]
 65%|██████▍   | 259/400 [00:02<00:01, 121.03it/s]
 68%|██████▊   | 272/400 [00:02<00:01, 120.38it/s]
 71%|███████▏  | 285/400 [00:02<00:00, 121.30it/s]
 74%|███████▍  | 298/400 [00:02<00:00, 121.88it/s]
 78%|███████▊  | 311/400 [00:02<00:00, 122.23it/s]
 81%|████████  | 324/400 [00:02<00:00, 121.23it/s]
 84%|████████▍ | 337/400 [00:02<00:00, 121.83it/s]
 88%|████████▊ | 350/400 [00:02<00:00, 122.32it/s]
 91%|█████████ | 363/400 [00:02<00:00, 121.34it/s]
 94%|█████████▍| 376/400 [00:03<00:00, 121.95it/s]
 97%|█████████▋| 389/400 [00:03<00:00, 122.24it/s]
100%|██████████| 400/400 [00:03<00:00, 121.44it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  3%|▎         | 12/400 [00:00<00:03, 117.32it/s]
  6%|▋         | 25/400 [00:00<00:03, 122.76it/s]
 10%|▉         | 38/400 [00:00<00:02, 123.76it/s]
 13%|█▎        | 51/400 [00:00<00:02, 121.63it/s]
 16%|█▌        | 64/400 [00:00<00:02, 123.99it/s]
 19%|█▉        | 77/400 [00:00<00:02, 125.39it/s]
 22%|██▎       | 90/400 [00:00<00:02, 123.73it/s]
 26%|██▌       | 103/400 [00:00<00:02, 124.97it/s]
 29%|██▉       | 116/400 [00:00<00:02, 125.86it/s]
 32%|███▏      | 129/400 [00:01<00:02, 125.04it/s]
 36%|███▌      | 142/400 [00:01<00:02, 122.57it/s]
 39%|███▉      | 155/400 [00:01<00:01, 124.27it/s]
 42%|████▏     | 168/400 [00:01<00:01, 125.28it/s]
 45%|████▌     | 181/400 [00:01<00:01, 124.09it/s]
 48%|████▊     | 194/400 [00:01<00:01, 124.78it/s]
 52%|█████▏    | 207/400 [00:01<00:01, 125.68it/s]
 55%|█████▌    | 220/400 [00:01<00:01, 126.56it/s]
 58%|█████▊    | 233/400 [00:01<00:01, 125.82it/s]
 62%|██████▏   | 246/400 [00:01<00:01, 126.36it/s]
 65%|██████▍   | 259/400 [00:02<00:01, 126.59it/s]
 68%|██████▊   | 272/400 [00:02<00:01, 126.41it/s]
 71%|███████▏  | 285/400 [00:02<00:00, 125.23it/s]
 74%|███████▍  | 298/400 [00:02<00:00, 124.98it/s]
 78%|███████▊  | 311/400 [00:02<00:00, 125.39it/s]
 81%|████████  | 324/400 [00:02<00:00, 126.40it/s]
 84%|████████▍ | 337/400 [00:02<00:00, 125.54it/s]
 88%|████████▊ | 350/400 [00:02<00:00, 124.68it/s]
 91%|█████████ | 363/400 [00:02<00:00, 125.47it/s]
 94%|█████████▍| 376/400 [00:03<00:00, 125.69it/s]
 97%|█████████▋| 389/400 [00:03<00:00, 125.82it/s]
100%|██████████| 400/400 [00:03<00:00, 125.01it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  3%|▎         | 13/400 [00:00<00:03, 126.24it/s]
  6%|▋         | 26/400 [00:00<00:02, 127.08it/s]
 10%|▉         | 39/400 [00:00<00:02, 125.43it/s]
 13%|█▎        | 52/400 [00:00<00:02, 126.37it/s]
 16%|█▋        | 65/400 [00:00<00:02, 126.95it/s]
 20%|█▉        | 78/400 [00:00<00:02, 127.24it/s]
 23%|██▎       | 91/400 [00:00<00:02, 125.33it/s]
 26%|██▌       | 104/400 [00:00<00:02, 125.92it/s]
 29%|██▉       | 117/400 [00:00<00:02, 126.64it/s]
 32%|███▎      | 130/400 [00:01<00:02, 126.83it/s]
 36%|███▌      | 143/400 [00:01<00:02, 125.32it/s]
 39%|███▉      | 156/400 [00:01<00:01, 126.15it/s]
 42%|████▏     | 169/400 [00:01<00:01, 126.47it/s]
 46%|████▌     | 182/400 [00:01<00:01, 126.97it/s]
 49%|████▉     | 195/400 [00:01<00:01, 125.41it/s]
 52%|█████▏    | 208/400 [00:01<00:01, 126.17it/s]
 55%|█████▌    | 221/400 [00:01<00:01, 126.20it/s]
 58%|█████▊    | 234/400 [00:01<00:01, 126.15it/s]
 62%|██████▏   | 247/400 [00:01<00:01, 124.78it/s]
 65%|██████▌   | 260/400 [00:02<00:01, 125.03it/s]
 68%|██████▊   | 273/400 [00:02<00:01, 125.38it/s]
 72%|███████▏  | 286/400 [00:02<00:00, 124.59it/s]
 75%|███████▍  | 299/400 [00:02<00:00, 125.04it/s]
 78%|███████▊  | 312/400 [00:02<00:00, 125.39it/s]
 81%|████████▏ | 325/400 [00:02<00:00, 125.42it/s]
 84%|████████▍ | 338/400 [00:02<00:00, 124.56it/s]
 88%|████████▊ | 351/400 [00:02<00:00, 124.22it/s]
 91%|█████████ | 364/400 [00:02<00:00, 124.89it/s]
 94%|█████████▍| 377/400 [00:03<00:00, 124.22it/s]
 98%|█████████▊| 390/400 [00:03<00:00, 123.91it/s]
100%|██████████| 400/400 [00:03<00:00, 124.11it/s]

  0%|          | 0/400 [00:00<?, ?it/s]
  3%|▎         | 13/400 [00:00<00:03, 124.74it/s]
  6%|▋         | 26/400 [00:00<00:03, 114.63it/s]
 10%|▉         | 38/400 [00:00<00:03, 115.11it/s]
 13%|█▎        | 51/400 [00:00<00:02, 118.32it/s]
 16%|█▌        | 64/400 [00:00<00:02, 120.27it/s]
 19%|█▉        | 77/400 [00:00<00:02, 120.95it/s]
 22%|██▎       | 90/400 [00:00<00:02, 121.65it/s]
 26%|██▌       | 103/400 [00:00<00:02, 121.28it/s]
 29%|██▉       | 116/400 [00:00<00:02, 122.89it/s]
 32%|███▏      | 129/400 [00:01<00:02, 123.38it/s]
 36%|███▌      | 142/400 [00:01<00:02, 122.16it/s]
 39%|███▉      | 155/400 [00:01<00:01, 122.67it/s]
 42%|████▏     | 168/400 [00:01<00:01, 122.70it/s]
 45%|████▌     | 181/400 [00:01<00:01, 121.93it/s]
 48%|████▊     | 194/400 [00:01<00:01, 122.04it/s]
 52%|█████▏    | 207/400 [00:01<00:01, 122.37it/s]
 55%|█████▌    | 220/400 [00:01<00:01, 121.49it/s]
 58%|█████▊    | 233/400 [00:01<00:01, 121.70it/s]
 62%|██████▏   | 246/400 [00:02<00:01, 123.11it/s]
 65%|██████▍   | 259/400 [00:02<00:01, 122.59it/s]
 68%|██████▊   | 272/400 [00:02<00:01, 120.73it/s]
 71%|███████▏  | 285/400 [00:02<00:00, 122.63it/s]
 74%|███████▍  | 298/400 [00:02<00:00, 123.99it/s]
 78%|███████▊  | 311/400 [00:02<00:00, 122.85it/s]
 81%|████████  | 324/400 [00:02<00:00, 121.36it/s]
 84%|████████▍ | 337/400 [00:02<00:00, 123.01it/s]
 88%|████████▊ | 350/400 [00:02<00:00, 123.59it/s]
 91%|█████████ | 363/400 [00:02<00:00, 124.13it/s]
 94%|█████████▍| 376/400 [00:03<00:00, 124.60it/s]
 97%|█████████▋| 389/400 [00:03<00:00, 124.53it/s]
100%|██████████| 400/400 [00:03<00:00, 122.37it/s]
    AUC=0.948  중요도(level/diff/diff2)=0.061/0.466/0.472

[RandomForest] (Tree ensemble(bagging)) 학습 중...
    AUC=0.931  중요도(level/diff/diff2)=0.084/0.456/0.460

[ExtraTrees] (Tree ensemble(bagging,extra-random)) 학습 중...
    AUC=0.927  중요도(level/diff/diff2)=0.145/0.389/0.466

[GradBoost_sklearn] (Tree ensemble(boosting)) 학습 중...
    AUC=0.924  중요도(level/diff/diff2)=0.068/0.371/0.561

[XGBoost] (Tree ensemble(boosting)) 학습 중...
    AUC=0.935  중요도(level/diff/diff2)=0.148/0.419/0.433

[LightGBM] (Tree ensemble(boosting)) 학습 중...
    AUC=0.940  중요도(level/diff/diff2)=0.232/0.459/0.309

==========================================================================================
3. [시퀀스 형태] 6개 독립 아키텍처 학습 및 채널별 saliency 추출
==========================================================================================
등록된 시퀀스 모델: 6개 -> ['CNN1D', 'TCN', 'BiLSTM', 'BiGRU', 'TinyTransformer', 'LightMamba']

[CNN1D] (Convolutional) 학습 중...
    AUC=0.983  saliency(level/diff/diff2)=0.303/0.301/0.396

[TCN] (Convolutional(dilated causal)) 학습 중...
    AUC=0.993  saliency(level/diff/diff2)=0.200/0.380/0.419

[BiLSTM] (Recurrent) 학습 중...
    AUC=0.980  saliency(level/diff/diff2)=0.211/0.497/0.292

[BiGRU] (Recurrent) 학습 중...
    AUC=0.989  saliency(level/diff/diff2)=0.285/0.388/0.327

[TinyTransformer] (Attention) 학습 중...
    AUC=0.994  saliency(level/diff/diff2)=0.178/0.438/0.384

[LightMamba] (State-space(SSM)) 학습 중...
    AUC=0.969  saliency(level/diff/diff2)=0.428/0.441/0.130

==========================================================================================
4. 16개 모델 통합 결과 + BOCPD(층4) 대비 성능 리더보드 (MLP_shallow 제외)
==========================================================================================
[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/modelclass_feature_importance.csv  (16개 모델)
                   model                              family      auc  recall_at_youden  fa_rate_at_youden
         TinyTransformer                           Attention 0.993594          0.979310           0.040185
                     TCN       Convolutional(dilated causal) 0.992890          0.931034           0.021252
                   BiGRU                           Recurrent 0.988832          0.958621           0.061437
                   CNN1D                       Convolutional 0.983342          0.951724           0.051777
                  BiLSTM                           Recurrent 0.980232          0.944828           0.053323
              LightMamba                    State-space(SSM) 0.968556          0.875862           0.016229
                 SVM_RBF                              Kernel 0.948447          0.875000           0.084300
                LightGBM             Tree ensemble(boosting) 0.940468          0.833333           0.042875
                 XGBoost             Tree ensemble(boosting) 0.934744          0.875000           0.080779
            RandomForest              Tree ensemble(bagging) 0.931308          0.892857           0.099006
              ExtraTrees Tree ensemble(bagging,extra-random) 0.926562          0.803571           0.059031
       GradBoost_sklearn             Tree ensemble(boosting) 0.924248          0.863095           0.057581
               LogReg_L2                              Linear 0.920379          0.815476           0.076015
               LogReg_L1                      Linear(sparse) 0.920355          0.815476           0.076015
                     kNN                      Instance-based 0.918486          0.797619           0.042461
              GaussianNB                       Probabilistic 0.900168          0.803571           0.101906
BOCPD_tau_Youden(층4 참조값)                Reference(확률모델,비교대상)      NaN          0.684531           0.180943

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/modelclass_performance_leaderboard.csv

==========================================================================================
5. 교차모델 합의도(agreement) 분석
==========================================================================================
Kendall's W(일치도 계수) = 0.6094  (0=완전 무작위, 1=완전 일치)
카이제곱 근사 검정: chi2=19.500, df=2, p=0.000058

level이 최고 중요 피처가 아닌 모델: 16/16개
이항검정(귀무가설: 무작위 순위에서 기대비율 2/3): p=0.001522

모델군(family)별 평균 중요도:
                             family  n_models  mean_level  mean_diff  mean_diff2  diff_plus_diff2_beats_level
                          Attention         1    0.178382   0.437567    0.384051                         True
                      Convolutional         1    0.302857   0.301402    0.395741                         True
      Convolutional(dilated causal)         1    0.200410   0.380395    0.419194                         True
                     Instance-based         1    0.128603   0.390942    0.480455                         True
                             Kernel         1    0.061049   0.466466    0.472484                         True
                             Linear         1    0.059526   0.406677    0.533798                         True
                     Linear(sparse)         1    0.057248   0.411637    0.531115                         True
                      Probabilistic         1    0.050967   0.446326    0.502707                         True
                          Recurrent         2    0.247947   0.442502    0.309550                         True
                   State-space(SSM)         1    0.428209   0.441446    0.130345                         True
             Tree ensemble(bagging)         1    0.083780   0.456382    0.459839                         True
Tree ensemble(bagging,extra-random)         1    0.145145   0.388816    0.466039                         True
            Tree ensemble(boosting)         3    0.149168   0.416493    0.434339                         True

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/modelclass_agreement_summary.csv

==========================================================================================
6. 표 모델 부트스트랩 안정성 (N_BOOT_TABULAR=200)
==========================================================================================
[RandomForest_boot] 200회 중 diff/diff2가 level을 능가한 비율: 100.0%
[LogReg_boot] 200회 중 diff/diff2가 level을 능가한 비율: 100.0%

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/modelclass_bootstrap_stability.csv

==========================================================================================
7. 최종 요약 리포트
==========================================================================================
==========================================================================================
모델군 교차검증(Model-Class Robustness Analysis) — 최종 요약
==========================================================================================

총 16개 독립 모델 (표 형태 10개 + 시퀀스 형태 6개)

Kendall's W = 0.6094 (0=무작위, 1=완전 일치)
  -> 카이제곱 근사 검정 p = 0.000058
level이 최고 중요 피처가 아닌 모델: 16/16개 (이항검정 p = 0.001522)

모델군(family)별 diff+diff2 vs level 우세 여부:
  - Attention: diff/diff2 우세 (level=0.178, diff=0.438, diff2=0.384)
  - Convolutional: diff/diff2 우세 (level=0.303, diff=0.301, diff2=0.396)
  - Convolutional(dilated causal): diff/diff2 우세 (level=0.200, diff=0.380, diff2=0.419)
  - Instance-based: diff/diff2 우세 (level=0.129, diff=0.391, diff2=0.480)
  - Kernel: diff/diff2 우세 (level=0.061, diff=0.466, diff2=0.472)
  - Linear: diff/diff2 우세 (level=0.060, diff=0.407, diff2=0.534)
  - Linear(sparse): diff/diff2 우세 (level=0.057, diff=0.412, diff2=0.531)
  - Probabilistic: diff/diff2 우세 (level=0.051, diff=0.446, diff2=0.503)
  - Recurrent: diff/diff2 우세 (level=0.248, diff=0.443, diff2=0.310)
  - State-space(SSM): diff/diff2 우세 (level=0.428, diff=0.441, diff2=0.130)
  - Tree ensemble(bagging): diff/diff2 우세 (level=0.084, diff=0.456, diff2=0.460)
  - Tree ensemble(bagging,extra-random): diff/diff2 우세 (level=0.145, diff=0.389, diff2=0.466)
  - Tree ensemble(boosting): diff/diff2 우세 (level=0.149, diff=0.416, diff2=0.434)

[해석 가이드]
1) Kendall's W가 높고(예: >0.6) p값이 유의하면, 16개(MLP_shallow는 학습
   실패로 제외)의 서로 다른 귀납편향을
   가진 모델들이 우연이라고 보기 어려운 수준으로 같은 순위(diff/diff2 > level)에
   합의하고 있다는 뜻이다. 이는 층3의 '준실험적 pre/post 비교' 결론을 완전히
   독립적인 방법론으로 재현한 것이다.
2) 모델군별 표에서 예외(level 우세)가 있는 모델군이 있다면, 그 모델군의 귀납편향
   (예: 거리기반 모델은 스케일에 민감, 커널모델은 국소성에 민감 등)과 연결지어
   설명해야 한다 — 무시하지 말고 논문에 그 원인 가설을 명시할 것.
3) 표 형태(사람이 설계한 요약통계량)와 시퀀스 형태(원시 3채널 시계열, saliency)가
   같은 결론에 도달했다면, 이는 '요약통계량 설계가 결론을 편향시켰다'는 우려를
   배제하는 가장 강력한 근거다 — 두 표현 방식의 family 열을 나란히 비교해 확인할 것.
4) 부트스트랩 안정성(frac_diff_or_diff2_beats_level)이 두 대표 모델(RF, LogReg)
   모두에서 90% 이상이면, 이 결론이 특정 표본 구성에 우연히 의존한 것이 아니라는
   근거로 추가할 수 있다.
5) 성능 리더보드(AUC)에서 BOCPD_tau_Youden(층4 참조값)과 비교했을 때, 만약 일부
   독립 모델이 BOCPD보다 더 높은 AUC를 보인다면, 이는 'BOCPD가 최적'이라는 주장을
   약화시키는 것이 아니라 '실시간 온라인 확률적 변화점 탐지(BOCPD)'와 '사후적
   배치 분류(표/시퀀스 모델)'는 애초에 다른 문제를 푸는 것이라는 점을 논문에
   명시해야 한다 — 이 비교의 목적은 '누가 이기는가'가 아니라 '같은 신호를
   본다고 하는가'이다.

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/modelclass_final_report.txt

==========================================================================================
완료
==========================================================================================
모든 산출물은 OUT_DIR(causal_pipeline_outputs) 아래에 저장되었습니다.
논문 서술 시 modelclass_agreement_summary.csv(Kendall's W, 이항검정)와
modelclass_family_summary.csv(모델군별 breakdown)를 주 결과표로,
modelclass_bootstrap_stability.csv를 견고성 근거로 사용하는 것을 권장합니다.


"""
==============================================================================
Table 1. Cross-model-class feature-attribution agreement
         (N = 16 independent models; MLP_shallow excluded — AUC = 0.403,
         below chance, indicative of a failed fit rather than a genuine
         level-favoring inductive bias)
==============================================================================

Model                | Family                          |   AUC | level | diff | diff2
----------------------+----------------------------------+-------+-------+------+-------
TinyTransformer       | Attention                        | 0.994 | 0.178 | .438 | .384
TCN                   | Convolutional (dilated causal)   | 0.993 | 0.200 | .380 | .419
BiGRU                 | Recurrent                        | 0.989 | 0.285 | .388 | .327
CNN1D                 | Convolutional                    | 0.983 | 0.303 | .301 | .396
BiLSTM                | Recurrent                        | 0.980 | 0.211 | .497 | .292
LightMamba            | State-space (SSM)                | 0.969 | 0.428 | .441 | .130
SVM_RBF               | Kernel                           | 0.948 | 0.061 | .466 | .472
LightGBM              | Tree ensemble (boosting)         | 0.940 | 0.232 | .459 | .309
XGBoost               | Tree ensemble (boosting)         | 0.935 | 0.148 | .419 | .433
RandomForest          | Tree ensemble (bagging)          | 0.931 | 0.084 | .456 | .460
ExtraTrees            | Tree ensemble (bagging, extra)   | 0.927 | 0.145 | .389 | .466
GradBoost_sklearn     | Tree ensemble (boosting)         | 0.924 | 0.068 | .371 | .561
LogReg_L2             | Linear                           | 0.920 | 0.060 | .407 | .534
LogReg_L1             | Linear (sparse)                  | 0.920 | 0.057 | .412 | .531
kNN                   | Instance-based                   | 0.918 | 0.129 | .391 | .480
GaussianNB            | Probabilistic                    | 0.900 | 0.051 | .446 | .503
----------------------+----------------------------------+-------+-------+------+-------
BOCPD_tau_Youden(ref) | Reference (probabilistic, online) |   n/a | recall=0.685, FA=0.181 (Stage-4 baseline; not
                       |                                    |       | commensurable with batch-classifier AUC)

Columns "level / diff / diff2" report each model's normalized feature-
importance share (SHAP or permutation for tabular models; channel-wise
Integrated-Gradients saliency for sequence models), summing to 1 per row.

------------------------------------------------------------------------------
Table 2. Family-level aggregation (13 inductive-bias families) and
         cross-model agreement statistics
------------------------------------------------------------------------------

Family                              | n | level | diff | diff2 | diff+diff2 > level
-------------------------------------+---+-------+------+-------+--------------------
Linear                               | 1 | 0.060 | .407 | .534  | Yes
Linear (sparse)                      | 1 | 0.057 | .412 | .531  | Yes
Probabilistic                        | 1 | 0.051 | .446 | .503  | Yes
Instance-based                       | 1 | 0.129 | .391 | .480  | Yes
Kernel                               | 1 | 0.061 | .466 | .472  | Yes
Tree ensemble (bagging)              | 1 | 0.084 | .456 | .460  | Yes
Tree ensemble (bagging, extra)       | 1 | 0.145 | .389 | .466  | Yes
Tree ensemble (boosting)             | 3 | 0.149 | .416 | .434  | Yes
Convolutional                        | 1 | 0.303 | .301 | .396  | Yes
Convolutional (dilated causal)       | 1 | 0.200 | .380 | .419  | Yes
Recurrent                            | 2 | 0.248 | .443 | .310  | Yes
Attention                            | 1 | 0.178 | .438 | .384  | Yes
State-space (SSM)                    | 1 | 0.428 | .441 | .130  | Yes (narrow margin)
-------------------------------------+---+-------+------+-------+--------------------
13/13 families favor diff+diff2 over level; margin narrowest for the
State-space (SSM) family (0.441+0.130=0.571 vs 0.428), reflecting the
lightweight (non-CUDA-kernel) Mamba-style substitute block used here.

------------------------------------------------------------------------------
Table 3. Cross-model agreement and stability statistics (N = 16)
------------------------------------------------------------------------------
Statistic                                          | Value
-----------------------------------------------------+------------------------
Kendall's coefficient of concordance (W)             | 0.609
Chi-square approximation (df = 2)                    | chi2 = 19.50, p < .001
Models ranking level as most important feature       | 0 / 16
Binomial test (H0: chance rate = 1/3 favor diff/diff2)| p = .0015
Bootstrap stability, RandomForest (200 resamples)     | 100% favor diff/diff2
Bootstrap stability, LogisticRegression (200 resamples)| 100% favor diff/diff2
==============================================================================
"""




분석 결과 요약
핵심 결론: 16개의 서로 다른 귀납편향을 가진 독립 모델(선형, 확률적, 사례기반, 커널, 트리 앙상블 3종, CNN, TCN, 양방향 RNN 2종, Transformer, SSM)이 예외 없이 전원(16/16) diff/diff²(1·2차 차분의 분산 변화)를 level(신호 절대값)보다 이상 판별에 더 중요한 피처로 꼽았습니다.
Kendall's W = 0.609 (p < .001) — 이 정도 일치도는 "상당한 합의(substantial agreement)" 수준으로 분류되며, 무작위로 나올 확률은 사실상 0에 가깝습니다.
이항검정 p = 0.0015 — 16개 모델 전원이 level을 최고 피처로 뽑지 않은 것이 우연일 가능성을 강하게 배제합니다.
부트스트랩(200회 재표본): RandomForest·LogReg 모두 **100%**에서 diff/diff² 우세 — 특정 표본 구성에 의존한 결과가 아님을 확인.
**표 형태(사람이 설계한 요약통계) vs 시퀀스 형태(원시 시계열 + saliency)**가 완전히 독립적인 데이터 표현임에도 13개 모델군 전부 같은 결론에 도달 — "요약통계 설계가 결론을 편향시켰다"는 우려를 배제하는 가장 강력한 근거입니다.
유일하게 주목할 지점은 State-space(SSM/Mamba) 계열로, level 비중(0.428)이 다른 계열보다 두드러지게 높습니다(그래도 diff+diff² 합이 근소하게 우세).




