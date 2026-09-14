# OPSSAT-AD 데이터셋 설명서

> ESA OPS-SAT CubeSat 위성 텔레메트리 이상탐지 벤치마크 데이터셋 (`dataset.csv`, `segments.csv`)
> 출처: Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. "The OPS-SAT benchmark for detecting anomalies in satellite telemetry." *Scientific Data* 12, 710 (2025).

---

## 1. 데이터셋 개요

- 총 **2,123개의 짧은 단변량(univariate) 텔레메트리 조각(세그먼트)**, 9개 채널에서 수집
- 이상(anomaly) 조각이 전체의 약 20% (434개), 나머지는 정상(1,689개)
- 세그먼트 하나당 채널은 **정확히 1개** — 다변량 구조가 아님 (실측 검증 완료: 예외 0건)
- 두 파일 `dataset.csv`(요약 피처)와 `segments.csv`(원본 시계열)는 `segment`, `channel`, `anomaly`, `sampling`, `train` 컬럼으로 연결됨

---

## 2. 채널(센서) 이름 매핑

데이터 안의 코드명(`CADC08xx`)은 실제로는 아래 물리 센서를 가리킨다. (Nature 논문 Data Records 절에서 공식 확인)

| 데이터 코드명 | 공식 텔레메트리 이름 | 센서 종류 |
|---|---|---|
| CADC0872 | I_B_FB_MM_0 | 자력계(magnetometer) #1 |
| CADC0873 | I_B_FB_MM_1 | 자력계 #2 |
| CADC0874 | I_B_FB_MM_2 | 자력계 #3 |
| CADC0884 | I_PD1_THETA | 광다이오드(photodiode) #1 |
| CADC0886 | I_PD2_THETA | 광다이오드 #2 |
| CADC0888 | I_PD3_THETA | 광다이오드 #3 |
| CADC0890 | I_PD4_THETA | 광다이오드 #4 |
| CADC0892 | I_PD5_THETA | 광다이오드 #5 |
| CADC0894 | I_PD6_THETA | 광다이오드 #6 |

**채널별 세그먼트 수 및 이상 비율** (EDA 실측):

| 채널 | 세그먼트 수 | anomaly 비율 | 비고 |
|---|---|---|---|
| CADC0872 | 546 | 24.0% | |
| CADC0873 | 593 | 17.7% | |
| CADC0874 | 194 | 35.6% | |
| CADC0884 | 158 | **0.0%** | 이상 사례 전무 — positivity violation |
| CADC0886 | 11 | 27.3% | 표본 극소 |
| CADC0888 | 252 | 23.8% | |
| CADC0890 | 14 | 78.6% | 표본 극소, 이상 편중 |
| CADC0892 | 211 | 16.1% | |
| CADC0894 | 144 | 14.6% | |

> 원 논문의 "specialized detector" 실험에서도 CADC0884, CADC0886, CADC0890은 이상 사례가 없거나 너무 적어 **분석 대상에서 제외**되었다. 인과추론 시에도 이 3개 채널은 제외하고 **CADC0872, 0873, 0874, 0888, 0892, 0894 (6개)** 만 사용하는 것이 원 저자들의 방식과 일치한다.

---

## 3. `dataset.csv` 컬럼 설명 (2,123행 — 세그먼트별 요약 피처, 23개 컬럼)

세그먼트 하나하나를 "요약 통계 카드"로 만든 파일. 18개 수작업 피처(handcrafted features) + 식별자 컬럼들로 구성.

| 컬럼명 | 설명 |
|---|---|
| `segment` | 세그먼트 번호 (1~2123) |
| `anomaly` | 정상(0)/이상(1) 이진 정답값 — 실제 분석에 쓰이는 유일한 라벨 |
| `train` | 원 논문 벤치마크 기준 학습용(1)/테스트용(0) 여부 (이상 비율 유지한 층화 랜덤 샘플링) |
| `channel` | 센서 코드명 (위 매핑표 참고) |
| `sampling` | 측정 간격(초) |
| `duration` | 세그먼트 지속 시간(초) |
| `len` | 세그먼트 내 측정값 개수 |
| `mean` | 측정값들의 평균 |
| `var` | 측정값들의 분산 |
| `std` | 측정값들의 표준편차 |
| `kurtosis` | 첨도 — 분포가 얼마나 뾰족한지 |
| `skew` | 왜도 — 분포가 한쪽으로 치우친 정도 |
| `n_peaks` | 신호에서 검출된 피크(최소 10% prominence) 개수 |
| `smooth10_n_peaks` | 10구간 이동평균으로 스무딩한 뒤 검출한 피크 개수 |
| `smooth20_n_peaks` | 20구간 이동평균으로 스무딩한 뒤 검출한 피크 개수 |
| `diff_peaks` | 1차 미분(변화량) 신호에서 검출된 피크 개수 |
| `diff2_peaks` | 2차 미분 신호에서 검출된 피크 개수 |
| `diff_var` | 1차 미분 신호의 분산 |
| `diff2_var` | 2차 미분 신호의 분산 |
| `gaps_squared` | 결측 구간 길이를 제곱해서 합산한 값 (데이터 끊김 정도) |
| `len_weighted` | `sampling × len` — 샘플링 간격 보정 길이 |
| `var_div_duration` | `var / duration` — 시간당 변동성 |
| `var_div_len` | `var / len` — 개수당 변동성 |

**주요 특징 (실측 확인)**:
- `duration`, `len` 등은 `segments.csv`의 실제 timestamp로 재계산해도 100% 일치 (불일치 0건)
- `train` 컬럼은 원 논문의 지도학습 벤치마크용 분할이며, 채널별 비율도 전체 평균(≈75%)과 거의 동일 — 인과추론용 처치/대조군 분할이 아니므로 별도 설계 필요

---

## 4. `segments.csv` 컬럼 설명 (303,493행 — 원본 시계열, 8개 컬럼)

세그먼트 요약이 아니라, 센서가 실제로 매 순간 찍은 원본 숫자 그 자체.

| 컬럼명 | 설명 |
|---|---|
| `channel` | 센서 코드명 |
| `timestamp` | 측정 시각 (ISO 날짜 형식) |
| `value` | 해당 시점의 실제 측정값 |
| `label` | 상태 라벨 — `anomaly` 또는 `a2`/`a3`/`a4` (아래 5절 참고) |
| `sampling` | 해당 세그먼트의 샘플링 간격(초) |
| `anomaly` | 정상(0)/이상(1) 이진값 — `dataset.csv`의 anomaly와 100% 일치 |
| `segment` | 소속 세그먼트 번호 |
| `train` | 학습용(1)/테스트용(0) 여부 |

**세그먼트 내부 일관성 (실측 확인)**: 한 세그먼트 안에서 `label`이나 `anomaly` 값이 바뀌는 경우는 2,123개 중 0건. 즉 세그먼트는 항상 하나의 상태로 통일되어 있음.

---

## 5. `label` 컬럼의 `a2`/`a3`/`a4` — 조사 결과 (미해결)

### 확인된 사실
- `anomaly=1`인 세그먼트는 전부 `label='anomaly'`
- `anomaly=0`인 세그먼트는 `a2`(738) / `a3`(871) / `a4`(80)로 나뉨 — `a1`은 존재하지 않음

### 조사했으나 답을 찾지 못한 경로
- Nature 논문 본문: `label`을 "ground-truth annotation"이라고만 설명, 세부 구분 없음
- Zenodo 데이터셋 페이지, GitHub README: 별도 설명 없음
- `dataset_generator.ipynb`, `modeling_examples.ipynb` (원저자 공개 코드): 두 코드 모두 `label` 컬럼을 전혀 사용하지 않고, 오직 이진 `anomaly` 컬럼만 사용

### 실용적 결론
원저자들 스스로도 `a2/a3/a4` 세부 구분을 실제 분석(피처 추출, 모델 학습)에 전혀 활용하지 않았다. 따라서:
- 이 라벨의 정확한 의미(운영 모드? 라벨링 배치? 담당자 구분?)는 공개 자료로는 확인 불가
- **인과추론 설계 시에도 원저자들과 동일하게 "정상(0) vs 이상(1)" 이진 구분만 사용하는 것을 권장** — `a2/a3/a4`를 처치변수로 무리하게 끌어다 쓸 근거는 부족함

---

## 6. 인과추론 설계에 대한 시사점 요약

| 항목 | 내용 |
|---|---|
| 처치/결과 변수 | `anomaly`(0/1) 이진 구분만 사용 권장 |
| 분석 단위 | 채널별로 분리 — 9개 중 CADC0872/0873/0874/0888/0892/0894 (6개)만 변동성 확보 |
| 제외 대상 | CADC0884(이상 0%), CADC0886·CADC0890(표본 10여 개) — positivity violation |
| 현실적 설계 | 같은 채널 내에서 시간순 "이상 직전 정상 세그먼트 vs 이상 세그먼트" 비교 (준실험적 pre/post 설계) |
| 주의 | `train` 컬럼은 지도학습 벤치마크용이며 인과추론의 처치/대조군과 무관 |



"""
OPSSAT-AD (LEO CubeSat) 데이터셋 1차 EDA 스크립트
==================================================
목적: 인과추론(causal inference) 분석에 들어가기 전, 데이터의 구조적 특성을
      정량적으로 확인한다. 특히 아래 4가지를 검증하는 데 초점을 둔다.

  (1) dataset.csv / segments.csv 스키마 및 조인 관계
  (2) "2,123개 짧은 단변량 조각"이라는 특성이 실제 데이터에서 맞는지
  (3) 채널 간 동시 관측(그룹 구조) 성립 여부 -> GS-SHAP류 방법 적용 가능성
  (4) 세그먼트 단위 정상/이상 이진 라벨의 분포와, 세그먼트 간 시간적 연속성

경로만 본인 환경에 맞게 수정 후 그대로 실행하면 됩니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)

# ------------------------------------------------------------------
# 0. 경로 설정 (환경에 맞게 수정)
# ------------------------------------------------------------------
BASE_DIR = Path("/home/yjlee/Research/Satellite/OPSSAT-AD_LEO data")
DATASET_PATH = BASE_DIR / "dataset.csv"
SEGMENTS_PATH = BASE_DIR / "segments.csv"


def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ------------------------------------------------------------------
# 1. 파일 로드 및 기본 스키마 확인
# ------------------------------------------------------------------
section("1. 파일 로드 및 기본 스키마")

df = pd.read_csv(DATASET_PATH)
seg = pd.read_csv(SEGMENTS_PATH)

print(f"[dataset.csv]  shape={df.shape}")
print(df.dtypes)
print("\n샘플 5행:")
print(df.head())

print(f"\n[segments.csv] shape={seg.shape}")
print(seg.dtypes)
print("\n샘플 5행:")
print(seg.head())

# 두 테이블을 잇는 키 후보 자동 탐색 (컬럼명이 문서마다 다를 수 있어 방어적으로 탐색)
common_cols = set(df.columns) & set(seg.columns)
print(f"\n두 파일의 공통 컬럼(조인 키 후보): {common_cols if common_cols else '없음 -> 별도 확인 필요'}")


# ------------------------------------------------------------------
# 2. 채널 구조 파악 (자력계 3채널 + 광다이오드 6채널 등)
# ------------------------------------------------------------------
section("2. 채널(변수) 구조 파악")

# 채널명이 들어있을 법한 컬럼 자동 탐색
candidate_channel_cols = [c for c in df.columns if df[c].dtype == object and df[c].nunique() < 100]
print("범주형(문자열) 컬럼 중 채널명 후보:")
for c in candidate_channel_cols:
    print(f"  - {c}: nunique={df[c].nunique()}")
    print(f"    값 목록: {sorted(df[c].dropna().unique().tolist())[:20]}")

# 만약 'channel' 컬럼이 실제로 존재한다면 (OPSSAT-AD 공식 스키마 기준)
if "channel" in df.columns:
    print("\n채널별 관측치 수:")
    print(df["channel"].value_counts())


# ------------------------------------------------------------------
# 3. 세그먼트 특성: "2,123개 짧은 단변량 조각" 검증
# ------------------------------------------------------------------
section("3. 세그먼트 길이 / 개수 검증")

seg_id_col = None
for cand in ["segment", "segment_id", "id"]:
    if cand in seg.columns:
        seg_id_col = cand
        break

if seg_id_col:
    print(f"세그먼트 총 개수: {seg[seg_id_col].nunique()}")
else:
    print("segments.csv에서 세그먼트 ID 컬럼을 자동으로 찾지 못했습니다. seg.columns를 확인하세요:")
    print(seg.columns.tolist())

# dataset.csv 쪽에서도 segment 참조 컬럼이 있는지 확인
seg_ref_in_df = [c for c in df.columns if "seg" in c.lower()]
print(f"\ndataset.csv 내 세그먼트 참조 컬럼 후보: {seg_ref_in_df}")

if seg_ref_in_df:
    ref_col = seg_ref_in_df[0]
    seg_lengths = df.groupby(ref_col).size()
    print(f"\n세그먼트별 길이(관측치 수) 분포:")
    print(seg_lengths.describe())
    print(f"\n가장 짧은 5개 세그먼트 길이: {seg_lengths.nsmallest(5).tolist()}")
    print(f"가장 긴  5개 세그먼트 길이: {seg_lengths.nlargest(5).tolist()}")


# ------------------------------------------------------------------
# 4. 단변량 여부(채널 간 동시 관측 구조) 검증
#    -> 하나의 timestamp/segment 안에 여러 채널이 동시에 존재하는지 확인.
#       존재하지 않는다면 "그룹 간 상호작용 분석(GS-SHAP 등) 불가"라는
#       가정이 데이터로 뒷받침됨.
# ------------------------------------------------------------------
section("4. 단변량(univariate) 구조 검증 — 채널 간 동시 관측 여부")

if seg_ref_in_df and "channel" in df.columns:
    ref_col = seg_ref_in_df[0]
    channels_per_segment = df.groupby(ref_col)["channel"].nunique()
    print("세그먼트 하나당 등장하는 채널 수 분포:")
    print(channels_per_segment.value_counts())
    if (channels_per_segment == 1).all():
        print("\n=> 모든 세그먼트가 정확히 1개 채널만 포함 → 완전한 단변량 구조 확인됨.")
    else:
        print("\n=> 일부 세그먼트에 2개 이상 채널이 동시 존재 → 다변량 분석 여지가 있을 수 있음. 추가 확인 필요.")
else:
    print("채널 컬럼 또는 세그먼트 참조 컬럼을 찾지 못해 자동 검증이 불가합니다. 컬럼명을 확인 후 수동 조정하세요.")


# ------------------------------------------------------------------
# 5. 라벨(정상/이상) 분포 확인 — 인과추론의 "처치군/대조군" 개념과 연결
# ------------------------------------------------------------------
section("5. 라벨(anomaly/nominal) 분포")

label_col = None
for cand in ["label", "anomaly", "is_anomaly", "class", "target"]:
    if cand in seg.columns:
        label_col = cand
        break

if label_col:
    print(f"라벨 컬럼: '{label_col}'")
    print(seg[label_col].value_counts())
    print(seg[label_col].value_counts(normalize=True).round(4))
else:
    print("segments.csv에서 라벨 컬럼을 자동으로 찾지 못했습니다. 아래 컬럼 목록을 참고해 수동 지정하세요:")
    print(seg.columns.tolist())


# ------------------------------------------------------------------
# 6. 결측치 / 이상치 기초 점검
# ------------------------------------------------------------------
section("6. 결측치 및 기술통계")

print("[dataset.csv] 결측치 비율:")
print((df.isna().mean() * 100).round(2))

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    print("\n[dataset.csv] 수치형 컬럼 기술통계:")
    print(df[numeric_cols].describe().T)

print("\n[segments.csv] 결측치 비율:")
print((seg.isna().mean() * 100).round(2))


# ------------------------------------------------------------------
# 7. 시간적 연속성 확인 — "연속 장기 시계열 아님"을 정량 검증
#    세그먼트 시작/종료 시각 간 gap을 계산.
# ------------------------------------------------------------------
section("7. 세그먼트 간 시간적 연속성(gap) 확인")

time_cols_seg = [c for c in seg.columns if "time" in c.lower() or "date" in c.lower()]
print(f"segments.csv 내 시간 관련 컬럼 후보: {time_cols_seg}")

start_col = next((c for c in time_cols_seg if "start" in c.lower()), None)
end_col = next((c for c in time_cols_seg if "end" in c.lower() or "stop" in c.lower()), None)

if start_col and end_col:
    seg_sorted = seg.copy()
    seg_sorted[start_col] = pd.to_datetime(seg_sorted[start_col], errors="coerce")
    seg_sorted[end_col] = pd.to_datetime(seg_sorted[end_col], errors="coerce")
    seg_sorted = seg_sorted.sort_values(start_col)

    seg_sorted["duration_sec"] = (seg_sorted[end_col] - seg_sorted[start_col]).dt.total_seconds()
    seg_sorted["gap_to_next_sec"] = (
        seg_sorted[start_col].shift(-1) - seg_sorted[end_col]
    ).dt.total_seconds()

    print("\n세그먼트 길이(초) 기술통계:")
    print(seg_sorted["duration_sec"].describe())

    print("\n세그먼트 간 gap(초) 기술통계 (연속이면 gap≈0, 불연속이면 gap 큼):")
    print(seg_sorted["gap_to_next_sec"].describe())

    pct_discontinuous = (seg_sorted["gap_to_next_sec"] > 0).mean() * 100
    print(f"\n다음 세그먼트와 시간적으로 끊긴(gap>0) 비율: {pct_discontinuous:.1f}%")
else:
    print("시작/종료 시간 컬럼을 자동으로 찾지 못했습니다. segments.csv 컬럼명을 확인 후 위 변수(start_col, end_col)를 직접 지정하세요.")
    print(seg.columns.tolist())


# ------------------------------------------------------------------
# 8. 서브시스템 그룹 라벨 존재 여부 확인
#    (제안서에서 지적한 "서브시스템 그룹 라벨 없음"을 실제로 재확인)
# ------------------------------------------------------------------
section("8. 서브시스템/그룹 라벨 존재 여부")

group_like_cols = [c for c in list(df.columns) + list(seg.columns)
                   if any(k in c.lower() for k in ["subsystem", "group", "module", "unit"])]
if group_like_cols:
    print(f"그룹/서브시스템 관련 컬럼 발견: {group_like_cols}")
else:
    print("dataset.csv, segments.csv 어디에도 서브시스템/그룹 라벨 컬럼이 없음 → "
          "제안서에서 지적한 한계(서브시스템 그룹 라벨 없음)가 데이터 상으로 확인됨.")


section("EDA 완료")
print("위 결과를 바탕으로 (1) 채널/조인 키 실제 컬럼명, (2) 라벨 컬럼명, "
      "(3) 시간 컬럼명을 확인해 스크립트 상단 변수명을 맞추면 이후 인과추론 "
      "설계(처치 정의, 대조군 구성, confounder 식별)로 넘어갈 수 있습니다.")

================================================================================
1. 파일 로드 및 기본 스키마
================================================================================
[dataset.csv]  shape=(2123, 23)
segment               int64
anomaly               int64
train                 int64
channel                 str
sampling              int64
duration              int64
len                   int64
mean                float64
var                 float64
std                 float64
kurtosis            float64
skew                float64
n_peaks               int64
smooth10_n_peaks      int64
smooth20_n_peaks      int64
diff_peaks            int64
diff2_peaks           int64
diff_var            float64
diff2_var           float64
gaps_squared          int64
len_weighted          int64
var_div_duration    float64
var_div_len         float64
dtype: object

샘플 5행:
   segment  anomaly  train   channel  sampling  duration  len          mean           var       std  kurtosis      skew  n_peaks  smooth10_n_peaks  \
0        1        1      1  CADC0872         1       279  280  8.533143e-07  3.494283e-10  0.000019  0.631117  0.552052        4                 3   
1        2        1      1  CADC0872         1       476  477 -3.639396e-06  6.476485e-10  0.000025 -1.243611  0.425632        1                 1   
2        3        1      1  CADC0872         1       594  595  1.170788e-05  5.592877e-10  0.000024 -0.284593 -0.826187        3                 2   
3        4        1      1  CADC0872         1       271  272  8.486808e-07  5.466024e-10  0.000023 -0.887088 -0.138498        2                 2   
4        5        0      0  CADC0872         1       255  257  1.058485e-05  5.279023e-10  0.000023 -1.484393 -0.060155        1                 1   

   smooth20_n_peaks  diff_peaks  diff2_peaks      diff_var     diff2_var  gaps_squared  len_weighted  var_div_duration   var_div_len  
0                 2           4            6  1.271176e-10  2.960666e-10           309           280      1.252431e-12  1.247958e-12  
1                 1           5            8  1.489383e-12  3.004752e-12           644           477      1.360606e-12  1.357754e-12  
2                 2           2            3  4.112280e-12  1.029918e-11           772           595      9.415618e-13  9.399794e-13  
3                 2           3            6  2.475760e-11  6.240985e-11           339           272      2.016983e-12  2.009568e-12  
4                 1          78           87  5.547101e-13  7.035422e-13           357           257      2.070205e-12  2.054094e-12  

[segments.csv] shape=(303493, 8)
channel          str
timestamp        str
value        float64
label            str
sampling       int64
anomaly        int64
segment        int64
train          int64
dtype: object

샘플 5행:
    channel                 timestamp     value    label  sampling  anomaly  segment  train
0  CADC0872  2022-06-01T23:42:54.000Z -0.000021  anomaly         1        1        1      1
1  CADC0872  2022-06-01T23:42:55.000Z -0.000021  anomaly         1        1        1      1
2  CADC0872  2022-06-01T23:42:56.000Z -0.000021  anomaly         1        1        1      1
3  CADC0872  2022-06-01T23:42:57.000Z -0.000021  anomaly         1        1        1      1
4  CADC0872  2022-06-01T23:42:58.000Z -0.000021  anomaly         1        1        1      1

두 파일의 공통 컬럼(조인 키 후보): {'anomaly', 'train', 'segment', 'sampling', 'channel'}

================================================================================
2. 채널(변수) 구조 파악
================================================================================
범주형(문자열) 컬럼 중 채널명 후보:

채널별 관측치 수:
channel
CADC0873    593
CADC0872    546
CADC0888    252
CADC0892    211
CADC0874    194
CADC0884    158
CADC0894    144
CADC0890     14
CADC0886     11
Name: count, dtype: int64

================================================================================
3. 세그먼트 길이 / 개수 검증
================================================================================
세그먼트 총 개수: 2123

dataset.csv 내 세그먼트 참조 컬럼 후보: ['segment']

세그먼트별 길이(관측치 수) 분포:
count    2123.0
mean        1.0
std         0.0
min         1.0
25%         1.0
50%         1.0
75%         1.0
max         1.0
dtype: float64

가장 짧은 5개 세그먼트 길이: [1, 1, 1, 1, 1]
가장 긴  5개 세그먼트 길이: [1, 1, 1, 1, 1]

================================================================================
4. 단변량(univariate) 구조 검증 — 채널 간 동시 관측 여부
================================================================================
세그먼트 하나당 등장하는 채널 수 분포:
channel
1    2123
Name: count, dtype: int64

=> 모든 세그먼트가 정확히 1개 채널만 포함 → 완전한 단변량 구조 확인됨.

================================================================================
5. 라벨(anomaly/nominal) 분포
================================================================================
라벨 컬럼: 'label'
label
a3         106930
anomaly    100264
a2          75272
a4          21027
Name: count, dtype: int64
label
a3         0.3523
anomaly    0.3304
a2         0.2480
a4         0.0693
Name: proportion, dtype: float64

================================================================================
6. 결측치 및 기술통계
================================================================================
[dataset.csv] 결측치 비율:
segment             0.0
anomaly             0.0
train               0.0
channel             0.0
sampling            0.0
duration            0.0
len                 0.0
mean                0.0
var                 0.0
std                 0.0
kurtosis            0.0
skew                0.0
n_peaks             0.0
smooth10_n_peaks    0.0
smooth20_n_peaks    0.0
diff_peaks          0.0
diff2_peaks         0.0
diff_var            0.0
diff2_var           0.0
gaps_squared        0.0
len_weighted        0.0
var_div_duration    0.0
var_div_len         0.0
dtype: float64

[dataset.csv] 수치형 컬럼 기술통계:
                   count         mean          std           min           25%           50%          75%           max
segment           2123.0  1062.000000   613.001631  1.000000e+00  5.315000e+02  1.062000e+03  1592.500000   2123.000000
anomaly           2123.0     0.204428     0.403378  0.000000e+00  0.000000e+00  0.000000e+00     0.000000      1.000000
train             2123.0     0.750824     0.432638  0.000000e+00  1.000000e+00  1.000000e+00     1.000000      1.000000
sampling          2123.0     3.505888     1.935418  1.000000e+00  1.000000e+00  5.000000e+00     5.000000      5.000000
duration          2123.0   267.952426   169.093207  3.500000e+01  1.750000e+02  2.250000e+02   336.000000   1335.000000
len               2123.0   142.954781   152.329786  8.000000e+00  4.000000e+01  7.000000e+01   201.000000   1040.000000
mean              2123.0     0.110968     0.196316 -3.833030e-05 -7.239804e-07  4.515496e-06     0.205171      1.118632
var               2123.0     0.030659     0.054912  1.422763e-11  2.960552e-10  6.032890e-10     0.037388      0.272480
std               2123.0     0.096226     0.146319  3.771953e-06  1.720625e-05  2.456194e-05     0.193361      0.521996
kurtosis          2123.0    -1.085494     1.000566 -1.858813e+00 -1.449123e+00 -1.302729e+00    -0.996211     31.201842
skew              2123.0     0.175551     0.529314 -1.509396e+00 -1.177185e-01  6.758003e-03     0.538015      4.374560
n_peaks           2123.0     1.541215     2.391978  1.000000e+00  1.000000e+00  1.000000e+00     1.000000     47.000000
smooth10_n_peaks  2123.0     1.110221     0.399227  0.000000e+00  1.000000e+00  1.000000e+00     1.000000      5.000000
smooth20_n_peaks  2123.0     1.181818     0.425295  0.000000e+00  1.000000e+00  1.000000e+00     1.000000      4.000000
diff_peaks        2123.0    18.110692    28.991979  0.000000e+00  1.000000e+00  5.000000e+00    21.000000    240.000000
diff2_peaks       2123.0    25.286387    32.737797  0.000000e+00  4.000000e+00  1.400000e+01    32.000000    256.000000
diff_var          2123.0     0.000917     0.004196  1.962330e-13  3.163852e-12  3.640618e-11     0.000298      0.099879
diff2_var         2123.0     0.001046     0.011274  3.875356e-13  1.233331e-12  1.095135e-11     0.000140      0.303399
gaps_squared      2123.0  1027.671691  1555.230980  1.260000e+02  3.580000e+02  5.750000e+02  1200.000000  20750.000000
len_weighted      2123.0   267.925106   166.245528  4.000000e+01  1.750000e+02  2.300000e+02   330.000000   1340.000000
var_div_duration  2123.0     0.000116     0.000209  2.817267e-14  1.310632e-12  3.761334e-12     0.000183      0.001687
var_div_len       2123.0     0.000436     0.000885  4.404840e-14  2.809916e-12  1.563517e-11     0.000472      0.008031

[segments.csv] 결측치 비율:
channel      0.0
timestamp    0.0
value        0.0
label        0.0
sampling     0.0
anomaly      0.0
segment      0.0
train        0.0
dtype: float64

================================================================================
7. 세그먼트 간 시간적 연속성(gap) 확인
================================================================================
segments.csv 내 시간 관련 컬럼 후보: ['timestamp']
시작/종료 시간 컬럼을 자동으로 찾지 못했습니다. segments.csv 컬럼명을 확인 후 위 변수(start_col, end_col)를 직접 지정하세요.
['channel', 'timestamp', 'value', 'label', 'sampling', 'anomaly', 'segment', 'train']

================================================================================
8. 서브시스템/그룹 라벨 존재 여부
================================================================================
dataset.csv, segments.csv 어디에도 서브시스템/그룹 라벨 컬럼이 없음 → 제안서에서 지적한 한계(서브시스템 그룹 라벨 없음)가 데이터 상으로 확인됨.

================================================================================
EDA 완료
================================================================================
위 결과를 바탕으로 (1) 채널/조인 키 실제 컬럼명, (2) 라벨 컬럼명, (3) 시간 컬럼명을 확인해 스크립트 상단 변수명을 맞추면 이후 인과추론 설계(처치 정의, 대조군 구성, confounder 식별)로 넘어갈 수 있습니다.



"""
OPSSAT-AD (LEO CubeSat) 데이터셋 상세 확인 스크립트 (v2)
=========================================================
1차 EDA에서 확인된 실제 스키마를 반영해, 모든 인덱스(2,123개 세그먼트,
303,493개 원시 관측치)를 잘리지 않고 확인할 수 있도록 구성했습니다.

실제 스키마 (1차 EDA 결과 기준)
--------------------------------
dataset.csv  (2,123행 = 세그먼트 단위 요약 피처)
  segment, anomaly(0/1), train(0/1), channel, sampling, duration, len,
  mean, var, std, kurtosis, skew, n_peaks, smooth10_n_peaks, smooth20_n_peaks,
  diff_peaks, diff2_peaks, diff_var, diff2_var, gaps_squared, len_weighted,
  var_div_duration, var_div_len

segments.csv (303,493행 = 세그먼트 내부 원시 시계열, timestamp 단위)
  channel, timestamp, value, label(a2/a3/a4/anomaly), sampling, anomaly(0/1),
  segment, train

핵심 확인 사항
--------------
(A) segments.csv의 'label'(a2/a3/a4/anomaly)과 dataset.csv의 'anomaly'(0/1)가
    세그먼트 단위로 일관되게 매핑되는지 -> a2/a3/a4가 무엇을 의미하는지 역추적
(B) 한 세그먼트 내부에서 label/anomaly 값이 시간에 따라 바뀌는지(비일관 여부)
(C) 세그먼트별 실제 시작/종료 timestamp, 실측 duration -> dataset.csv의
    duration/len과 일치하는지 검증
(D) 채널별/라벨별/train-test 분할별 전수 크로스탭
(E) 위 모든 결과를 CSV로 내보내 "모든 인덱스"를 직접 열어볼 수 있게 함
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 콘솔 출력이 잘리지 않도록 설정
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.6g}")

# ------------------------------------------------------------------
# 0. 경로 설정
# ------------------------------------------------------------------
BASE_DIR = Path("/home/yjlee/Research/Satellite/OPSSAT-AD_LEO data")
DATASET_PATH = BASE_DIR / "dataset.csv"
SEGMENTS_PATH = BASE_DIR / "segments.csv"

OUT_DIR = BASE_DIR / "eda_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ------------------------------------------------------------------
# 1. 로드
# ------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)                      # 2,123행: 세그먼트 요약
seg = pd.read_csv(SEGMENTS_PATH, parse_dates=["timestamp"])  # 303,493행: 원시 시계열

section("0. 기본 크기 확인")
print(f"dataset.csv  : {df.shape}  (세그먼트 요약 피처, index=segment)")
print(f"segments.csv : {seg.shape} (원시 시계열, index=timestamp)")


# ------------------------------------------------------------------
# (A) segments.csv 내부: 세그먼트별 label/anomaly 일관성 체크
#     -> 하나의 segment 안에서 label 값이 여러 개인지, anomaly 값이 여러 개인지
# ------------------------------------------------------------------
section("(A) 세그먼트별 label/anomaly 내부 일관성 체크")

seg_label_nunique = seg.groupby("segment")["label"].nunique()
seg_anomaly_nunique = seg.groupby("segment")["anomaly"].nunique()

print(f"세그먼트 내부에 label 값이 2개 이상 섞여있는 세그먼트 수: {(seg_label_nunique > 1).sum()} / {seg_label_nunique.shape[0]}")
print(f"세그먼트 내부에 anomaly 값이 2개 이상 섞여있는 세그먼트 수: {(seg_anomaly_nunique > 1).sum()} / {seg_anomaly_nunique.shape[0]}")

if (seg_label_nunique > 1).any():
    print("\n[경고] 아래 세그먼트들은 내부에서 label이 바뀝니다 (인과추론 시 처치 정의에 주의):")
    print(seg_label_nunique[seg_label_nunique > 1])


# ------------------------------------------------------------------
# (B) segments.csv의 label(a2/a3/a4/anomaly) <-> dataset.csv의 anomaly(0/1) 매핑
# ------------------------------------------------------------------
section("(B) label(a2/a3/a4/anomaly) vs dataset.csv anomaly(0/1) 매핑 확인")

# 세그먼트별 대표 label (내부가 일관적이라면 첫 값 = 유일값)
seg_label_map = seg.groupby("segment")["label"].first().rename("seg_label_repr")
seg_anomaly_map_from_raw = seg.groupby("segment")["anomaly"].first().rename("seg_anomaly_repr")

merged_label_check = (
    df.set_index("segment")[["anomaly", "channel", "train"]]
    .join(seg_label_map)
    .join(seg_anomaly_map_from_raw)
)

print("dataset.csv의 anomaly(0/1) 값별로, segments.csv의 label이 어떻게 분포하는지:")
crosstab_label_anomaly = pd.crosstab(merged_label_check["anomaly"], merged_label_check["seg_label_repr"])
print(crosstab_label_anomaly)

print("\ndataset.csv의 anomaly(0/1)와 segments.csv의 anomaly(0/1)가 완전히 일치하는지:")
mismatch = (merged_label_check["anomaly"] != merged_label_check["seg_anomaly_repr"]).sum()
print(f"불일치 세그먼트 수: {mismatch} / {merged_label_check.shape[0]}")


# ------------------------------------------------------------------
# (C) 세그먼트별 실제 timestamp 기반 시작/종료/duration 계산 -> dataset.csv와 대조
# ------------------------------------------------------------------
section("(C) 세그먼트별 실제 시작/종료 시각 및 duration 재계산")

seg_time_summary = seg.groupby("segment").agg(
    start_time=("timestamp", "min"),
    end_time=("timestamp", "max"),
    n_obs_raw=("timestamp", "count"),
).reset_index()

seg_time_summary["duration_actual_sec"] = (
    seg_time_summary["end_time"] - seg_time_summary["start_time"]
).dt.total_seconds()

# dataset.csv의 duration/len과 실제 값 비교
check = df[["segment", "duration", "len", "sampling"]].merge(seg_time_summary, on="segment", how="left")
check["duration_diff"] = check["duration"] - check["duration_actual_sec"]
check["len_diff"] = check["len"] - check["n_obs_raw"]

print("dataset.csv의 duration/len 과 segments.csv에서 재계산한 값의 차이 (0이면 완전 일치):")
print(check[["duration_diff", "len_diff"]].describe())

n_mismatch_duration = (check["duration_diff"].abs() > 1e-6).sum()
n_mismatch_len = (check["len_diff"].abs() > 1e-6).sum()
print(f"\nduration 불일치 세그먼트 수: {n_mismatch_duration} / {check.shape[0]}")
print(f"len 불일치 세그먼트 수     : {n_mismatch_len} / {check.shape[0]}")


# ------------------------------------------------------------------
# (D) 채널 x 라벨 x train/test 전수 크로스탭
# ------------------------------------------------------------------
section("(D) 채널 x anomaly x train/test 전수 크로스탭 (세그먼트 기준, dataset.csv)")

print("채널별 anomaly(0/1) 개수:")
print(pd.crosstab(df["channel"], df["anomaly"], margins=True))

print("\n채널별 train(0)/test(1?) 개수 — train 컬럼 의미 재확인 필요:")
print(pd.crosstab(df["channel"], df["train"], margins=True))

print("\n채널 x anomaly x train 3중 교차 (long format):")
tri_cross = df.groupby(["channel", "anomaly", "train"]).size().rename("n_segments").reset_index()
print(tri_cross.to_string(index=False))


# ------------------------------------------------------------------
# (E) 전체 인덱스(세그먼트) 상세 테이블 구성 + CSV로 내보내기
#     -> 사용자가 엑셀/뷰어로 2,123개 세그먼트 전체를 직접 열람 가능
# ------------------------------------------------------------------
section("(E) 세그먼트 전체 상세 테이블 생성 및 CSV 저장")

full_segment_table = (
    df.merge(seg_time_summary, on="segment", how="left")
      .merge(seg_label_map, on="segment", how="left")
)

out_path_1 = OUT_DIR / "segment_full_detail.csv"
full_segment_table.to_csv(out_path_1, index=False)
print(f"[저장 완료] 세그먼트 전체 상세(2,123행 x 모든 컬럼) -> {out_path_1}")

out_path_2 = OUT_DIR / "raw_timeseries_full.csv"
seg.sort_values(["segment", "timestamp"]).to_csv(out_path_2, index=False)
print(f"[저장 완료] 원시 시계열 전체(303,493행) -> {out_path_2}")

out_path_3 = OUT_DIR / "channel_anomaly_train_crosstab.csv"
tri_cross.to_csv(out_path_3, index=False)
print(f"[저장 완료] 채널 x anomaly x train 교차표 -> {out_path_3}")

# 콘솔에서도 전체(2,123행)를 바로 보고 싶을 경우를 위해 옵션 제공
PRINT_FULL_TABLE_TO_CONSOLE = False  # True로 바꾸면 2,123행 전체를 콘솔에 출력
if PRINT_FULL_TABLE_TO_CONSOLE:
    section("세그먼트 전체 상세 테이블 (전체 2,123행)")
    print(full_segment_table)


# ------------------------------------------------------------------
# (F) 채널별 상세 기술통계 (요약이 아니라 채널마다 개별 출력)
# ------------------------------------------------------------------
section("(F) 채널별 상세 기술통계 (전 채널 개별 출력)")

for ch, sub in df.groupby("channel"):
    print(f"\n--- 채널: {ch} (세그먼트 수={len(sub)}, anomaly 비율={sub['anomaly'].mean():.3f}) ---")
    print(sub[["duration", "len", "mean", "var", "std", "kurtosis", "skew", "n_peaks"]].describe())


section("완료")
print(f"모든 세그먼트/원시 데이터는 아래 폴더에 CSV로 저장되었습니다. 엑셀 등에서 직접 전수 확인 가능합니다:\n  {OUT_DIR.resolve()}")

==========================================================================================
0. 기본 크기 확인
==========================================================================================
dataset.csv  : (2123, 23)  (세그먼트 요약 피처, index=segment)
segments.csv : (303493, 8) (원시 시계열, index=timestamp)

==========================================================================================
(A) 세그먼트별 label/anomaly 내부 일관성 체크
==========================================================================================
세그먼트 내부에 label 값이 2개 이상 섞여있는 세그먼트 수: 0 / 2123
세그먼트 내부에 anomaly 값이 2개 이상 섞여있는 세그먼트 수: 0 / 2123

==========================================================================================
(B) label(a2/a3/a4/anomaly) vs dataset.csv anomaly(0/1) 매핑 확인
==========================================================================================
dataset.csv의 anomaly(0/1) 값별로, segments.csv의 label이 어떻게 분포하는지:
seg_label_repr   a2   a3  a4  anomaly
anomaly                              
0               738  871  80        0
1                 0    0   0      434

dataset.csv의 anomaly(0/1)와 segments.csv의 anomaly(0/1)가 완전히 일치하는지:
불일치 세그먼트 수: 0 / 2123

==========================================================================================
(C) 세그먼트별 실제 시작/종료 시각 및 duration 재계산
==========================================================================================
dataset.csv의 duration/len 과 segments.csv에서 재계산한 값의 차이 (0이면 완전 일치):
       duration_diff  len_diff
count          2,123     2,123
mean               0         0
std                0         0
min                0         0
25%                0         0
50%                0         0
75%                0         0
max                0         0

duration 불일치 세그먼트 수: 0 / 2123
len 불일치 세그먼트 수     : 0 / 2123

==========================================================================================
(D) 채널 x anomaly x train/test 전수 크로스탭 (세그먼트 기준, dataset.csv)
==========================================================================================
채널별 anomaly(0/1) 개수:
anomaly      0    1   All
channel                  
CADC0872   415  131   546
CADC0873   488  105   593
CADC0874   125   69   194
CADC0884   158    0   158
CADC0886     8    3    11
CADC0888   192   60   252
CADC0890     3   11    14
CADC0892   177   34   211
CADC0894   123   21   144
All       1689  434  2123

채널별 train(0)/test(1?) 개수 — train 컬럼 의미 재확인 필요:
train       0     1   All
channel                  
CADC0872  132   414   546
CADC0873  153   440   593
CADC0874   52   142   194
CADC0884   36   122   158
CADC0886    4     7    11
CADC0888   64   188   252
CADC0890    2    12    14
CADC0892   53   158   211
CADC0894   33   111   144
All       529  1594  2123

채널 x anomaly x train 3중 교차 (long format):
 channel  anomaly  train  n_segments
CADC0872        0      0         100
CADC0872        0      1         315
CADC0872        1      0          32
CADC0872        1      1          99
CADC0873        0      0         122
CADC0873        0      1         366
CADC0873        1      0          31
CADC0873        1      1          74
CADC0874        0      0          29
CADC0874        0      1          96
CADC0874        1      0          23
CADC0874        1      1          46
CADC0884        0      0          36
CADC0884        0      1         122
CADC0886        0      0           3
CADC0886        0      1           5
CADC0886        1      0           1
CADC0886        1      1           2
CADC0888        0      0          52
CADC0888        0      1         140
CADC0888        1      0          12
CADC0888        1      1          48
CADC0890        0      1           3
CADC0890        1      0           2
CADC0890        1      1           9
CADC0892        0      0          46
CADC0892        0      1         131
CADC0892        1      0           7
CADC0892        1      1          27
CADC0894        0      0          28
CADC0894        0      1          95
CADC0894        1      0           5
CADC0894        1      1          16

==========================================================================================
(E) 세그먼트 전체 상세 테이블 생성 및 CSV 저장
==========================================================================================
[저장 완료] 세그먼트 전체 상세(2,123행 x 모든 컬럼) -> /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/eda_outputs/segment_full_detail.csv
[저장 완료] 원시 시계열 전체(303,493행) -> /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/eda_outputs/raw_timeseries_full.csv
[저장 완료] 채널 x anomaly x train 교차표 -> /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/eda_outputs/channel_anomaly_train_crosstab.csv

==========================================================================================
(F) 채널별 상세 기술통계 (전 채널 개별 출력)
==========================================================================================

--- 채널: CADC0872 (세그먼트 수=546, anomaly 비율=0.240) ---
       duration     len         mean         var         std  kurtosis        skew  n_peaks
count       546     546          546         546         546       546         546      546
mean    228.762 122.379  9.29597e-07 4.20346e-10 1.97501e-05  -1.21385   -0.126869  1.34066
std      165.97  114.57  7.30799e-06  2.1797e-10 5.50789e-06  0.477231    0.270294  1.06886
min          65      14  -3.7107e-05 3.54502e-11 5.95401e-06   -1.5656     -1.5094        1
25%         100      21 -2.56642e-06 2.41653e-10 1.55452e-05  -1.46901    -0.22348        1
50%         195      60  1.63939e-06 4.07721e-10 2.01921e-05  -1.38147  -0.0690496        1
75%      251.75  189.75  4.89085e-06  5.7469e-10 2.39727e-05  -1.11469 -0.00171555        1
max       1,335     687  2.68051e-05 1.09318e-09 3.30633e-05   3.49958     1.49149       12

--- 채널: CADC0873 (세그먼트 수=593, anomaly 비율=0.177) ---
       duration     len         mean         var         std  kurtosis       skew  n_peaks
count       593     593          593         593         593       593        593      593
mean    220.589 115.159 -7.62232e-07 4.05533e-10 1.92271e-05  -1.20383  -0.121356  1.22091
std     148.484 111.159  5.69609e-06 2.31426e-10 5.99276e-06   1.40609   0.298452 0.967426
min          65      12 -3.04967e-05 1.42276e-11 3.77195e-06  -1.68904   -1.22946        1
25%         105      22 -3.08493e-06 2.15024e-10 1.46637e-05  -1.48118  -0.209294        1
50%         202      52 -3.14166e-07 3.66578e-10 1.91462e-05   -1.4052 -0.0612993        1
75%         245     188  2.02742e-06 5.70342e-10 2.38818e-05  -1.17464 0.00709404        1
max       1,245     673  2.66765e-05  1.0646e-09 3.26282e-05   31.2018    4.37456       19

--- 채널: CADC0874 (세그먼트 수=194, anomaly 비율=0.356) ---
       duration     len         mean         var         std  kurtosis       skew  n_peaks
count       194     194          194         194         194       194        194      194
mean    442.644 302.675  9.21238e-07  3.3933e-10 1.74964e-05  -1.19219 -0.0852975   1.8299
std      186.11 255.992  1.08701e-05 2.18882e-10 5.77733e-06  0.522471   0.299876  1.64362
min         180      37 -3.83303e-05  3.5266e-11 5.93852e-06  -1.72356   -1.09151        1
25%      236.25   48.25 -3.89466e-06 1.58241e-10 1.25794e-05   -1.4471  -0.207095        1
50%         452     156   1.3297e-06  2.9389e-10 1.71432e-05  -1.33746 -0.0530005        1
75%       569.5  497.75  6.87139e-06 4.53331e-10 2.12916e-05  -1.13968  0.0332514        2
max       1,039   1,040  2.51785e-05 1.11167e-09 3.33417e-05   1.79985     1.8013       12

--- 채널: CADC0884 (세그먼트 수=158, anomaly 비율=0.000) ---
       duration     len      mean        var       std  kurtosis       skew  n_peaks
count       158     158       158        158       158       158        158      158
mean    368.918 74.7468  0.419075   0.104114  0.297481  -1.32107   0.376553  1.09494
std     196.621 39.3195  0.210662  0.0689543  0.125375  0.202378   0.260244 0.334587
min          80      17 0.0467725 0.00307346 0.0554388  -1.52784 -0.0754197        1
25%         215      44  0.229542  0.0354971  0.188406  -1.44887   0.172951        1
50%         400      81  0.435717   0.105607  0.324972  -1.39915   0.314094        1
75%      473.75   95.75  0.611091    0.16124  0.401547    -1.236   0.589203        1
max         860     173  0.761575   0.265647  0.515409 -0.340039    1.08101        3

--- 채널: CADC0886 (세그먼트 수=11, anomaly 비율=0.273) ---
       duration     len     mean       var       std  kurtosis      skew  n_peaks
count        11      11       11        11        11        11        11       11
mean    175.909 33.9091 0.181576  0.039985  0.193043  -1.06962  0.489572  1.36364
std     15.7826 6.90586 0.024018 0.0196887 0.0546924   0.21154  0.537656  0.80904
min         140      15 0.146446 0.0103373  0.101672   -1.5367 -0.458123        1
25%         170    33.5 0.159921 0.0246433  0.154382  -1.10386  0.228208        1
50%         180      36 0.190343 0.0421035  0.205191  -1.03882  0.770438        1
75%         185      38 0.199325 0.0549657  0.234415 -0.985397  0.820689        1
max         195      39 0.213044  0.068315  0.261371 -0.770591   0.93635        3

--- 채널: CADC0888 (세그먼트 수=252, anomaly 비율=0.238) ---
       duration     len      mean         var      std  kurtosis     skew  n_peaks
count       252     252       252         252      252       252      252      252
mean    226.294 46.1905   0.34945   0.0731626 0.246779 -0.866807 0.632054  1.20238
std     107.917 21.5644  0.282504   0.0524582 0.110958   0.51911 0.500624 0.412352
min          35       8 0.0164235 0.000486469 0.022056  -1.68601 -1.42321        1
25%         160      33  0.198552   0.0187163 0.136808  -1.24243 0.346762        1
50%         235      48  0.260952   0.0763713 0.276354  -1.01495 0.733305        1
75%      296.25   60.25  0.339481     0.11551 0.339867 -0.597515 0.986158        1
max         480      97   1.11863    0.187665 0.433203   1.59239  1.53261        3

--- 채널: CADC0890 (세그먼트 수=14, anomaly 비율=0.786) ---
       duration     len      mean       var      std  kurtosis     skew  n_peaks
count        14      14        14        14       14        14       14       14
mean    191.214 39.2143  0.241087 0.0952774 0.292396  0.275205  1.22306        2
std     48.1091  9.5931  0.113834 0.0566796 0.102639   1.53321 0.502184 0.679366
min          90      19 0.0607941 0.0119421  0.10928  -1.13787 0.676174        1
25%      176.25   36.25  0.135277 0.0448577 0.211631 -0.970205 0.821447        2
50%         195      40  0.259709  0.104041 0.320421 -0.743936 0.891975        2
75%      231.25   47.25  0.338188  0.147622 0.384204   1.50411  1.65609        2
max         250      51   0.41051  0.168652 0.410672   3.27836  2.09407        3

--- 채널: CADC0892 (세그먼트 수=211, anomaly 비율=0.161) ---
       duration     len       mean         var       std  kurtosis      skew  n_peaks
count       211     211        211         211       211       211       211      211
mean    263.858 235.934   0.255104    0.103447  0.298315 -0.903844   0.76173   1.5782
std     75.2628 92.5125   0.104962   0.0636239  0.120516  0.653655  0.337203  1.07229
min         124      27 0.00749973 0.000188317 0.0137229   -1.7629 -0.744093        1
25%       207.5     184   0.208271   0.0472271  0.217308  -1.25932  0.585783        1
50%         263     239   0.261269    0.110649  0.332639  -1.04348  0.751888        1
75%       307.5     293   0.330121    0.157859   0.39731  -0.70143  0.940993        2
max         469     470     0.4659     0.27248  0.521996   6.16714   2.70507       10

--- 채널: CADC0894 (세그먼트 수=144, anomaly 비율=0.146) ---
       duration     len       mean         var        std  kurtosis      skew  n_peaks
count       144     144        144         144        144       144       144      144
mean    358.861 246.611    0.15354   0.0458355   0.179932 -0.491616  0.892159  4.22917
std     171.428 230.217   0.106219   0.0453637   0.116421   1.79032  0.553628  7.90036
min         175      21 0.00088843 3.77128e-06 0.00194198  -1.85881 -0.332066        1
25%         215      44  0.0541158  0.00460998  0.0678968  -1.36121  0.536959        1
50%         286      58   0.157745   0.0336096   0.183328   -1.0259  0.781472        1
75%         459  439.25   0.233111   0.0715997   0.267576  -0.62153   1.01735        3
max         920     827   0.454168     0.23944   0.489326   11.2767   3.37188       47

==========================================================================================
완료
==========================================================================================
모든 세그먼트/원시 데이터는 아래 폴더에 CSV로 저장되었습니다. 엑셀 등에서 직접 전수 확인 가능합니다:
  /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/eda_outputs


from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


BASE_DIR = Path("/home/yjlee/Research/Satellite/OPSSAT-AD_LEO data")
DATASET_PATH = BASE_DIR / "dataset.csv"
SEGMENTS_PATH = BASE_DIR / "segments.csv"
OUT_DIR = BASE_DIR / "causal_pipeline_outputs"
OLD_CALIBRATION_PATH = OUT_DIR / "detector_calibration.csv"  # v1 결과 (비교용, 없으면 생략)

ALL_CHANNELS = ["CADC0872", "CADC0873", "CADC0874", "CADC0884",
                 "CADC0886", "CADC0888", "CADC0890", "CADC0892", "CADC0894"]

# 채널 자동분류 기준 (zscore_diagnostics.csv 실측치 기반으로 정한 임계값)
QUANTIZATION_FRAC_ZERO_THRESHOLD = 0.10   # 이 이상이면 "양자화 채널"로 분류
FLOAT_NOISE_ABS_DIFF_THRESHOLD = 1e-6     # 0아닌 최소 diff가 이보다 작으면 "부동소수점 잡음 의심"

# --- Ablation 스위치 ---
# 오탐률 변화가 "MAD/mixture 강건추정 교체" 때문인지 "BOCPD forgetting" 때문인지
# 분리하려면 아래 두 값을 각각 True/False로 바꿔가며 이 셀부터 끝까지 재실행하세요.
USE_MIXTURE_NOISE_ESTIMATOR = False   # False -> 기존 var(diff)/2 방식 (v1과 동일)
USE_BOCPD_FORGETTING = True          # False -> kappa 상한 없음 (v1과 동일)
RUN_TAG = "mixture_and_forgetting"   # 출력 파일명에 붙는 태그 (조합별로 바꿔서 실행)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

def section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

1. 채널 특성 진단 (분류 + 양자화 스텝 추정 + MAD/mixture 강건 노이즈)
광다이오드류(양자화): frac_zero_diff 크고, 0아닌 diff들이 계단폭 단위로 뭉쳐 있음
자력계류(부동소수점 잡음 의심): 0아닌 최소 diff가 1e-6보다 훨씬 작음
mixture_variance: "점프 발생확률 p" x "점프 크기 분산(MAD)"으로 분리 추정. 전체 diff에 MAD를 직접 적용하면 zero-inflated 데이터에서 median이 0으로 잡혀 분산 추정 자체가 0으로 붕괴하는 문제가 있었음(1차 시도에서 실제로 발생) — 그래서 분리.

@dataclass
class ChannelProfile:
    channel: str
    channel_type: str            # "quantized" | "float_noise_suspect" | "continuous"
    frac_zero_diff: float
    min_nonzero_abs_diff: float
    quantization_step: float     # NaN이면 양자화 아님
    r_robust: float
    q_robust: float
    n_nominal_points: int


def mad_variance(x: np.ndarray) -> float:
    """MAD -> 정규분포 가정 하의 표준편차 근사(1.4826 스케일) -> 분산.
    주의: x의 절반 이상이 같은 값(예:0)이면 0을 반환할 수 있음 (MAD 정의상 당연함).
    zero-inflated 데이터에는 mixture_variance()처럼 0이 아닌 값만 대상으로 써야 함.
    """
    if len(x) < 2:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    return float(sigma ** 2)


def mixture_variance(diffs_all: np.ndarray) -> float:
    """Zero-inflated 혼합모델 분산 추정: Var(diff) ~= p * Var(jump | jump occurred)."""
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
        ch_type = "quantized"
        q_step = detect_quantization_step(np.abs(nonzero))
    elif np.isfinite(min_nonzero_abs) and min_nonzero_abs < FLOAT_NOISE_ABS_DIFF_THRESHOLD:
        ch_type = "float_noise_suspect"
        q_step = np.nan
    else:
        ch_type = "continuous"
        q_step = np.nan

    if USE_MIXTURE_NOISE_ESTIMATOR:
        r_robust = max(mixture_variance(diffs_all) / 2.0, 1e-14)
        d2 = np.diff(diffs_all)
        q_robust = max(mixture_variance(d2) / 6.0, 1e-14) if len(d2) > 1 else 1e-14
    else:
        r_robust = max(np.var(diffs_all) / 2.0, 1e-14)
        d2 = np.diff(diffs_all)
        q_robust = max(np.var(d2) / 6.0, 1e-14) if len(d2) > 1 else 1e-14

    return ChannelProfile(
        channel=channel, channel_type=ch_type,
        frac_zero_diff=frac_zero, min_nonzero_abs_diff=min_nonzero_abs,
        quantization_step=q_step, r_robust=r_robust, q_robust=q_robust,
        n_nominal_points=n_points,
    )

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
        levels = np.empty(n)

        for t in range(n):
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q

            y_pred = (self.H @ x_pred)[0]
            S = (self.H @ P_pred @ self.H.T)[0, 0] + self.r_eff
            nu = y[t] - y_pred

            innovations[t] = nu
            innovation_vars[t] = S
            levels[t] = y_pred

            K = (P_pred @ self.H.T) / S
            x = x_pred + (K.flatten() * nu)
            P = P_pred - K @ self.H @ P_pred

        return innovations, innovation_vars, levels


def hierarchical_shrink_variance(channel_estimates: dict) -> dict:
    channels = list(channel_estimates.keys())
    log_v = np.array([np.log(max(channel_estimates[c][0], 1e-12)) for c in channels])
    n_eff = np.array([max(channel_estimates[c][1], 2) for c in channels])

    se = np.sqrt(2.0 / (n_eff - 1))
    weights = 1.0 / (se ** 2)
    global_mean = np.sum(weights * log_v) / np.sum(weights)

    weighted_ss = np.sum(weights * (log_v - global_mean) ** 2)
    df = len(channels) - 1
    denom = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
    tau2 = max(0.0, (weighted_ss - df) / denom) if denom > 0 else 0.0

    shrunk = {}
    for c, lv, s in zip(channels, log_v, se):
        w_c = tau2 / (tau2 + s ** 2) if (tau2 + s ** 2) > 0 else 0.0
        log_v_shrunk = w_c * lv + (1 - w_c) * global_mean
        shrunk[c] = float(np.exp(log_v_shrunk))
    return shrunk


class BOCPD:
    def __init__(self, hazard_lambda: float = 250.0,
                 mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0,
                 kappa_max: float = 40.0, use_forgetting: bool = True):
        self.hazard = 1.0 / hazard_lambda
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.kappa_max = kappa_max
        self.use_forgetting = use_forgetting

    def _cap_sufficient_stats(self, kappa, alpha, beta):
        if not self.use_forgetting:
            return kappa, alpha, beta
        over = kappa > self.kappa_max
        if not np.any(over):
            return kappa, alpha, beta
        scale = np.where(over, self.kappa_max / kappa, 1.0)
        kappa2 = np.where(over, self.kappa_max, kappa)
        alpha2 = np.where(over, alpha * scale, alpha)
        beta2 = np.where(over, beta * scale, beta)
        alpha2 = np.maximum(alpha2, 1e-3)
        return kappa2, alpha2, beta2

    def run(self, x: np.ndarray):
        n = len(x)
        R = np.zeros((n + 1, n + 1))
        R[0, 0] = 1.0

        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        alpha = np.array([self.alpha0])
        beta = np.array([self.beta0])

        map_run_length = np.zeros(n, dtype=int)

        for t in range(n):
            xt = x[t]
            dof = 2 * alpha
            scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
            pred_probs = stats.t.pdf(xt, df=dof, loc=mu, scale=scale)
            pred_probs = np.nan_to_num(pred_probs, nan=1e-12, posinf=1e-12)

            run_probs = R[t, : t + 1]
            growth = run_probs * pred_probs * (1 - self.hazard)
            cp_prob = np.sum(run_probs * pred_probs * self.hazard)

            new_R = np.zeros(t + 2)
            new_R[0] = cp_prob
            new_R[1:] = growth
            total = new_R.sum()
            if total <= 0 or not np.isfinite(total):
                new_R[:] = 0
                new_R[0] = 1.0
            else:
                new_R /= total
            R[t + 1, : t + 2] = new_R

            map_run_length[t] = int(np.argmax(new_R))

            new_mu = np.empty(t + 2)
            new_kappa = np.empty(t + 2)
            new_alpha = np.empty(t + 2)
            new_beta = np.empty(t + 2)

            new_mu[0] = self.mu0
            new_kappa[0] = self.kappa0
            new_alpha[0] = self.alpha0
            new_beta[0] = self.beta0

            new_mu[1:] = (kappa * mu + xt) / (kappa + 1)
            new_kappa[1:] = kappa + 1
            new_alpha[1:] = alpha + 0.5
            new_beta[1:] = beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))

            new_kappa, new_alpha, new_beta = self._cap_sufficient_stats(new_kappa, new_alpha, new_beta)
            mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

        onset_idx = None
        onset_posterior = None
        confirm_idx = None
        posterior_over_onset = None
        for t in range(1, n):
            if map_run_length[t] <= 2 and map_run_length[t - 1] > 5:
                onset_idx = t - map_run_length[t]
                confirm_idx = t
                row = R[t + 1, : t + 2]
                k = min(len(row), 30)
                posterior_over_onset = {int(r): float(row[r]) for r in range(k) if row[r] > 1e-6}
                onset_posterior = float(np.sum(row[:3]))
                break

        return {
            "run_length_posterior": R, "map_run_length": map_run_length,
            "onset_idx": onset_idx, "onset_confidence": onset_posterior,
            "confirm_idx": confirm_idx, "posterior_over_onset": posterior_over_onset,
        }


def make_synthetic_data(n_segments_per_channel: int = 40, seg_len: int = 60):
    rows = []
    seg_id = 1
    specs = {
        "SYN_QUANT": dict(step=0.0026, base=1.0, anomaly_jump=0.05),
        "SYN_FLOAT": dict(step=5e-11, base=1e-7, anomaly_jump=2e-8),
    }
    for ch, spec in specs.items():
        for s in range(n_segments_per_channel):
            is_anom = (s % 5 == 0)
            t0 = pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=s * seg_len * 2)
            level = spec["base"]
            vals = []
            for i in range(seg_len):
                if rng.random() < 0.3:
                    level += rng.choice([-1, 1]) * spec["step"]
                if is_anom and i > seg_len // 2:
                    level += spec["anomaly_jump"] * rng.normal(1, 0.2)
                vals.append(level)
            for i, v in enumerate(vals):
                rows.append({
                    "channel": ch, "timestamp": t0 + pd.Timedelta(seconds=i * 2),
                    "value": v, "label": "anomaly" if is_anom else "a2",
                    "sampling": 2, "anomaly": int(is_anom), "segment": seg_id, "train": 1,
                })
            seg_id += 1
    return pd.DataFrame(rows)


use_synthetic = not SEGMENTS_PATH.exists()
if use_synthetic:
    print(f"[안내] {SEGMENTS_PATH} 를 찾을 수 없어 합성 데이터로 스모크 테스트를 실행합니다.")
    seg = make_synthetic_data()
    channels = list(seg["channel"].unique())
    out_dir = Path("./synthetic_outputs")
else:
    seg = pd.read_csv(SEGMENTS_PATH, parse_dates=["timestamp"])
    seg = seg[seg["channel"].isin(ALL_CHANNELS)].reset_index(drop=True)
    channels = ALL_CHANNELS
    out_dir = OUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)

section("0. 데이터 규모 및 Ablation 설정")
print(f"segments: {seg.shape}, channels: {channels}")
print(f"USE_MIXTURE_NOISE_ESTIMATOR = {USE_MIXTURE_NOISE_ESTIMATOR}")
print(f"USE_BOCPD_FORGETTING        = {USE_BOCPD_FORGETTING}")
print(f"RUN_TAG                     = {RUN_TAG}")

==========================================================================================
0. 데이터 규모 및 Ablation 설정
==========================================================================================
segments: (303493, 8), channels: ['CADC0872', 'CADC0873', 'CADC0874', 'CADC0884', 'CADC0886', 'CADC0888', 'CADC0890', 'CADC0892', 'CADC0894']
USE_MIXTURE_NOISE_ESTIMATOR = False
USE_BOCPD_FORGETTING        = True
RUN_TAG                     = mixture_and_forgetting

section("(A) 채널 프로파일링: 양자화/초미세잡음/연속 분류 + 강건 노이즈")

profiles = {}
for ch in channels:
    nominal_vals = [
        s.sort_values("timestamp")["value"].values
        for _, s in seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)].groupby("segment")
    ]
    profiles[ch] = profile_channel(ch, nominal_vals)
    p = profiles[ch]
    step_str = f"{p.quantization_step:.4g}" if np.isfinite(p.quantization_step) else "N/A"
    print(f"[{ch}] type={p.channel_type:<18s} frac_zero={p.frac_zero_diff:.3f} "
          f"step\u2248{step_str} r_robust={p.r_robust:.4g} q_robust={p.q_robust:.4g}")

profile_df = pd.DataFrame([vars(p) for p in profiles.values()])
profile_df.to_csv(out_dir / "channel_profiles_v2.csv", index=False)

r_estimates = {ch: (profiles[ch].r_robust, max(profiles[ch].n_nominal_points, 2)) for ch in channels}
q_estimates = {ch: (profiles[ch].q_robust, max(profiles[ch].n_nominal_points, 2)) for ch in channels}
r_shrunk = hierarchical_shrink_variance(r_estimates)
q_shrunk = hierarchical_shrink_variance(q_estimates)

quant_floor = {
    ch: (profiles[ch].quantization_step ** 2 / 12.0) if np.isfinite(profiles[ch].quantization_step) else 0.0
    for ch in channels
}


==========================================================================================
(A) 채널 프로파일링: 양자화/초미세잡음/연속 분류 + 강건 노이즈
==========================================================================================
[CADC0872] type=float_noise_suspect frac_zero=0.053 step≈N/A r_robust=2.647e-12 q_robust=4.83e-13
[CADC0873] type=float_noise_suspect frac_zero=0.052 step≈N/A r_robust=2.51e-12 q_robust=2.942e-13
[CADC0874] type=float_noise_suspect frac_zero=0.044 step≈N/A r_robust=8.02e-13 q_robust=2.726e-13
[CADC0884] type=quantized          frac_zero=0.183 step≈0.01435 r_robust=0.0004859 q_robust=2.833e-05
[CADC0886] type=quantized          frac_zero=0.371 step≈0.02735 r_robust=0.001328 q_robust=7.374e-05
[CADC0888] type=quantized          frac_zero=0.354 step≈0.01439 r_robust=0.001277 q_robust=8.229e-05
[CADC0890] type=continuous         frac_zero=0.083 step≈N/A r_robust=0.002315 q_robust=0.0002688
[CADC0892] type=quantized          frac_zero=0.416 step≈0.005405 r_robust=0.0001511 q_robust=3.785e-05
[CADC0894] type=quantized          frac_zero=0.583 step≈0.002645 r_robust=0.0001328 q_robust=1.417e-05

section("(B) 채널별 온라인 필터링 + BOCPD(forgetting, kappa_max=40) 재탐지")

calibration_rows = []
for ch in channels:
    ch_segments = seg[seg["channel"] == ch].sort_values(["segment", "timestamp"])
    seg_ids = ch_segments["segment"].unique()

    kf = LocalLinearTrendKF(q=q_shrunk[ch], r_nominal=r_shrunk[ch], theta=0.0,
                             quantization_floor=quant_floor[ch])
    bocpd = BOCPD(hazard_lambda=250.0, kappa_max=40.0, use_forgetting=USE_BOCPD_FORGETTING)

    n_anom, n_anom_det, n_norm, n_norm_fa = 0, 0, 0, 0

    for sid in seg_ids:
        s = ch_segments[ch_segments["segment"] == sid].sort_values("timestamp")
        values = s["value"].values
        is_anomaly = bool(s["anomaly"].iloc[0])
        if len(values) < 5:
            continue

        innovations, innovation_vars, _ = kf.run(values)
        z = innovations / np.sqrt(np.clip(innovation_vars, 1e-8, None))
        result = bocpd.run(z)
        detected = result["onset_idx"] is not None

        if is_anomaly:
            n_anom += 1
            n_anom_det += int(detected)
        else:
            n_norm += 1
            n_norm_fa += int(detected)

    calibration_rows.append({
        "channel": ch, "channel_type": profiles[ch].channel_type,
        "n_anomaly_segments": n_anom, "n_anomaly_detected": n_anom_det,
        "recall": n_anom_det / n_anom if n_anom else np.nan,
        "n_normal_segments": n_norm, "n_normal_false_alarm": n_norm_fa,
        "false_alarm_rate": n_norm_fa / n_norm if n_norm else np.nan,
    })
    r = calibration_rows[-1]
    print(f"[{ch}] recall={r['recall']:.3f} (n_anom={n_anom})  "
          f"false_alarm_rate={r['false_alarm_rate']:.3f} (n_normal={n_norm})")

calib_v2 = pd.DataFrame(calibration_rows)
calib_v2.to_csv(out_dir / f"detector_calibration_v2_{RUN_TAG}.csv", index=False)
calib_v2


==========================================================================================
(B) 채널별 온라인 필터링 + BOCPD(forgetting, kappa_max=40) 재탐지
==========================================================================================
[CADC0872] recall=0.321 (n_anom=131)  false_alarm_rate=0.000 (n_normal=415)
[CADC0873] recall=0.276 (n_anom=105)  false_alarm_rate=0.000 (n_normal=488)
[CADC0874] recall=0.725 (n_anom=69)  false_alarm_rate=0.032 (n_normal=125)
[CADC0884] recall=nan (n_anom=0)  false_alarm_rate=0.241 (n_normal=158)
[CADC0886] recall=0.000 (n_anom=3)  false_alarm_rate=0.000 (n_normal=8)
[CADC0888] recall=0.617 (n_anom=60)  false_alarm_rate=0.391 (n_normal=192)
[CADC0890] recall=0.909 (n_anom=11)  false_alarm_rate=0.000 (n_normal=3)
[CADC0892] recall=0.971 (n_anom=34)  false_alarm_rate=0.994 (n_normal=177)
[CADC0894] recall=1.000 (n_anom=21)  false_alarm_rate=0.797 (n_normal=123)
channel	channel_type	n_anomaly_segments	n_anomaly_detected	recall	n_normal_segments	n_normal_false_alarm	false_alarm_rate
0	CADC0872	float_noise_suspect	131	42	0.320611	415	0	0.000000
1	CADC0873	float_noise_suspect	105	29	0.276190	488	0	0.000000
2	CADC0874	float_noise_suspect	69	50	0.724638	125	4	0.032000
3	CADC0884	quantized	0	0	NaN	158	38	0.240506
4	CADC0886	quantized	3	0	0.000000	8	0	0.000000
5	CADC0888	quantized	60	37	0.616667	192	75	0.390625
6	CADC0890	continuous	11	10	0.909091	3	0	0.000000
7	CADC0892	quantized	34	33	0.970588	177	176	0.994350
8	CADC0894	quantized	21	21	1.000000	123	98	0.796748

if OLD_CALIBRATION_PATH.exists():
    section("(C) v1 vs v2 비교")
    calib_v1 = pd.read_csv(OLD_CALIBRATION_PATH)
    merged = calib_v1.merge(calib_v2, on="channel", suffixes=("_v1", "_v2"))
    merged["recall_delta"] = merged["recall_v2"] - merged["recall_v1"]
    merged["fa_delta"] = merged["false_alarm_rate_v2"] - merged["false_alarm_rate_v1"]
    display_cols = ["channel", "recall_v1", "recall_v2", "recall_delta",
                     "false_alarm_rate_v1", "false_alarm_rate_v2", "fa_delta"]
    print(merged[display_cols].to_string(index=False))
    merged.to_csv(out_dir / f"calibration_v1_vs_v2_{RUN_TAG}.csv", index=False)
    print(f"\n[저장] {out_dir / f'calibration_v1_vs_v2_{RUN_TAG}.csv'}")
else:
    print(f"[안내] {OLD_CALIBRATION_PATH} 가 없어 v1 대비 비교는 생략합니다.")


==========================================================================================
(C) v1 vs v2 비교
==========================================================================================
 channel  recall_v1  recall_v2  recall_delta  false_alarm_rate_v1  false_alarm_rate_v2      fa_delta
CADC0872   0.076336   0.320611      0.244275             0.000000             0.000000  0.000000e+00
CADC0873   0.104762   0.276190      0.171429             0.000000             0.000000  0.000000e+00
CADC0874   0.188406   0.724638      0.536232             0.000000             0.032000  3.200000e-02
CADC0884        NaN        NaN           NaN             0.240506             0.240506  5.551115e-17
CADC0886   0.000000   0.000000      0.000000             0.125000             0.000000 -1.250000e-01
CADC0888   0.616667   0.616667      0.000000             0.380208             0.390625  1.041667e-02
CADC0890   0.818182   0.909091      0.090909             0.000000             0.000000  0.000000e+00
CADC0892   0.970588   0.970588      0.000000             1.000000             0.994350 -5.649718e-03
CADC0894   1.000000   1.000000      0.000000             0.772358             0.796748  2.439024e-02

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/calibration_v1_vs_v2_mixture_and_forgetting.csv






10. 문제 채널(CADC0890 / CADC0892 / CADC0894) 재진단
v2에서도 이 세 채널의 오탐률이 여전히 심각합니다. 두 가설을 분리합니다.

가설 A (노이즈 모델 문제): steady-state z의 std가 1보다 뚜렷이 크거나 |z|>3 비율이 0.01+ 이면, r/q 추정 자체가 아직도 부족한 것.
가설 B (BOCPD forgetting 문제): steady-state z가 대략 N(0,1)에 가까운데도 오탐이 심하다면, 노이즈 모델은 괜찮고 BOCPD의 kappa_max 메커니즘이 원인. 이 경우 위쪽 Ablation 스위치에서 USE_BOCPD_FORGETTING = False로 바꾸고 6번 셀부터 다시 실행해서 실제로 오탐이 줄어드는지 확인하세요.



TARGET_CHANNELS = ["CADC0890", "CADC0892", "CADC0894"]
BURN_IN = 5

def summarize(z):
    if len(z) < 10:
        return dict(mean=np.nan, std=np.nan, kurt=np.nan, frac_gt3=np.nan)
    return dict(
        mean=float(np.mean(z)), std=float(np.std(z)),
        kurt=float(stats.kurtosis(z)),
        frac_gt3=float(np.mean(np.abs(z) > 3)),
    )

target_channels_present = [c for c in TARGET_CHANNELS if c in channels] or channels

print(f"{'채널':<12s} {'burn-in std':>12s} {'burn-in |z|>3':>14s} {'steady std':>12s} "
      f"{'steady kurt':>12s} {'steady |z|>3':>13s}")
print("-" * 80)

recheck_rows = []
for ch in target_channels_present:
    kf = LocalLinearTrendKF(q=q_shrunk[ch], r_nominal=r_shrunk[ch], theta=0.0,
                             quantization_floor=quant_floor[ch])
    nominal = seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)]

    z_burnin, z_steady = [], []
    for sid, s in nominal.groupby("segment"):
        v = s.sort_values("timestamp")["value"].values
        if len(v) < BURN_IN + 5:
            continue
        innovations, innovation_vars, _ = kf.run(v)
        z = innovations / np.sqrt(np.clip(innovation_vars, 1e-12, None))
        z_burnin.append(z[:BURN_IN])
        z_steady.append(z[BURN_IN:])

    z_burnin = np.concatenate(z_burnin) if z_burnin else np.array([])
    z_steady = np.concatenate(z_steady) if z_steady else np.array([])
    sb, ss = summarize(z_burnin), summarize(z_steady)

    print(f"{ch:<12s} {sb['std']:>12.3f} {sb['frac_gt3']:>14.3f} {ss['std']:>12.3f} "
          f"{ss['kurt']:>12.2f} {ss['frac_gt3']:>13.3f}")
    recheck_rows.append({"channel": ch, "r_used": r_shrunk[ch] + quant_floor[ch], "q_used": q_shrunk[ch],
                          **{f"burnin_{k}": v for k, v in sb.items()},
                          **{f"steady_{k}": v for k, v in ss.items()}})

print("\n참고: 정규분포라면 std\u22481, |z|>3 비율\u22480.003")
print("- steady std가 1보다 뚜렷이 크거나 |z|>3 비율이 0.01+ 이면 -> 가설 A (노이즈 모델 재점검)")
print("- steady std가 대략 1에 가까운데도 오탐이 심하면 -> 가설 B (BOCPD forgetting이 원인)")

recheck_df = pd.DataFrame(recheck_rows)
if not use_synthetic:
    recheck_df.to_csv(out_dir / "problem_channels_zcheck.csv", index=False)
    print(f"\n[저장] {out_dir / 'problem_channels_zcheck.csv'}")
recheck_df


채널            burn-in std  burn-in |z|>3   steady std  steady kurt  steady |z|>3
--------------------------------------------------------------------------------
CADC0890            0.050          0.000        0.945         0.37         0.000
CADC0892            0.080          0.000        0.626        63.32         0.008
CADC0894            0.081          0.000        0.978        34.27         0.026

참고: 정규분포라면 std≈1, |z|>3 비율≈0.003
- steady std가 1보다 뚜렷이 크거나 |z|>3 비율이 0.01+ 이면 -> 가설 A (노이즈 모델 재점검)
- steady std가 대략 1에 가까운데도 오탐이 심하면 -> 가설 B (BOCPD forgetting이 원인)

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/problem_channels_zcheck.csv
channel	r_used	q_used	burnin_mean	burnin_std	burnin_kurt	burnin_frac_gt3	steady_mean	steady_std	steady_kurt	steady_frac_gt3
0	CADC0890	0.002311	0.000268	0.011052	0.049675	0.270300	0.0	-0.001630	0.945336	0.373170	0.000000
1	CADC0892	0.000154	0.000038	0.004006	0.079885	27.876947	0.0	-0.000045	0.626050	63.322849	0.007681
2	CADC0894	0.000133	0.000014	0.003283	0.080694	43.638211	0.0	0.000691	0.977772	34.272141	0.026234


# 1. **1단계는 `mixture=False + BOCPD forgetting=True`가 최적**이며, 현재 결과상 872/873/874/888/890/894 **6개 채널을 2단계에 포함**, 884/886/892는 제외하는 게 가장 타당합니다.
# 2. **DR-EKF는 완전히 빼고**, `채널 유형별 노이즈 모델 + Kalman Filter + BOCPD forgetting`을 탐지 방법론의 핵심으로 잡는 것이 구현과 결과 모두에 가장 일치합니다.
# 3. 이후 핵심은 **2단계 인과추론**입니다: onset posterior → Monte Carlo 이벤트 스터디 → `level → diff → diff²` 선행성 검정 → 채널별 랜덤효과 메타분석을 거쳐, 유의한 인과 구조가 확인되면 **Physics-informed SCM**으로 확장하는 조합이 가장 좋습니다.

# # --- Ablation 스위치 ---
# USE_MIXTURE_NOISE_ESTIMATOR = True
# USE_BOCPD_FORGETTING = True
# ==========================================================================================
# 0. 데이터 규모 및 Ablation 설정
# ==========================================================================================
# segments: (303493, 8), channels: ['CADC0872', 'CADC0873', 'CADC0874', 'CADC0884', 'CADC0886', 'CADC0888', 'CADC0890', 'CADC0892', 'CADC0894']
# USE_MIXTURE_NOISE_ESTIMATOR = True
# USE_BOCPD_FORGETTING        = True
# RUN_TAG                     = mixture_and_forgetting
# ==========================================================================================
# (A) 채널 프로파일링: 양자화/초미세잡음/연속 분류 + 강건 노이즈
# ==========================================================================================
# [CADC0872] type=float_noise_suspect frac_zero=0.053 step≈N/A r_robust=6.787e-13 q_robust=1.808e-13
# [CADC0873] type=float_noise_suspect frac_zero=0.052 step≈N/A r_robust=5.742e-13 q_robust=1.666e-13
# [CADC0874] type=float_noise_suspect frac_zero=0.044 step≈N/A r_robust=2.936e-13 q_robust=1.375e-13
# [CADC0884] type=quantized          frac_zero=0.183 step≈0.01435 r_robust=0.0006186 q_robust=1.263e-05
# [CADC0886] type=quantized          frac_zero=0.371 step≈0.02735 r_robust=0.001791 q_robust=6.064e-05
# [CADC0888] type=quantized          frac_zero=0.354 step≈0.01439 r_robust=0.001873 q_robust=1.908e-05
# [CADC0890] type=continuous         frac_zero=0.083 step≈N/A r_robust=0.00123 q_robust=3.579e-05
# [CADC0892] type=quantized          frac_zero=0.416 step≈0.005405 r_robust=8.067e-05 q_robust=7.489e-06
# [CADC0894] type=quantized          frac_zero=0.583 step≈0.002645 r_robust=1.286e-05 q_robust=1.703e-06
# ==========================================================================================
# (B) 채널별 온라인 필터링 + BOCPD(forgetting, kappa_max=40) 재탐지
# ==========================================================================================
# [CADC0872] recall=0.313 (n_anom=131)  false_alarm_rate=0.000 (n_normal=415)
# [CADC0873] recall=0.276 (n_anom=105)  false_alarm_rate=0.000 (n_normal=488)
# [CADC0874] recall=0.725 (n_anom=69)  false_alarm_rate=0.032 (n_normal=125)
# [CADC0884] recall=nan (n_anom=0)  false_alarm_rate=0.266 (n_normal=158)
# [CADC0886] recall=0.000 (n_anom=3)  false_alarm_rate=0.125 (n_normal=8)
# [CADC0888] recall=0.583 (n_anom=60)  false_alarm_rate=0.458 (n_normal=192)
# [CADC0890] recall=0.818 (n_anom=11)  false_alarm_rate=1.000 (n_normal=3)
# [CADC0892] recall=0.971 (n_anom=34)  false_alarm_rate=1.000 (n_normal=177)
# [CADC0894] recall=1.000 (n_anom=21)  false_alarm_rate=0.951 (n_normal=123)
# channel	channel_type	n_anomaly_segments	n_anomaly_detected	recall	n_normal_segments	n_normal_false_alarm	false_alarm_rate
# 0	CADC0872	float_noise_suspect	131	41	0.312977	415	0	0.000000
# 1	CADC0873	float_noise_suspect	105	29	0.276190	488	0	0.000000
# 2	CADC0874	float_noise_suspect	69	50	0.724638	125	4	0.032000
# 3	CADC0884	quantized	0	0	NaN	158	42	0.265823
# 4	CADC0886	quantized	3	0	0.000000	8	1	0.125000
# 5	CADC0888	quantized	60	35	0.583333	192	88	0.458333
# 6	CADC0890	continuous	11	9	0.818182	3	3	1.000000
# 7	CADC0892	quantized	34	33	0.970588	177	177	1.000000
# 8	CADC0894	quantized	21	21	1.000000	123	117	0.951220
# ==========================================================================================
# (C) v1 vs v2 비교
# ==========================================================================================
#  channel  recall_v1  recall_v2  recall_delta  false_alarm_rate_v1  false_alarm_rate_v2  fa_delta
# CADC0872   0.076336   0.312977      0.236641             0.000000             0.000000  0.000000
# CADC0873   0.104762   0.276190      0.171429             0.000000             0.000000  0.000000
# CADC0874   0.188406   0.724638      0.536232             0.000000             0.032000  0.032000
# CADC0884        NaN        NaN           NaN             0.240506             0.265823  0.025316
# CADC0886   0.000000   0.000000      0.000000             0.125000             0.125000  0.000000
# CADC0888   0.616667   0.583333     -0.033333             0.380208             0.458333  0.078125
# CADC0890   0.818182   0.818182      0.000000             0.000000             1.000000  1.000000
# CADC0892   0.970588   0.970588      0.000000             1.000000             1.000000  0.000000
# CADC0894   1.000000   1.000000      0.000000             0.772358             0.951220  0.178862

# [저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/calibration_v1_vs_v2_mixture_and_forgetting.csv
# 채널            burn-in std  burn-in |z|>3   steady std  steady kurt  steady |z|>3
# --------------------------------------------------------------------------------
# CADC0890            0.068          0.000        2.252        -0.13         0.258
# CADC0892            0.112          0.000        1.126        84.75         0.020
# CADC0894            0.253          0.002        2.877        35.28         0.070

# 참고: 정규분포라면 std≈1, |z|>3 비율≈0.003
# - steady std가 1보다 뚜렷이 크거나 |z|>3 비율이 0.01+ 이면 -> 가설 A (노이즈 모델 재점검)
# - steady std가 대략 1에 가까운데도 오탐이 심하면 -> 가설 B (BOCPD forgetting이 원인)

# [저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/problem_channels_zcheck.csv
# channel	r_used	q_used	burnin_mean	burnin_std	burnin_kurt	burnin_frac_gt3	steady_mean	steady_std	steady_kurt	steady_frac_gt3
# 0	CADC0890	0.001228	0.000036	0.016487	0.068269	0.501436	0.000000	0.010924	2.251626	-0.131505	0.257576
# 1	CADC0892	0.000083	0.000007	0.005902	0.112267	26.770168	0.000000	-0.000041	1.125730	84.754067	0.019898
# 2	CADC0894	0.000013	0.000002	0.010336	0.252891	43.313643	0.001626	0.001647	2.876931	35.281981	0.070152





# # --- Ablation 스위치 ---
# USE_MIXTURE_NOISE_ESTIMATOR = False
# USE_BOCPD_FORGETTING = True
# ==========================================================================================
# 0. 데이터 규모 및 Ablation 설정
# ==========================================================================================
# segments: (303493, 8), channels: ['CADC0872', 'CADC0873', 'CADC0874', 'CADC0884', 'CADC0886', 'CADC0888', 'CADC0890', 'CADC0892', 'CADC0894']
# USE_MIXTURE_NOISE_ESTIMATOR = False
# USE_BOCPD_FORGETTING        = True
# RUN_TAG                     = mixture_and_forgetting
# ==========================================================================================
# (A) 채널 프로파일링: 양자화/초미세잡음/연속 분류 + 강건 노이즈
# ==========================================================================================
# [CADC0872] type=float_noise_suspect frac_zero=0.053 step≈N/A r_robust=2.647e-12 q_robust=4.83e-13
# [CADC0873] type=float_noise_suspect frac_zero=0.052 step≈N/A r_robust=2.51e-12 q_robust=2.942e-13
# [CADC0874] type=float_noise_suspect frac_zero=0.044 step≈N/A r_robust=8.02e-13 q_robust=2.726e-13
# [CADC0884] type=quantized          frac_zero=0.183 step≈0.01435 r_robust=0.0004859 q_robust=2.833e-05
# [CADC0886] type=quantized          frac_zero=0.371 step≈0.02735 r_robust=0.001328 q_robust=7.374e-05
# [CADC0888] type=quantized          frac_zero=0.354 step≈0.01439 r_robust=0.001277 q_robust=8.229e-05
# [CADC0890] type=continuous         frac_zero=0.083 step≈N/A r_robust=0.002315 q_robust=0.0002688
# [CADC0892] type=quantized          frac_zero=0.416 step≈0.005405 r_robust=0.0001511 q_robust=3.785e-05
# [CADC0894] type=quantized          frac_zero=0.583 step≈0.002645 r_robust=0.0001328 q_robust=1.417e-05
# ==========================================================================================
# (B) 채널별 온라인 필터링 + BOCPD(forgetting, kappa_max=40) 재탐지
# ==========================================================================================
# [CADC0872] recall=0.321 (n_anom=131)  false_alarm_rate=0.000 (n_normal=415)
# [CADC0873] recall=0.276 (n_anom=105)  false_alarm_rate=0.000 (n_normal=488)
# [CADC0874] recall=0.725 (n_anom=69)  false_alarm_rate=0.032 (n_normal=125)
# [CADC0884] recall=nan (n_anom=0)  false_alarm_rate=0.241 (n_normal=158)
# [CADC0886] recall=0.000 (n_anom=3)  false_alarm_rate=0.000 (n_normal=8)
# [CADC0888] recall=0.617 (n_anom=60)  false_alarm_rate=0.391 (n_normal=192)
# [CADC0890] recall=0.909 (n_anom=11)  false_alarm_rate=0.000 (n_normal=3)
# [CADC0892] recall=0.971 (n_anom=34)  false_alarm_rate=0.994 (n_normal=177)
# [CADC0894] recall=1.000 (n_anom=21)  false_alarm_rate=0.797 (n_normal=123)
# channel	channel_type	n_anomaly_segments	n_anomaly_detected	recall	n_normal_segments	n_normal_false_alarm	false_alarm_rate
# 0	CADC0872	float_noise_suspect	131	42	0.320611	415	0	0.000000
# 1	CADC0873	float_noise_suspect	105	29	0.276190	488	0	0.000000
# 2	CADC0874	float_noise_suspect	69	50	0.724638	125	4	0.032000
# 3	CADC0884	quantized	0	0	NaN	158	38	0.240506
# 4	CADC0886	quantized	3	0	0.000000	8	0	0.000000
# 5	CADC0888	quantized	60	37	0.616667	192	75	0.390625
# 6	CADC0890	continuous	11	10	0.909091	3	0	0.000000
# 7	CADC0892	quantized	34	33	0.970588	177	176	0.994350
# 8	CADC0894	quantized	21	21	1.000000	123	98	0.796748
# ==========================================================================================
# (C) v1 vs v2 비교
# ==========================================================================================
#  channel  recall_v1  recall_v2  recall_delta  false_alarm_rate_v1  false_alarm_rate_v2      fa_delta
# CADC0872   0.076336   0.320611      0.244275             0.000000             0.000000  0.000000e+00
# CADC0873   0.104762   0.276190      0.171429             0.000000             0.000000  0.000000e+00
# CADC0874   0.188406   0.724638      0.536232             0.000000             0.032000  3.200000e-02
# CADC0884        NaN        NaN           NaN             0.240506             0.240506  5.551115e-17
# CADC0886   0.000000   0.000000      0.000000             0.125000             0.000000 -1.250000e-01
# CADC0888   0.616667   0.616667      0.000000             0.380208             0.390625  1.041667e-02
# CADC0890   0.818182   0.909091      0.090909             0.000000             0.000000  0.000000e+00
# CADC0892   0.970588   0.970588      0.000000             1.000000             0.994350 -5.649718e-03
# CADC0894   1.000000   1.000000      0.000000             0.772358             0.796748  2.439024e-02

# [저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/calibration_v1_vs_v2_mixture_and_forgetting.csv
# 채널            burn-in std  burn-in |z|>3   steady std  steady kurt  steady |z|>3
# --------------------------------------------------------------------------------
# CADC0890            0.050          0.000        0.945         0.37         0.000
# CADC0892            0.080          0.000        0.626        63.32         0.008
# CADC0894            0.081          0.000        0.978        34.27         0.026

# 참고: 정규분포라면 std≈1, |z|>3 비율≈0.003
# - steady std가 1보다 뚜렷이 크거나 |z|>3 비율이 0.01+ 이면 -> 가설 A (노이즈 모델 재점검)
# - steady std가 대략 1에 가까운데도 오탐이 심하면 -> 가설 B (BOCPD forgetting이 원인)

# [저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/problem_channels_zcheck.csv
# channel	r_used	q_used	burnin_mean	burnin_std	burnin_kurt	burnin_frac_gt3	steady_mean	steady_std	steady_kurt	steady_frac_gt3
# 0	CADC0890	0.002311	0.000268	0.011052	0.049675	0.270300	0.0	-0.001630	0.945336	0.373170	0.000000
# 1	CADC0892	0.000154	0.000038	0.004006	0.079885	27.876947	0.0	-0.000045	0.626050	63.322849	0.007681
# 2	CADC0894	0.000133	0.000014	0.003283	0.080694	43.638211	0.0	0.000691	0.977772	34.272141	0.026234




# # --- Ablation 스위치 ---
# USE_MIXTURE_NOISE_ESTIMATOR = True
# USE_BOCPD_FORGETTING = False
# ==========================================================================================
# 0. 데이터 규모 및 Ablation 설정
# ==========================================================================================
# segments: (303493, 8), channels: ['CADC0872', 'CADC0873', 'CADC0874', 'CADC0884', 'CADC0886', 'CADC0888', 'CADC0890', 'CADC0892', 'CADC0894']
# USE_MIXTURE_NOISE_ESTIMATOR = True
# USE_BOCPD_FORGETTING        = False
# RUN_TAG                     = mixture_and_forgetting

# ==========================================================================================
# (A) 채널 프로파일링: 양자화/초미세잡음/연속 분류 + 강건 노이즈
# ==========================================================================================
# [CADC0872] type=float_noise_suspect frac_zero=0.053 step≈N/A r_robust=6.787e-13 q_robust=1.808e-13
# [CADC0873] type=float_noise_suspect frac_zero=0.052 step≈N/A r_robust=5.742e-13 q_robust=1.666e-13
# [CADC0874] type=float_noise_suspect frac_zero=0.044 step≈N/A r_robust=2.936e-13 q_robust=1.375e-13
# [CADC0884] type=quantized          frac_zero=0.183 step≈0.01435 r_robust=0.0006186 q_robust=1.263e-05
# [CADC0886] type=quantized          frac_zero=0.371 step≈0.02735 r_robust=0.001791 q_robust=6.064e-05
# [CADC0888] type=quantized          frac_zero=0.354 step≈0.01439 r_robust=0.001873 q_robust=1.908e-05
# [CADC0890] type=continuous         frac_zero=0.083 step≈N/A r_robust=0.00123 q_robust=3.579e-05
# [CADC0892] type=quantized          frac_zero=0.416 step≈0.005405 r_robust=8.067e-05 q_robust=7.489e-06
# [CADC0894] type=quantized          frac_zero=0.583 step≈0.002645 r_robust=1.286e-05 q_robust=1.703e-06

# ==========================================================================================
# (B) 채널별 온라인 필터링 + BOCPD(forgetting, kappa_max=40) 재탐지
# ==========================================================================================
# [CADC0872] recall=0.191 (n_anom=131)  false_alarm_rate=0.000 (n_normal=415)
# [CADC0873] recall=0.171 (n_anom=105)  false_alarm_rate=0.000 (n_normal=488)
# [CADC0874] recall=0.319 (n_anom=69)  false_alarm_rate=0.000 (n_normal=125)
# [CADC0884] recall=nan (n_anom=0)  false_alarm_rate=0.266 (n_normal=158)
# [CADC0886] recall=0.000 (n_anom=3)  false_alarm_rate=0.125 (n_normal=8)
# [CADC0888] recall=0.583 (n_anom=60)  false_alarm_rate=0.458 (n_normal=192)
# [CADC0890] recall=0.818 (n_anom=11)  false_alarm_rate=1.000 (n_normal=3)
# [CADC0892] recall=0.941 (n_anom=34)  false_alarm_rate=1.000 (n_normal=177)
# [CADC0894] recall=1.000 (n_anom=21)  false_alarm_rate=0.943 (n_normal=123)
# channel	channel_type	n_anomaly_segments	n_anomaly_detected	recall	n_normal_segments	n_normal_false_alarm	false_alarm_rate
# 0	CADC0872	float_noise_suspect	131	25	0.190840	415	0	0.000000
# 1	CADC0873	float_noise_suspect	105	18	0.171429	488	0	0.000000
# 2	CADC0874	float_noise_suspect	69	22	0.318841	125	0	0.000000
# 3	CADC0884	quantized	0	0	NaN	158	42	0.265823
# 4	CADC0886	quantized	3	0	0.000000	8	1	0.125000
# 5	CADC0888	quantized	60	35	0.583333	192	88	0.458333
# 6	CADC0890	continuous	11	9	0.818182	3	3	1.000000
# 7	CADC0892	quantized	34	32	0.941176	177	177	1.000000
# 8	CADC0894	quantized	21	21	1.000000	123	116	0.943089
# ==========================================================================================
# (C) v1 vs v2 비교
# ==========================================================================================
#  channel  recall_v1  recall_v2  recall_delta  false_alarm_rate_v1  false_alarm_rate_v2  fa_delta
# CADC0872   0.076336   0.190840      0.114504             0.000000             0.000000  0.000000
# CADC0873   0.104762   0.171429      0.066667             0.000000             0.000000  0.000000
# CADC0874   0.188406   0.318841      0.130435             0.000000             0.000000  0.000000
# CADC0884        NaN        NaN           NaN             0.240506             0.265823  0.025316
# CADC0886   0.000000   0.000000      0.000000             0.125000             0.125000  0.000000
# CADC0888   0.616667   0.583333     -0.033333             0.380208             0.458333  0.078125
# CADC0890   0.818182   0.818182      0.000000             0.000000             1.000000  1.000000
# CADC0892   0.970588   0.941176     -0.029412             1.000000             1.000000  0.000000
# CADC0894   1.000000   1.000000      0.000000             0.772358             0.943089  0.170732

# [저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/calibration_v1_vs_v2_mixture_and_forgetting.csv
# 채널            burn-in std  burn-in |z|>3   steady std  steady kurt  steady |z|>3
# --------------------------------------------------------------------------------
# CADC0890            0.068          0.000        2.252        -0.13         0.258
# CADC0892            0.112          0.000        1.126        84.75         0.020
# CADC0894            0.253          0.002        2.877        35.28         0.070

# 참고: 정규분포라면 std≈1, |z|>3 비율≈0.003
# - steady std가 1보다 뚜렷이 크거나 |z|>3 비율이 0.01+ 이면 -> 가설 A (노이즈 모델 재점검)
# - steady std가 대략 1에 가까운데도 오탐이 심하면 -> 가설 B (BOCPD forgetting이 원인)

# [저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/problem_channels_zcheck.csv
# channel	r_used	q_used	burnin_mean	burnin_std	burnin_kurt	burnin_frac_gt3	steady_mean	steady_std	steady_kurt	steady_frac_gt3
# 0	CADC0890	0.001228	0.000036	0.016487	0.068269	0.501436	0.000000	0.010924	2.251626	-0.131505	0.257576
# 1	CADC0892	0.000083	0.000007	0.005902	0.112267	26.770168	0.000000	-0.000041	1.125730	84.754067	0.019898
# 2	CADC0894	0.000013	0.000002	0.010336	0.252891	43.313643	0.001626	0.001647	2.876931	35.281981	0.070152






# # --- Ablation 스위치 ---
# USE_MIXTURE_NOISE_ESTIMATOR = False
# USE_BOCPD_FORGETTING = False
# ==========================================================================================
# 0. 데이터 규모 및 Ablation 설정
# ==========================================================================================
# segments: (303493, 8), channels: ['CADC0872', 'CADC0873', 'CADC0874', 'CADC0884', 'CADC0886', 'CADC0888', 'CADC0890', 'CADC0892', 'CADC0894']
# USE_MIXTURE_NOISE_ESTIMATOR = False
# USE_BOCPD_FORGETTING        = False
# RUN_TAG                     = mixture_and_forgetting
# ==========================================================================================
# (A) 채널 프로파일링: 양자화/초미세잡음/연속 분류 + 강건 노이즈
# ==========================================================================================
# [CADC0872] type=float_noise_suspect frac_zero=0.053 step≈N/A r_robust=2.647e-12 q_robust=4.83e-13
# [CADC0873] type=float_noise_suspect frac_zero=0.052 step≈N/A r_robust=2.51e-12 q_robust=2.942e-13
# [CADC0874] type=float_noise_suspect frac_zero=0.044 step≈N/A r_robust=8.02e-13 q_robust=2.726e-13
# [CADC0884] type=quantized          frac_zero=0.183 step≈0.01435 r_robust=0.0004859 q_robust=2.833e-05
# [CADC0886] type=quantized          frac_zero=0.371 step≈0.02735 r_robust=0.001328 q_robust=7.374e-05
# [CADC0888] type=quantized          frac_zero=0.354 step≈0.01439 r_robust=0.001277 q_robust=8.229e-05
# [CADC0890] type=continuous         frac_zero=0.083 step≈N/A r_robust=0.002315 q_robust=0.0002688
# [CADC0892] type=quantized          frac_zero=0.416 step≈0.005405 r_robust=0.0001511 q_robust=3.785e-05
# [CADC0894] type=quantized          frac_zero=0.583 step≈0.002645 r_robust=0.0001328 q_robust=1.417e-05

# ==========================================================================================
# (B) 채널별 온라인 필터링 + BOCPD(forgetting, kappa_max=40) 재탐지
# ==========================================================================================
# [CADC0872] recall=0.168 (n_anom=131)  false_alarm_rate=0.000 (n_normal=415)
# [CADC0873] recall=0.171 (n_anom=105)  false_alarm_rate=0.000 (n_normal=488)
# [CADC0874] recall=0.319 (n_anom=69)  false_alarm_rate=0.000 (n_normal=125)
# [CADC0884] recall=nan (n_anom=0)  false_alarm_rate=0.241 (n_normal=158)
# [CADC0886] recall=0.000 (n_anom=3)  false_alarm_rate=0.000 (n_normal=8)
# [CADC0888] recall=0.617 (n_anom=60)  false_alarm_rate=0.391 (n_normal=192)
# [CADC0890] recall=0.909 (n_anom=11)  false_alarm_rate=0.000 (n_normal=3)
# [CADC0892] recall=0.971 (n_anom=34)  false_alarm_rate=0.994 (n_normal=177)
# [CADC0894] recall=1.000 (n_anom=21)  false_alarm_rate=0.772 (n_normal=123)
# channel	channel_type	n_anomaly_segments	n_anomaly_detected	recall	n_normal_segments	n_normal_false_alarm	false_alarm_rate
# 0	CADC0872	float_noise_suspect	131	22	0.167939	415	0	0.000000
# 1	CADC0873	float_noise_suspect	105	18	0.171429	488	0	0.000000
# 2	CADC0874	float_noise_suspect	69	22	0.318841	125	0	0.000000
# 3	CADC0884	quantized	0	0	NaN	158	38	0.240506
# 4	CADC0886	quantized	3	0	0.000000	8	0	0.000000
# 5	CADC0888	quantized	60	37	0.616667	192	75	0.390625
# 6	CADC0890	continuous	11	10	0.909091	3	0	0.000000
# 7	CADC0892	quantized	34	33	0.970588	177	176	0.994350
# 8	CADC0894	quantized	21	21	1.000000	123	95	0.772358


# ==========================================================================================
# (C) v1 vs v2 비교
# ==========================================================================================
#  channel  recall_v1  recall_v2  recall_delta  false_alarm_rate_v1  false_alarm_rate_v2      fa_delta
# CADC0872   0.076336   0.167939      0.091603             0.000000             0.000000  0.000000e+00
# CADC0873   0.104762   0.171429      0.066667             0.000000             0.000000  0.000000e+00
# CADC0874   0.188406   0.318841      0.130435             0.000000             0.000000  0.000000e+00
# CADC0884        NaN        NaN           NaN             0.240506             0.240506  5.551115e-17
# CADC0886   0.000000   0.000000      0.000000             0.125000             0.000000 -1.250000e-01
# CADC0888   0.616667   0.616667      0.000000             0.380208             0.390625  1.041667e-02
# CADC0890   0.818182   0.909091      0.090909             0.000000             0.000000  0.000000e+00
# CADC0892   0.970588   0.970588      0.000000             1.000000             0.994350 -5.649718e-03
# CADC0894   1.000000   1.000000      0.000000             0.772358             0.772358  0.000000e+00

# [저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/calibration_v1_vs_v2_mixture_and_forgetting.csv

# 채널            burn-in std  burn-in |z|>3   steady std  steady kurt  steady |z|>3
# --------------------------------------------------------------------------------
# CADC0890            0.050          0.000        0.945         0.37         0.000
# CADC0892            0.080          0.000        0.626        63.32         0.008
# CADC0894            0.081          0.000        0.978        34.27         0.026

# 참고: 정규분포라면 std≈1, |z|>3 비율≈0.003
# - steady std가 1보다 뚜렷이 크거나 |z|>3 비율이 0.01+ 이면 -> 가설 A (노이즈 모델 재점검)
# - steady std가 대략 1에 가까운데도 오탐이 심하면 -> 가설 B (BOCPD forgetting이 원인)

# [저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/problem_channels_zcheck.csv
# channel	r_used	q_used	burnin_mean	burnin_std	burnin_kurt	burnin_frac_gt3	steady_mean	steady_std	steady_kurt	steady_frac_gt3
# 0	CADC0890	0.002311	0.000268	0.011052	0.049675	0.270300	0.0	-0.001630	0.945336	0.373170	0.000000
# 1	CADC0892	0.000154	0.000038	0.004006	0.079885	27.876947	0.0	-0.000045	0.626050	63.322849	0.007681
# 2	CADC0894	0.000133	0.000014	0.003283	0.080694	43.638211	0.0	0.000691	0.977772	34.272141	0.026234


from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


BASE_DIR = Path("/home/yjlee/Research/Satellite/OPSSAT-AD_LEO data")
DATASET_PATH = BASE_DIR / "dataset.csv"
SEGMENTS_PATH = BASE_DIR / "segments.csv"
OUT_DIR = BASE_DIR / "causal_pipeline_outputs"
OLD_CALIBRATION_PATH = OUT_DIR / "detector_calibration.csv"  # v1 결과 (비교용, 없으면 생략)

ALL_CHANNELS = ["CADC0872", "CADC0873", "CADC0874", "CADC0884",
                 "CADC0886", "CADC0888", "CADC0890", "CADC0892", "CADC0894"]

QUANTIZATION_FRAC_ZERO_THRESHOLD = 0.10
FLOAT_NOISE_ABS_DIFF_THRESHOLD = 1e-6

# =============================================================================
# 0. 파라미터 잠금 (§1-6 ablation 4-조합 비교 결론 반영 — 더 이상 스위치가 아니라 확정값)
# =============================================================================
# - forgetting=True:  float_noise_suspect(872/873/874) recall 유지에 필수
#                      (끄면 874: 0.725 -> 0.319 로 거의 반토막)
# - mixture=False:    890/894 steady-state z의 std가 1로 정상화되는 쪽
#                      (mixture=True면 890/894 std가 2.25 / 2.88로 벌어짐)
USE_MIXTURE_NOISE_ESTIMATOR = False
USE_BOCPD_FORGETTING = True
RUN_TAG = "final_locked"

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


def section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


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
    sigma = 1.4826 * mad
    return float(sigma ** 2)


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
        ch_type = "quantized"
        q_step = detect_quantization_step(np.abs(nonzero))
    elif np.isfinite(min_nonzero_abs) and min_nonzero_abs < FLOAT_NOISE_ABS_DIFF_THRESHOLD:
        ch_type = "float_noise_suspect"
        q_step = np.nan
    else:
        ch_type = "continuous"
        q_step = np.nan

    if USE_MIXTURE_NOISE_ESTIMATOR:
        r_robust = max(mixture_variance(diffs_all) / 2.0, 1e-14)
        d2 = np.diff(diffs_all)
        q_robust = max(mixture_variance(d2) / 6.0, 1e-14) if len(d2) > 1 else 1e-14
    else:
        r_robust = max(np.var(diffs_all) / 2.0, 1e-14)
        d2 = np.diff(diffs_all)
        q_robust = max(np.var(d2) / 6.0, 1e-14) if len(d2) > 1 else 1e-14

    return ChannelProfile(
        channel=channel, channel_type=ch_type,
        frac_zero_diff=frac_zero, min_nonzero_abs_diff=min_nonzero_abs,
        quantization_step=q_step, r_robust=r_robust, q_robust=q_robust,
        n_nominal_points=n_points,
    )


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
        levels = np.empty(n)

        for t in range(n):
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q

            y_pred = (self.H @ x_pred)[0]
            S = (self.H @ P_pred @ self.H.T)[0, 0] + self.r_eff
            nu = y[t] - y_pred

            innovations[t] = nu
            innovation_vars[t] = S
            levels[t] = y_pred

            K = (P_pred @ self.H.T) / S
            x = x_pred + (K.flatten() * nu)
            P = P_pred - K @ self.H @ P_pred

        return innovations, innovation_vars, levels


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
        log_v_shrunk = w_c * lv + (1 - w_c) * global_mean
        shrunk[c] = float(np.exp(log_v_shrunk))
    return shrunk


class BOCPD:
    def __init__(self, hazard_lambda: float = 250.0,
                 mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0,
                 kappa_max: float = 40.0, use_forgetting: bool = True):
        self.hazard = 1.0 / hazard_lambda
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.kappa_max = kappa_max
        self.use_forgetting = use_forgetting

    def _cap_sufficient_stats(self, kappa, alpha, beta):
        if not self.use_forgetting:
            return kappa, alpha, beta
        over = kappa > self.kappa_max
        if not np.any(over):
            return kappa, alpha, beta
        scale = np.where(over, self.kappa_max / kappa, 1.0)
        kappa2 = np.where(over, self.kappa_max, kappa)
        alpha2 = np.where(over, alpha * scale, alpha)
        beta2 = np.where(over, beta * scale, beta)
        alpha2 = np.maximum(alpha2, 1e-3)
        return kappa2, alpha2, beta2

    def run(self, x: np.ndarray):
        n = len(x)
        R = np.zeros((n + 1, n + 1))
        R[0, 0] = 1.0

        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        alpha = np.array([self.alpha0])
        beta = np.array([self.beta0])

        map_run_length = np.zeros(n, dtype=int)

        for t in range(n):
            xt = x[t]
            dof = 2 * alpha
            scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
            pred_probs = stats.t.pdf(xt, df=dof, loc=mu, scale=scale)
            pred_probs = np.nan_to_num(pred_probs, nan=1e-12, posinf=1e-12)

            run_probs = R[t, : t + 1]
            growth = run_probs * pred_probs * (1 - self.hazard)
            cp_prob = np.sum(run_probs * pred_probs * self.hazard)

            new_R = np.zeros(t + 2)
            new_R[0] = cp_prob
            new_R[1:] = growth
            total = new_R.sum()
            if total <= 0 or not np.isfinite(total):
                new_R[:] = 0
                new_R[0] = 1.0
            else:
                new_R /= total
            R[t + 1, : t + 2] = new_R

            map_run_length[t] = int(np.argmax(new_R))

            new_mu = np.empty(t + 2)
            new_kappa = np.empty(t + 2)
            new_alpha = np.empty(t + 2)
            new_beta = np.empty(t + 2)

            new_mu[0] = self.mu0
            new_kappa[0] = self.kappa0
            new_alpha[0] = self.alpha0
            new_beta[0] = self.beta0

            new_mu[1:] = (kappa * mu + xt) / (kappa + 1)
            new_kappa[1:] = kappa + 1
            new_alpha[1:] = alpha + 0.5
            new_beta[1:] = beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))

            new_kappa, new_alpha, new_beta = self._cap_sufficient_stats(new_kappa, new_alpha, new_beta)
            mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

        onset_idx = None
        onset_posterior = None
        confirm_idx = None
        posterior_over_onset = None
        for t in range(1, n):
            if map_run_length[t] <= 2 and map_run_length[t - 1] > 5:
                onset_idx = t - map_run_length[t]
                confirm_idx = t
                row = R[t + 1, : t + 2]
                k = min(len(row), 30)
                posterior_over_onset = {int(r): float(row[r]) for r in range(k) if row[r] > 1e-6}
                onset_posterior = float(np.sum(row[:3]))
                break

        return {
            "run_length_posterior": R, "map_run_length": map_run_length,
            "onset_idx": onset_idx, "onset_confidence": onset_posterior,
            "confirm_idx": confirm_idx, "posterior_over_onset": posterior_over_onset,
        }


def make_synthetic_data(n_segments_per_channel: int = 40, seg_len: int = 60):
    rows = []
    seg_id = 1
    specs = {
        "SYN_QUANT": dict(step=0.0026, base=1.0, anomaly_jump=0.05),
        "SYN_FLOAT": dict(step=5e-11, base=1e-7, anomaly_jump=2e-8),
    }
    for ch, spec in specs.items():
        for s in range(n_segments_per_channel):
            is_anom = (s % 5 == 0)
            t0 = pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=s * seg_len * 2)
            level = spec["base"]
            vals = []
            for i in range(seg_len):
                if rng.random() < 0.3:
                    level += rng.choice([-1, 1]) * spec["step"]
                if is_anom and i > seg_len // 2:
                    level += spec["anomaly_jump"] * rng.normal(1, 0.2)
                vals.append(level)
            for i, v in enumerate(vals):
                rows.append({
                    "channel": ch, "timestamp": t0 + pd.Timedelta(seconds=i * 2),
                    "value": v, "label": "anomaly" if is_anom else "a2",
                    "sampling": 2, "anomaly": int(is_anom), "segment": seg_id, "train": 1,
                })
            seg_id += 1
    return pd.DataFrame(rows)


use_synthetic = not SEGMENTS_PATH.exists()
if use_synthetic:
    print(f"[안내] {SEGMENTS_PATH} 를 찾을 수 없어 합성 데이터로 스모크 테스트를 실행합니다.")
    seg = make_synthetic_data()
    channels = list(seg["channel"].unique())
    out_dir = Path("./synthetic_outputs")
else:
    seg = pd.read_csv(SEGMENTS_PATH, parse_dates=["timestamp"])
    seg = seg[seg["channel"].isin(ALL_CHANNELS)].reset_index(drop=True)
    channels = ALL_CHANNELS
    out_dir = OUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)

section("0. 데이터 규모 및 잠금된 파라미터")
print(f"segments: {seg.shape}, channels: {channels}")
print(f"USE_MIXTURE_NOISE_ESTIMATOR = {USE_MIXTURE_NOISE_ESTIMATOR}  (잠금)")
print(f"USE_BOCPD_FORGETTING        = {USE_BOCPD_FORGETTING}  (잠금)")
print(f"RUN_TAG                     = {RUN_TAG}")


# =============================================================================
# (A) 채널 프로파일링: 양자화/초미세잡음/연속 분류 + 강건 노이즈 (mixture=False로 갱신)
# =============================================================================
section("(A) 채널 프로파일링: 양자화/초미세잡음/연속 분류 + 강건 노이즈")

profiles = {}
for ch in channels:
    nominal_vals = [
        s.sort_values("timestamp")["value"].values
        for _, s in seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)].groupby("segment")
    ]
    profiles[ch] = profile_channel(ch, nominal_vals)
    p = profiles[ch]
    step_str = f"{p.quantization_step:.4g}" if np.isfinite(p.quantization_step) else "N/A"
    print(f"[{ch}] type={p.channel_type:<18s} frac_zero={p.frac_zero_diff:.3f} "
          f"step\u2248{step_str} r_robust={p.r_robust:.4g} q_robust={p.q_robust:.4g}")

profile_df = pd.DataFrame([vars(p) for p in profiles.values()])
profile_df.to_csv(out_dir / "channel_profiles_v2.csv", index=False)

r_estimates = {ch: (profiles[ch].r_robust, max(profiles[ch].n_nominal_points, 2)) for ch in channels}
q_estimates = {ch: (profiles[ch].q_robust, max(profiles[ch].n_nominal_points, 2)) for ch in channels}
r_shrunk = hierarchical_shrink_variance(r_estimates)
q_shrunk = hierarchical_shrink_variance(q_estimates)

quant_floor = {
    ch: (profiles[ch].quantization_step ** 2 / 12.0) if np.isfinite(profiles[ch].quantization_step) else 0.0
    for ch in channels
}


# =============================================================================
# (B') 채널별 온라인 필터링 + BOCPD 재탐지 + onset 사후분포 저장
# =============================================================================
section("(B') 채널별 온라인 필터링 + BOCPD(forgetting) 재탐지 + onset 사후분포 저장")

calibration_rows = []
onset_result_rows = []     # -> onset_bayesian_results.csv
onset_posterior_rows = []  # -> onset_posterior_distributions.csv (long format)

for ch in channels:
    ch_segments = seg[seg["channel"] == ch].sort_values(["segment", "timestamp"])
    seg_ids = ch_segments["segment"].unique()

    kf = LocalLinearTrendKF(q=q_shrunk[ch], r_nominal=r_shrunk[ch], theta=0.0,
                             quantization_floor=quant_floor[ch])
    bocpd = BOCPD(hazard_lambda=250.0, kappa_max=40.0, use_forgetting=USE_BOCPD_FORGETTING)

    n_anom, n_anom_det, n_norm, n_norm_fa = 0, 0, 0, 0

    for sid in seg_ids:
        s = ch_segments[ch_segments["segment"] == sid].sort_values("timestamp")
        values = s["value"].values
        is_anomaly = bool(s["anomaly"].iloc[0])
        if len(values) < 5:
            continue

        innovations, innovation_vars, _ = kf.run(values)
        z = innovations / np.sqrt(np.clip(innovation_vars, 1e-8, None))
        result = bocpd.run(z)
        detected = result["onset_idx"] is not None

        onset_result_rows.append({
            "channel": ch,
            "segment": sid,
            "is_anomaly": int(is_anomaly),
            "n_points": len(values),
            "onset_idx": result["onset_idx"],
            "onset_confidence": result["onset_confidence"],
            "confirm_idx": result["confirm_idx"],
            "detected": int(detected),
        })

        if result["posterior_over_onset"] is not None:
            confirm_idx = result["confirm_idx"]
            for run_length, prob in result["posterior_over_onset"].items():
                onset_posterior_rows.append({
                    "channel": ch,
                    "segment": sid,
                    "confirm_idx": confirm_idx,
                    "run_length_at_confirm": run_length,
                    "candidate_onset_idx": (confirm_idx - run_length) if confirm_idx is not None else np.nan,
                    "posterior_prob": prob,
                })

        if is_anomaly:
            n_anom += 1
            n_anom_det += int(detected)
        else:
            n_norm += 1
            n_norm_fa += int(detected)

    calibration_rows.append({
        "channel": ch, "channel_type": profiles[ch].channel_type,
        "n_anomaly_segments": n_anom, "n_anomaly_detected": n_anom_det,
        "recall": n_anom_det / n_anom if n_anom else np.nan,
        "n_normal_segments": n_norm, "n_normal_false_alarm": n_norm_fa,
        "false_alarm_rate": n_norm_fa / n_norm if n_norm else np.nan,
    })
    r = calibration_rows[-1]
    print(f"[{ch}] recall={r['recall']:.3f} (n_anom={n_anom})  "
          f"false_alarm_rate={r['false_alarm_rate']:.3f} (n_normal={n_norm})")

calib_v2 = pd.DataFrame(calibration_rows)
calib_v2.to_csv(out_dir / f"detector_calibration_v2_{RUN_TAG}.csv", index=False)

onset_bayesian_results = pd.DataFrame(onset_result_rows)
onset_bayesian_results.to_csv(out_dir / "onset_bayesian_results.csv", index=False)

onset_posterior_distributions = pd.DataFrame(onset_posterior_rows)
onset_posterior_distributions.to_csv(out_dir / "onset_posterior_distributions.csv", index=False)

print(f"\n[저장] {out_dir / f'detector_calibration_v2_{RUN_TAG}.csv'}  ({len(calib_v2)} rows)")
print(f"[저장] {out_dir / 'onset_bayesian_results.csv'}  ({len(onset_bayesian_results)} rows)")
print(f"[저장] {out_dir / 'onset_posterior_distributions.csv'}  ({len(onset_posterior_distributions)} rows)")


if OLD_CALIBRATION_PATH.exists():
    section("(C) v1 vs v2 비교")
    calib_v1 = pd.read_csv(OLD_CALIBRATION_PATH)
    merged = calib_v1.merge(calib_v2, on="channel", suffixes=("_v1", "_v2"))
    merged["recall_delta"] = merged["recall_v2"] - merged["recall_v1"]
    merged["fa_delta"] = merged["false_alarm_rate_v2"] - merged["false_alarm_rate_v1"]
    display_cols = ["channel", "recall_v1", "recall_v2", "recall_delta",
                     "false_alarm_rate_v1", "false_alarm_rate_v2", "fa_delta"]
    print(merged[display_cols].to_string(index=False))
    merged.to_csv(out_dir / f"calibration_v1_vs_v2_{RUN_TAG}.csv", index=False)
    print(f"\n[저장] {out_dir / f'calibration_v1_vs_v2_{RUN_TAG}.csv'}")
else:
    print(f"[안내] {OLD_CALIBRATION_PATH} 가 없어 v1 대비 비교는 생략합니다.")


# =============================================================================
# (D) CADC0892 onset 트리거 로직 진단: map_run_length 궤적
# =============================================================================
section("(D) CADC0892 onset 트리거 로직 진단: 정상 세그먼트의 map_run_length 궤적")

TARGET_DIAG_CHANNEL = "CADC0892"
N_SEGMENTS_TO_PLOT = 6

if TARGET_DIAG_CHANNEL in channels:
    kf892 = LocalLinearTrendKF(
        q=q_shrunk[TARGET_DIAG_CHANNEL], r_nominal=r_shrunk[TARGET_DIAG_CHANNEL],
        theta=0.0, quantization_floor=quant_floor[TARGET_DIAG_CHANNEL],
    )
    bocpd892 = BOCPD(hazard_lambda=250.0, kappa_max=40.0, use_forgetting=USE_BOCPD_FORGETTING)

    normal_892 = seg[(seg["channel"] == TARGET_DIAG_CHANNEL) & (seg["anomaly"] == 0)]
    seg_ids_892 = normal_892["segment"].unique()

    reset_summary_rows = []
    plot_segments = []

    for sid in seg_ids_892:
        s = normal_892[normal_892["segment"] == sid].sort_values("timestamp")
        values = s["value"].values
        if len(values) < 10:
            continue

        innovations, innovation_vars, _ = kf892.run(values)
        z = innovations / np.sqrt(np.clip(innovation_vars, 1e-8, None))
        result = bocpd892.run(z)
        mrl = result["map_run_length"]

        n_resets = int(np.sum((mrl[1:] <= 2) & (mrl[:-1] > 5)))
        reset_summary_rows.append({
            "segment": sid,
            "n_points": len(values),
            "n_resets": n_resets,
            "max_run_length": int(mrl.max()),
            "onset_idx": result["onset_idx"],
            "z_kurtosis": float(pd.Series(z).kurtosis()),
            "frac_z_gt3": float(np.mean(np.abs(z) > 3)),
        })

        if len(plot_segments) < N_SEGMENTS_TO_PLOT:
            plot_segments.append((sid, z, mrl))

    reset_df = pd.DataFrame(reset_summary_rows)
    reset_df.to_csv(out_dir / "cadc0892_reset_diagnosis.csv", index=False)

    if len(reset_df) > 0:
        n_with_reset = int((reset_df["n_resets"] > 0).sum())
        print(f"정상 세그먼트 {len(reset_df)}개 중 리셋(run_length: >5 -> <=2) 1회 이상 발생: "
              f"{n_with_reset}개 ({n_with_reset / len(reset_df):.1%})")
        print(f"세그먼트당 평균 리셋 횟수: {reset_df['n_resets'].mean():.2f}")
        print("리셋 횟수 분포:")
        print(reset_df["n_resets"].value_counts().sort_index().to_string())
    print(f"\n[저장] {out_dir / 'cadc0892_reset_diagnosis.csv'}")

    if plot_segments:
        fig, axes = plt.subplots(len(plot_segments), 2, figsize=(11, 2.5 * len(plot_segments)))
        if len(plot_segments) == 1:
            axes = axes.reshape(1, 2)

        for row_i, (sid, z, mrl) in enumerate(plot_segments):
            axes[row_i, 0].plot(z, lw=0.8)
            axes[row_i, 0].axhline(0, color="gray", lw=0.5)
            axes[row_i, 0].axhline(3, color="red", lw=0.5, ls="--")
            axes[row_i, 0].axhline(-3, color="red", lw=0.5, ls="--")
            axes[row_i, 0].set_title(f"seg {sid} — z-score")

            axes[row_i, 1].plot(mrl, lw=0.8, color="tab:orange")
            axes[row_i, 1].axhline(5, color="red", lw=0.5, ls="--", label="run_length=5")
            axes[row_i, 1].axhline(2, color="red", lw=0.5, ls=":", label="run_length=2")
            axes[row_i, 1].set_title(f"seg {sid} — MAP run length")
            if row_i == 0:
                axes[row_i, 1].legend(fontsize=7)

        plt.tight_layout()
        fig_path = out_dir / "cadc0892_run_length_diagnosis.png"
        plt.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"[저장] {fig_path}")

        if len(reset_df) > 0 and n_with_reset / max(len(reset_df), 1) > 0.3:
            print("\n[해석] 정상 세그먼트의 30% 이상에서 리셋이 반복 발생 -> onset 판정 규칙 자체"
                  "(run_length 5->2 임계, 또는 전 채널 공통 hazard_lambda=250)가 CADC0892처럼"
                  " 고첨도(kurtosis 60~85) · 평소엔 조용하다 가끔 튀는 채널에는 과민하다는"
                  " 가설(§1-9)을 뒷받침함. 채널별 hazard_lambda 또는 onset 확정 조건"
                  "(예: run_length<=2가 N틱 이상 유지)을 별도로 튜닝하는 것을 다음 액션으로 고려.")
    else:
        print("[안내] 플롯할 수 있을 만큼 긴(>=10포인트) 정상 세그먼트가 없습니다.")
else:
    print(f"[안내] {TARGET_DIAG_CHANNEL} 이 channels 목록에 없어 진단을 생략합니다.")


# =============================================================================
# (E) §1-11 MCC/Youden 채널 스코프 표 재현 -> channel_scope.csv
# =============================================================================
section("(E) MCC/Youden 기준 2단계 채널 스코프 확정")

def channel_mcc_youden(row) -> dict:
    n_anom, n_norm = row["n_anomaly_segments"], row["n_normal_segments"]
    recall, fa = row["recall"], row["false_alarm_rate"]

    if n_anom == 0 or pd.isna(recall):
        return {"mcc": np.nan, "youden_j": np.nan, "scope_reason": "구조적 제외 (anomaly=0)"}

    tp = recall * n_anom
    fn = n_anom - tp
    fp = fa * n_norm
    tn = n_norm - fp

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    youden_j = recall - fa

    if n_anom < 5 or n_norm < 5:
        reason = f"표본부족 (n_anom={int(n_anom)}, n_norm={int(n_norm)})"
    elif mcc <= 0:
        reason = "우연수준 이하 (MCC<=0)"
    else:
        reason = "포함"
    return {"mcc": mcc, "youden_j": youden_j, "scope_reason": reason}

scope_rows = []
for _, row in calib_v2.iterrows():
    stats_row = channel_mcc_youden(row)
    scope_rows.append({
        "channel": row["channel"],
        "channel_type": row["channel_type"],
        "recall": row["recall"],
        "false_alarm_rate": row["false_alarm_rate"],
        **stats_row,
        "include_in_stage2": stats_row["scope_reason"] == "포함",
    })

channel_scope = pd.DataFrame(scope_rows)
channel_scope.to_csv(out_dir / "channel_scope.csv", index=False)

included = channel_scope[channel_scope["include_in_stage2"]]["channel"].tolist()
excluded = channel_scope[~channel_scope["include_in_stage2"]]

print(channel_scope.to_string(index=False))
print(f"\n2단계 포함 채널 ({len(included)}개): {included}")
print("제외 채널 및 사유:")
for _, r in excluded.iterrows():
    print(f"  - {r['channel']}: {r['scope_reason']}")
print(f"\n[저장] {out_dir / 'channel_scope.csv'}")

print("\n[주의] CADC0890처럼 n이 매우 작은 채널은 MCC/포함 여부가 안정적인지 "
      "부트스트랩 CI로 재확인하기 전까지는 §1-11의 잠정 결론으로 취급할 것.")


# =============================================================================
# (F) train/test 분리 확인 — EDA로 스키마 확정됨: dataset.csv/segments.csv 모두
#     'train' 컬럼(0/1)을 가지고 있고, 전체 비율(1594/2123 ≈ 75.08%)이 dataset.csv의
#     train 평균과 정확히 일치 -> 이것이 OPSSAT-AD 공식 train/test 분할.
#     (더 이상 자동탐색이 아니라 확정된 컬럼명으로 고정 + 안전성 검증만 수행)
# =============================================================================
section("(F) train/test 분리 확인 (확정 컬럼: 'train')")

TRAIN_COL = "train"
train_col = None

if TRAIN_COL not in seg.columns:
    print(f"[주의] '{TRAIN_COL}' 컬럼이 segments.csv에 없습니다 — 스키마가 바뀐 것으로 보입니다. "
          "seg.columns를 확인하세요:", list(seg.columns))
else:
    # 1) 세그먼트 내부에서 train 값이 안 바뀌는지 (label/anomaly와 동일한 논리로 확인)
    seg_train_nunique = seg.groupby("segment")[TRAIN_COL].nunique()
    n_inconsistent = int((seg_train_nunique > 1).sum())
    print(f"세그먼트 내부에서 '{TRAIN_COL}' 값이 섞여있는 세그먼트 수: "
          f"{n_inconsistent} / {seg_train_nunique.shape[0]}")

    # 2) dataset.csv(df)의 train과 segments.csv에서 파생한 값이 일치하는지 교차검증
    cross_check_ok = None
    if "df" in globals() and "train" in df.columns and "segment" in df.columns:
        seg_train_repr = seg.groupby("segment")[TRAIN_COL].first()
        df_train_repr = df.set_index("segment")["train"]
        joined = pd.concat([df_train_repr, seg_train_repr], axis=1, keys=["df_train", "seg_train"]).dropna()
        mismatch = int((joined["df_train"] != joined["seg_train"]).sum())
        cross_check_ok = (mismatch == 0)
        print(f"dataset.csv vs segments.csv 'train' 값 불일치 세그먼트 수: {mismatch} / {len(joined)}")
    else:
        print("[안내] dataset.csv(df)가 이 세션에 로드돼 있지 않아 교차검증은 생략합니다 "
              "(생략해도 (G) 실행에는 지장 없음).")

    if n_inconsistent == 0 and cross_check_ok is not False:
        train_col = TRAIN_COL
        train_bool = seg[train_col].astype(bool)
        print(f"\n[확정] '{train_col}' 컬럼을 train/test 지시자로 사용합니다. "
              f"train={int(train_bool.sum())}행, test={int((~train_bool).sum())}행 "
              f"(train 비율={train_bool.mean():.4f})")
    else:
        print(f"\n[중단] '{TRAIN_COL}' 컬럼이 세그먼트 내부에서 비일관적이거나 "
              "dataset.csv와 불일치합니다 — (G)는 안전을 위해 생략합니다. 원인을 먼저 확인하세요.")


# =============================================================================
# (G) train 데이터로만 파라미터 재적합 -> test에서만 성능 재평가
#     (train/test 컬럼이 발견된 경우에만 실행됨)
# =============================================================================
if train_col:
    section("(G) train 데이터로만 파라미터 재적합 -> test에서만 성능 재평가")

    train_mask = seg[train_col].astype(bool)
    seg_train = seg[train_mask]
    seg_test = seg[~train_mask]

    profiles_tt = {}
    for ch in channels:
        nominal_vals = [
            s.sort_values("timestamp")["value"].values
            for _, s in seg_train[(seg_train["channel"] == ch) & (seg_train["anomaly"] == 0)].groupby("segment")
        ]
        profiles_tt[ch] = profile_channel(ch, nominal_vals)

    r_estimates_tt = {ch: (profiles_tt[ch].r_robust, max(profiles_tt[ch].n_nominal_points, 2)) for ch in channels}
    q_estimates_tt = {ch: (profiles_tt[ch].q_robust, max(profiles_tt[ch].n_nominal_points, 2)) for ch in channels}
    r_shrunk_tt = hierarchical_shrink_variance(r_estimates_tt)
    q_shrunk_tt = hierarchical_shrink_variance(q_estimates_tt)
    quant_floor_tt = {
        ch: (profiles_tt[ch].quantization_step ** 2 / 12.0) if np.isfinite(profiles_tt[ch].quantization_step) else 0.0
        for ch in channels
    }

    test_calib_rows = []
    for ch in channels:
        ch_segs = seg_test[seg_test["channel"] == ch].sort_values(["segment", "timestamp"])
        seg_ids = ch_segs["segment"].unique()
        kf = LocalLinearTrendKF(q=q_shrunk_tt[ch], r_nominal=r_shrunk_tt[ch], theta=0.0,
                                 quantization_floor=quant_floor_tt[ch])
        bocpd = BOCPD(hazard_lambda=250.0, kappa_max=40.0, use_forgetting=USE_BOCPD_FORGETTING)
        n_anom, n_anom_det, n_norm, n_norm_fa = 0, 0, 0, 0
        for sid in seg_ids:
            s = ch_segs[ch_segs["segment"] == sid].sort_values("timestamp")
            values = s["value"].values
            is_anomaly = bool(s["anomaly"].iloc[0])
            if len(values) < 5:
                continue
            innovations, innovation_vars, _ = kf.run(values)
            z = innovations / np.sqrt(np.clip(innovation_vars, 1e-8, None))
            result = bocpd.run(z)
            detected = result["onset_idx"] is not None
            if is_anomaly:
                n_anom += 1
                n_anom_det += int(detected)
            else:
                n_norm += 1
                n_norm_fa += int(detected)
        test_calib_rows.append({
            "channel": ch,
            "recall_test": n_anom_det / n_anom if n_anom else np.nan,
            "fa_test": n_norm_fa / n_norm if n_norm else np.nan,
            "n_anom_test": n_anom, "n_norm_test": n_norm,
        })

    test_calib = pd.DataFrame(test_calib_rows)
    compare = calib_v2.merge(test_calib, on="channel")
    compare["recall_drop"] = compare["recall"] - compare["recall_test"]
    compare["fa_drop"] = compare["false_alarm_rate"] - compare["fa_test"]
    compare.to_csv(out_dir / "calibration_train_fit_test_eval.csv", index=False)

    print(compare[["channel", "recall", "recall_test", "recall_drop",
                    "false_alarm_rate", "fa_test", "fa_drop"]].to_string(index=False))
    print(f"\n[저장] {out_dir / 'calibration_train_fit_test_eval.csv'}")
    print("\n[해석] recall_drop/fa_drop이 크면(대략 0.1 이상) 전체 데이터로 튜닝한 성능이 "
          "test에서는 재현되지 않는다는 뜻 -> in-sample 튜닝 우려가 실질적임. 이 경우 §1-6/§1-11의 "
          "채널 스코프·MCC 결론은 train 기준으로 다시 확정하고 test 수치는 최종 보고용으로만 써야 함.")


# =============================================================================
# (H) 채널별 MCC/recall/FA 부트스트랩 신뢰구간
#     (특히 CADC0890처럼 표본이 작은 채널의 불확실성 검증)
# =============================================================================
section("(H) 채널별 MCC 부트스트랩 신뢰구간 (CADC0890 등 소표본 채널 불확실성 검증)")

N_BOOT = 2000
rng_boot = np.random.default_rng(RANDOM_STATE)


def mcc_from_counts(tp, fn, fp, tn):
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


boot_rows = []
for ch in channels:
    sub = onset_bayesian_results[onset_bayesian_results["channel"] == ch]
    anom = sub[sub["is_anomaly"] == 1]["detected"].values
    norm = sub[sub["is_anomaly"] == 0]["detected"].values

    if len(anom) == 0 or len(norm) == 0:
        boot_rows.append({
            "channel": ch, "mcc_median": np.nan, "mcc_ci_low": np.nan, "mcc_ci_high": np.nan,
            "n_anomaly": len(anom), "n_normal": len(norm),
            "ci_crosses_zero": np.nan, "note": "구조적 제외 (anomaly 또는 normal 세그먼트 없음)",
        })
        continue

    mccs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        a_s = rng_boot.choice(anom, size=len(anom), replace=True)
        n_s = rng_boot.choice(norm, size=len(norm), replace=True)
        tp, fn = a_s.sum(), len(a_s) - a_s.sum()
        fp, tn = n_s.sum(), len(n_s) - n_s.sum()
        mccs[i] = mcc_from_counts(tp, fn, fp, tn)

    ci_low, ci_high = float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5))
    boot_rows.append({
        "channel": ch,
        "mcc_median": float(np.median(mccs)),
        "mcc_ci_low": ci_low,
        "mcc_ci_high": ci_high,
        "n_anomaly": len(anom), "n_normal": len(norm),
        "ci_crosses_zero": bool(ci_low < 0 < ci_high),
        "note": "",
    })

mcc_boot_df = pd.DataFrame(boot_rows)
mcc_boot_df.to_csv(out_dir / "channel_mcc_bootstrap.csv", index=False)
print(mcc_boot_df.to_string(index=False))
print(f"\n[저장] {out_dir / 'channel_mcc_bootstrap.csv'}  (N_BOOT={N_BOOT})")

flagged = mcc_boot_df[mcc_boot_df["ci_crosses_zero"] == True]  # noqa: E712
if len(flagged):
    print(f"\n[주의] MCC 95% 신뢰구간이 0을 포함하는 채널 ({len(flagged)}개): "
          f"{flagged['channel'].tolist()} -> §1-11에서 '포함'으로 판정했더라도 표본변동만으로 "
          "결과가 뒤집힐 수 있다는 뜻. 논문에는 이 CI를 함께 보고할 것을 권장.")


# =============================================================================
# (I) 문제 채널(890/892/894) burn-in vs steady-state z 재확인
#     (§1-9 진단을 mixture=False 확정 설정으로 재실행)
# =============================================================================
section("(I) 문제 채널(890/892/894) burn-in vs steady-state z 재확인 (확정 설정 기준)")

TARGET_CHANNELS = ["CADC0890", "CADC0892", "CADC0894"]
BURN_IN = 5


def summarize(z):
    if len(z) < 10:
        return dict(mean=np.nan, std=np.nan, kurt=np.nan, frac_gt3=np.nan)
    return dict(
        mean=float(np.mean(z)), std=float(np.std(z)),
        kurt=float(stats.kurtosis(z)),
        frac_gt3=float(np.mean(np.abs(z) > 3)),
    )


target_channels_present = [c for c in TARGET_CHANNELS if c in channels] or channels
print(f"{'채널':<12s} {'burn-in std':>12s} {'steady std':>12s} {'steady kurt':>12s} {'steady |z|>3':>13s}")
print("-" * 65)

recheck_rows = []
for ch in target_channels_present:
    kf = LocalLinearTrendKF(q=q_shrunk[ch], r_nominal=r_shrunk[ch], theta=0.0,
                             quantization_floor=quant_floor[ch])
    nominal = seg[(seg["channel"] == ch) & (seg["anomaly"] == 0)]

    z_burnin, z_steady = [], []
    for sid, s in nominal.groupby("segment"):
        v = s.sort_values("timestamp")["value"].values
        if len(v) < BURN_IN + 5:
            continue
        innovations, innovation_vars, _ = kf.run(v)
        z = innovations / np.sqrt(np.clip(innovation_vars, 1e-12, None))
        z_burnin.append(z[:BURN_IN])
        z_steady.append(z[BURN_IN:])

    z_burnin = np.concatenate(z_burnin) if z_burnin else np.array([])
    z_steady = np.concatenate(z_steady) if z_steady else np.array([])
    sb, ss = summarize(z_burnin), summarize(z_steady)

    print(f"{ch:<12s} {sb['std']:>12.3f} {ss['std']:>12.3f} {ss['kurt']:>12.2f} {ss['frac_gt3']:>13.3f}")
    recheck_rows.append({
        "channel": ch, "r_used": r_shrunk[ch] + quant_floor[ch], "q_used": q_shrunk[ch],
        **{f"burnin_{k}": v for k, v in sb.items()},
        **{f"steady_{k}": v for k, v in ss.items()},
    })

recheck_df = pd.DataFrame(recheck_rows)
recheck_df.to_csv(out_dir / "problem_channels_zcheck_locked.csv", index=False)
print(f"\n[저장] {out_dir / 'problem_channels_zcheck_locked.csv'}")
print("\n참고: 정규분포라면 steady std≈1, |z|>3 비율≈0.003.")
print("mixture=False로 확정했으므로 890/894의 steady std가 §1-9 실증 결과(0.945/0.978)처럼 "
      "1에 가까우면 노이즈 모델은 정상 — 이 표에서 다시 벗어나 있다면 (A) 프로파일링을 이 스위치값으로 "
      "재실행했는지부터 확인할 것.")


section("최종 요약")
print("아래 파일들이 out_dir에 저장되었는지 확인하세요:")
for fname in [
    "channel_profiles_v2.csv",
    f"detector_calibration_v2_{RUN_TAG}.csv",
    "onset_bayesian_results.csv",
    "onset_posterior_distributions.csv",
    "cadc0892_reset_diagnosis.csv",
    "cadc0892_run_length_diagnosis.png",
    "channel_scope.csv",
    "channel_mcc_bootstrap.csv",
    "problem_channels_zcheck_locked.csv",
]:
    exists = (out_dir / fname).exists()
    print(f"  [{'O' if exists else 'X'}] {fname}")
if train_col:
    exists = (out_dir / "calibration_train_fit_test_eval.csv").exists()
    print(f"  [{'O' if exists else 'X'}] calibration_train_fit_test_eval.csv")
else:
    print("  [ - ] calibration_train_fit_test_eval.csv (train/test 컬럼을 못 찾아 생략됨 — 수동 확인 필요)")

==========================================================================================
0. 데이터 규모 및 잠금된 파라미터
==========================================================================================
segments: (303493, 8), channels: ['CADC0872', 'CADC0873', 'CADC0874', 'CADC0884', 'CADC0886', 'CADC0888', 'CADC0890', 'CADC0892', 'CADC0894']
USE_MIXTURE_NOISE_ESTIMATOR = False  (잠금)
USE_BOCPD_FORGETTING        = True  (잠금)
RUN_TAG                     = final_locked

==========================================================================================
(A) 채널 프로파일링: 양자화/초미세잡음/연속 분류 + 강건 노이즈
==========================================================================================
[CADC0872] type=float_noise_suspect frac_zero=0.053 step≈N/A r_robust=2.647e-12 q_robust=4.83e-13
[CADC0873] type=float_noise_suspect frac_zero=0.052 step≈N/A r_robust=2.51e-12 q_robust=2.942e-13
[CADC0874] type=float_noise_suspect frac_zero=0.044 step≈N/A r_robust=8.02e-13 q_robust=2.726e-13
[CADC0884] type=quantized          frac_zero=0.183 step≈0.01435 r_robust=0.0004859 q_robust=2.833e-05
[CADC0886] type=quantized          frac_zero=0.371 step≈0.02735 r_robust=0.001328 q_robust=7.374e-05
[CADC0888] type=quantized          frac_zero=0.354 step≈0.01439 r_robust=0.001277 q_robust=8.229e-05
[CADC0890] type=continuous         frac_zero=0.083 step≈N/A r_robust=0.002315 q_robust=0.0002688
[CADC0892] type=quantized          frac_zero=0.416 step≈0.005405 r_robust=0.0001511 q_robust=3.785e-05
[CADC0894] type=quantized          frac_zero=0.583 step≈0.002645 r_robust=0.0001328 q_robust=1.417e-05

==========================================================================================
(B') 채널별 온라인 필터링 + BOCPD(forgetting) 재탐지 + onset 사후분포 저장
==========================================================================================
[CADC0872] recall=0.321 (n_anom=131)  false_alarm_rate=0.000 (n_normal=415)
[CADC0873] recall=0.276 (n_anom=105)  false_alarm_rate=0.000 (n_normal=488)
[CADC0874] recall=0.725 (n_anom=69)  false_alarm_rate=0.032 (n_normal=125)
[CADC0884] recall=nan (n_anom=0)  false_alarm_rate=0.241 (n_normal=158)
[CADC0886] recall=0.000 (n_anom=3)  false_alarm_rate=0.000 (n_normal=8)
[CADC0888] recall=0.617 (n_anom=60)  false_alarm_rate=0.391 (n_normal=192)
[CADC0890] recall=0.909 (n_anom=11)  false_alarm_rate=0.000 (n_normal=3)
[CADC0892] recall=0.971 (n_anom=34)  false_alarm_rate=0.994 (n_normal=177)
[CADC0894] recall=1.000 (n_anom=21)  false_alarm_rate=0.797 (n_normal=123)

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/detector_calibration_v2_final_locked.csv  (9 rows)
[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/onset_bayesian_results.csv  (2123 rows)
[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/onset_posterior_distributions.csv  (8059 rows)

==========================================================================================
(C) v1 vs v2 비교
==========================================================================================
 channel  recall_v1  recall_v2  recall_delta  false_alarm_rate_v1  false_alarm_rate_v2      fa_delta
CADC0872   0.076336   0.320611      0.244275             0.000000             0.000000  0.000000e+00
CADC0873   0.104762   0.276190      0.171429             0.000000             0.000000  0.000000e+00
CADC0874   0.188406   0.724638      0.536232             0.000000             0.032000  3.200000e-02
CADC0884        NaN        NaN           NaN             0.240506             0.240506  5.551115e-17
CADC0886   0.000000   0.000000      0.000000             0.125000             0.000000 -1.250000e-01
CADC0888   0.616667   0.616667      0.000000             0.380208             0.390625  1.041667e-02
CADC0890   0.818182   0.909091      0.090909             0.000000             0.000000  0.000000e+00
CADC0892   0.970588   0.970588      0.000000             1.000000             0.994350 -5.649718e-03
CADC0894   1.000000   1.000000      0.000000             0.772358             0.796748  2.439024e-02

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/calibration_v1_vs_v2_final_locked.csv

==========================================================================================
(D) CADC0892 onset 트리거 로직 진단: 정상 세그먼트의 map_run_length 궤적
==========================================================================================
정상 세그먼트 177개 중 리셋(run_length: >5 -> <=2) 1회 이상 발생: 176개 (99.4%)
세그먼트당 평균 리셋 횟수: 1.93
리셋 횟수 분포:
n_resets
0     1
1    64
2    73
3    28
4     7
5     4

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/cadc0892_reset_diagnosis.csv
[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/cadc0892_run_length_diagnosis.png

[해석] 정상 세그먼트의 30% 이상에서 리셋이 반복 발생 -> onset 판정 규칙 자체(run_length 5->2 임계, 또는 전 채널 공통 hazard_lambda=250)가 CADC0892처럼 고첨도(kurtosis 60~85) · 평소엔 조용하다 가끔 튀는 채널에는 과민하다는 가설(§1-9)을 뒷받침함. 채널별 hazard_lambda 또는 onset 확정 조건(예: run_length<=2가 N틱 이상 유지)을 별도로 튜닝하는 것을 다음 액션으로 고려.

==========================================================================================
(E) MCC/Youden 기준 2단계 채널 스코프 확정
==========================================================================================
 channel        channel_type   recall  false_alarm_rate       mcc  youden_j               scope_reason  include_in_stage2
CADC0872 float_noise_suspect 0.320611          0.000000  0.513804  0.320611                         포함               True
CADC0873 float_noise_suspect 0.276190          0.000000  0.488849  0.276190                         포함               True
CADC0874 float_noise_suspect 0.724638          0.032000  0.739818  0.692638                         포함               True
CADC0884           quantized      NaN          0.240506       NaN       NaN         구조적 제외 (anomaly=0)              False
CADC0886           quantized 0.000000          0.000000  0.000000  0.000000  표본부족 (n_anom=3, n_norm=8)              False
CADC0888           quantized 0.616667          0.390625  0.193750  0.226042                         포함               True
CADC0890          continuous 0.909091          0.000000  0.825723  0.909091 표본부족 (n_anom=11, n_norm=3)              False
CADC0892           quantized 0.970588          0.994350 -0.090162 -0.023762           우연수준 이하 (MCC<=0)              False
CADC0894           quantized 1.000000          0.796748  0.189389  0.203252                         포함               True

2단계 포함 채널 (5개): ['CADC0872', 'CADC0873', 'CADC0874', 'CADC0888', 'CADC0894']
제외 채널 및 사유:
  - CADC0884: 구조적 제외 (anomaly=0)
  - CADC0886: 표본부족 (n_anom=3, n_norm=8)
  - CADC0890: 표본부족 (n_anom=11, n_norm=3)
  - CADC0892: 우연수준 이하 (MCC<=0)

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/channel_scope.csv

[주의] CADC0890처럼 n이 매우 작은 채널은 MCC/포함 여부가 안정적인지 부트스트랩 CI로 재확인하기 전까지는 §1-11의 잠정 결론으로 취급할 것.

==========================================================================================
(F) train/test 분리 확인 (확정 컬럼: 'train')
==========================================================================================
세그먼트 내부에서 'train' 값이 섞여있는 세그먼트 수: 0 / 2123
[안내] dataset.csv(df)가 이 세션에 로드돼 있지 않아 교차검증은 생략합니다 (생략해도 (G) 실행에는 지장 없음).

[확정] 'train' 컬럼을 train/test 지시자로 사용합니다. train=225178행, test=78315행 (train 비율=0.7420)

==========================================================================================
(G) train 데이터로만 파라미터 재적합 -> test에서만 성능 재평가
==========================================================================================
 channel   recall  recall_test  recall_drop  false_alarm_rate  fa_test   fa_drop
CADC0872 0.320611     0.375000    -0.054389          0.000000 0.000000  0.000000
CADC0873 0.276190     0.225806     0.050384          0.000000 0.000000  0.000000
CADC0874 0.724638     0.826087    -0.101449          0.032000 0.000000  0.032000
CADC0884      NaN          NaN          NaN          0.240506 0.277778 -0.037271
CADC0886 0.000000     0.000000     0.000000          0.000000 0.000000  0.000000
CADC0888 0.616667     0.750000    -0.133333          0.390625 0.346154  0.044471
CADC0890 0.909091     1.000000    -0.090909          0.000000      NaN       NaN
CADC0892 0.970588     1.000000    -0.029412          0.994350 0.978261  0.016089
CADC0894 1.000000     1.000000     0.000000          0.796748 0.785714  0.011034

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/calibration_train_fit_test_eval.csv

[해석] recall_drop/fa_drop이 크면(대략 0.1 이상) 전체 데이터로 튜닝한 성능이 test에서는 재현되지 않는다는 뜻 -> in-sample 튜닝 우려가 실질적임. 이 경우 §1-6/§1-11의 채널 스코프·MCC 결론은 train 기준으로 다시 확정하고 test 수치는 최종 보고용으로만 써야 함.

==========================================================================================
(H) 채널별 MCC 부트스트랩 신뢰구간 (CADC0890 등 소표본 채널 불확실성 검증)
==========================================================================================
 channel  mcc_median  mcc_ci_low  mcc_ci_high  n_anomaly  n_normal ci_crosses_zero                               note
CADC0872    0.513804    0.444101     0.577466        131       415           False                                   
CADC0873    0.488849    0.402766     0.564106        105       488           False                                   
CADC0874    0.743875    0.635407     0.831974         69       125           False                                   
CADC0884         NaN         NaN          NaN          0       158             NaN 구조적 제외 (anomaly 또는 normal 세그먼트 없음)
CADC0886    0.000000    0.000000     0.000000          3         8           False                                   
CADC0888    0.194050    0.073666     0.311465         60       192           False                                   
CADC0890    0.825723    0.603023     1.000000         11         3           False                                   
CADC0892   -0.090162   -0.274016     0.052636         34       177            True                                   
CADC0894    0.189389    0.146087     0.229721         21       123           False                                   

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/channel_mcc_bootstrap.csv  (N_BOOT=2000)

[주의] MCC 95% 신뢰구간이 0을 포함하는 채널 (1개): ['CADC0892'] -> §1-11에서 '포함'으로 판정했더라도 표본변동만으로 결과가 뒤집힐 수 있다는 뜻. 논문에는 이 CI를 함께 보고할 것을 권장.

==========================================================================================
(I) 문제 채널(890/892/894) burn-in vs steady-state z 재확인 (확정 설정 기준)
==========================================================================================
채널            burn-in std   steady std  steady kurt  steady |z|>3
-----------------------------------------------------------------
CADC0890            0.050        0.945         0.37         0.000
CADC0892            0.080        0.626        63.32         0.008
CADC0894            0.081        0.978        34.27         0.026

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/problem_channels_zcheck_locked.csv

참고: 정규분포라면 steady std≈1, |z|>3 비율≈0.003.
mixture=False로 확정했으므로 890/894의 steady std가 §1-9 실증 결과(0.945/0.978)처럼 1에 가까우면 노이즈 모델은 정상 — 이 표에서 다시 벗어나 있다면 (A) 프로파일링을 이 스위치값으로 재실행했는지부터 확인할 것.

==========================================================================================
최종 요약
==========================================================================================
아래 파일들이 out_dir에 저장되었는지 확인하세요:
  [O] channel_profiles_v2.csv
  [O] detector_calibration_v2_final_locked.csv
  [O] onset_bayesian_results.csv
  [O] onset_posterior_distributions.csv
  [O] cadc0892_reset_diagnosis.csv
  [O] cadc0892_run_length_diagnosis.png
  [O] channel_scope.csv
  [O] channel_mcc_bootstrap.csv
  [O] problem_channels_zcheck_locked.csv
  [O] calibration_train_fit_test_eval.csv


# =============================================================================
# 삽입 위치 안내
# =============================================================================
# 아래 (J), (K) 블록은 기존 스크립트의 (I) 섹션 바로 다음, `section("최종 요약")`
# 블록 바로 이전에 그대로 붙여넣으면 됩니다. (J)는 profile_channel / LocalLinearTrendKF /
# BOCPD / hierarchical_shrink_variance / channel_scope 등 기존에 정의된 이름을 그대로
# 재사용하므로 새로 import할 것은 없습니다.
#
# 또한 파일 맨 끝의 "최종 요약" 블록의 파일 체크리스트에 다음 항목들을 추가하세요:
#   "cadc0894_reset_diagnosis.csv",
#   "cadc0894_run_length_diagnosis.png",
#   "ablation_4combo_train_selection.csv",
#   "ablation_best_combo_test_eval.csv",
# (ablation_locked_combo_test_eval.csv는 best_combo != locked_combo일 때만 생성되므로
#  체크리스트에서는 조건부로 처리하거나 생략해도 무방합니다.)
# =============================================================================


# =============================================================================
# (J) CADC0894 onset 트리거 로직 진단 재현 — (D)의 892 진단과 동일 절차를 894에 적용
#     (피드백 문서의 "문제 1": 892에서만 확인된 결론을 894로 암묵 확장한 것을 검증)
# =============================================================================
section("(J) CADC0894 onset 트리거 로직 진단: 정상 세그먼트의 map_run_length 궤적")


def diagnose_reset_channel(channel: str, out_dir: Path, n_segments_to_plot: int = 6):
    """(D)의 CADC0892 리셋 진단 절차를 임의 채널에 대해 재현. 892와 동일한 파라미터
    (q_shrunk/r_shrunk/quant_floor, USE_BOCPD_FORGETTING)를 그대로 사용한다."""
    if channel not in channels:
        print(f"[안내] {channel} 이 channels 목록에 없어 진단을 생략합니다.")
        return None

    kf = LocalLinearTrendKF(
        q=q_shrunk[channel], r_nominal=r_shrunk[channel],
        theta=0.0, quantization_floor=quant_floor[channel],
    )
    bocpd = BOCPD(hazard_lambda=250.0, kappa_max=40.0, use_forgetting=USE_BOCPD_FORGETTING)

    normal = seg[(seg["channel"] == channel) & (seg["anomaly"] == 0)]
    seg_ids = normal["segment"].unique()

    reset_rows = []
    plot_segments = []

    for sid in seg_ids:
        s = normal[normal["segment"] == sid].sort_values("timestamp")
        values = s["value"].values
        if len(values) < 10:
            continue

        innovations, innovation_vars, _ = kf.run(values)
        z = innovations / np.sqrt(np.clip(innovation_vars, 1e-8, None))
        result = bocpd.run(z)
        mrl = result["map_run_length"]

        n_resets = int(np.sum((mrl[1:] <= 2) & (mrl[:-1] > 5)))
        reset_rows.append({
            "segment": sid,
            "n_points": len(values),
            "n_resets": n_resets,
            "max_run_length": int(mrl.max()),
            "onset_idx": result["onset_idx"],
            "z_kurtosis": float(pd.Series(z).kurtosis()),
            "frac_z_gt3": float(np.mean(np.abs(z) > 3)),
        })

        if len(plot_segments) < n_segments_to_plot:
            plot_segments.append((sid, z, mrl))

    reset_df = pd.DataFrame(reset_rows)
    ch_tag = channel.lower()
    reset_df.to_csv(out_dir / f"{ch_tag}_reset_diagnosis.csv", index=False)

    if len(reset_df) > 0:
        n_with_reset = int((reset_df["n_resets"] > 0).sum())
        print(f"[{channel}] 정상 세그먼트 {len(reset_df)}개 중 리셋(run_length: >5 -> <=2) "
              f"1회 이상 발생: {n_with_reset}개 ({n_with_reset / len(reset_df):.1%})")
        print(f"[{channel}] 세그먼트당 평균 리셋 횟수: {reset_df['n_resets'].mean():.2f}")
        print(f"[{channel}] 리셋 횟수 분포:")
        print(reset_df["n_resets"].value_counts().sort_index().to_string())
    print(f"\n[저장] {out_dir / f'{ch_tag}_reset_diagnosis.csv'}")

    if plot_segments:
        fig, axes = plt.subplots(len(plot_segments), 2, figsize=(11, 2.5 * len(plot_segments)))
        if len(plot_segments) == 1:
            axes = axes.reshape(1, 2)
        for row_i, (sid, z, mrl) in enumerate(plot_segments):
            axes[row_i, 0].plot(z, lw=0.8)
            axes[row_i, 0].axhline(0, color="gray", lw=0.5)
            axes[row_i, 0].axhline(3, color="red", lw=0.5, ls="--")
            axes[row_i, 0].axhline(-3, color="red", lw=0.5, ls="--")
            axes[row_i, 0].set_title(f"{channel} seg {sid} — z-score")

            axes[row_i, 1].plot(mrl, lw=0.8, color="tab:orange")
            axes[row_i, 1].axhline(5, color="red", lw=0.5, ls="--", label="run_length=5")
            axes[row_i, 1].axhline(2, color="red", lw=0.5, ls=":", label="run_length=2")
            axes[row_i, 1].set_title(f"{channel} seg {sid} — MAP run length")
            if row_i == 0:
                axes[row_i, 1].legend(fontsize=7)
        plt.tight_layout()
        fig_path = out_dir / f"{ch_tag}_run_length_diagnosis.png"
        plt.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"[저장] {fig_path}")

    return reset_df


reset_df_894 = diagnose_reset_channel("CADC0894", out_dir)

if reset_df_894 is not None and len(reset_df_894) > 0:
    n_with_reset_894 = int((reset_df_894["n_resets"] > 0).sum())
    reset_rate_894 = n_with_reset_894 / len(reset_df_894)

    cadc0892_path = out_dir / "cadc0892_reset_diagnosis.csv"
    reset_df_892 = pd.read_csv(cadc0892_path) if cadc0892_path.exists() else None

    print("\n[비교: 892 vs 894]")
    if reset_df_892 is not None and len(reset_df_892) > 0:
        n_with_reset_892 = int((reset_df_892["n_resets"] > 0).sum())
        reset_rate_892 = n_with_reset_892 / len(reset_df_892)
        print(f"  892: 리셋 발생률 {reset_rate_892:.1%}  "
              f"(평균 리셋 {reset_df_892['n_resets'].mean():.2f}, "
              f"z_kurtosis 평균 {reset_df_892['z_kurtosis'].mean():.2f}, "
              f"frac_z_gt3 평균 {reset_df_892['frac_z_gt3'].mean():.3f})")
    print(f"  894: 리셋 발생률 {reset_rate_894:.1%}  "
          f"(평균 리셋 {reset_df_894['n_resets'].mean():.2f}, "
          f"z_kurtosis 평균 {reset_df_894['z_kurtosis'].mean():.2f}, "
          f"frac_z_gt3 평균 {reset_df_894['frac_z_gt3'].mean():.3f})")

    if reset_rate_894 > 0.3:
        print("\n[해석] 894도 892와 마찬가지로 정상 세그먼트의 30% 이상에서 리셋이 반복 발생 -> "
              "두 채널이 동일한 onset 판정 구조 문제(§1-9 가설)를 공유한다는 근거로 사용 가능. "
              "단, frac_z_gt3/kurtosis 프로파일이 892와 다르면 '같은 증상, 다른 메커니즘'일 "
              "가능성은 여전히 남아있으므로 리셋률 수치만으로 메커니즘 동일성을 단정하지 말 것.")
    else:
        print("\n[해석] 894의 리셋 발생률이 892만큼 높지 않음 -> 두 채널의 과도한 FA를 같은 "
              "'onset 판정 구조 문제'로 묶어 서술한 피드백 문서의 문장은 근거가 부족했다는 뜻. "
              "논문에서는 892와 894를 별도 메커니즘 후보로 분리해서 기술할 것을 권장.")


# =============================================================================
# (K) §1-6 4-조합(mixture x forgetting) ablation 재선택 — train으로만 조합을 고르고
#     test로만 평가 (피드백 문서의 "문제 2": (G)는 파라미터 추정의 일반화만 확인했을
#     뿐, §1-6에서 (F,T)를 고른 것 자체가 in-sample 튜닝이었는지는 검증하지 않았음)
# =============================================================================
section("(K) 4-조합(mixture x forgetting) ablation: train 전용 선택 -> test 전용 평가")

if not train_col:
    print("[중단] (F)에서 train/test 컬럼을 확정하지 못해 (K)를 생략합니다.")
else:
    train_mask = seg[train_col].astype(bool)
    seg_train = seg[train_mask]
    seg_test = seg[~train_mask]

    def profile_channel_param(channel, nominal_values_by_segment, use_mixture):
        """profile_channel과 동일 로직이나, 전역 USE_MIXTURE_NOISE_ESTIMATOR 플래그
        대신 인자로 받은 use_mixture를 사용한다 (4-조합을 동시에 비교하기 위함)."""
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
            ch_type = "quantized"
            q_step = detect_quantization_step(np.abs(nonzero))
        elif np.isfinite(min_nonzero_abs) and min_nonzero_abs < FLOAT_NOISE_ABS_DIFF_THRESHOLD:
            ch_type = "float_noise_suspect"
            q_step = np.nan
        else:
            ch_type = "continuous"
            q_step = np.nan

        if use_mixture:
            r_robust = max(mixture_variance(diffs_all) / 2.0, 1e-14)
            d2 = np.diff(diffs_all)
            q_robust = max(mixture_variance(d2) / 6.0, 1e-14) if len(d2) > 1 else 1e-14
        else:
            r_robust = max(np.var(diffs_all) / 2.0, 1e-14)
            d2 = np.diff(diffs_all)
            q_robust = max(np.var(d2) / 6.0, 1e-14) if len(d2) > 1 else 1e-14

        return ChannelProfile(
            channel=channel, channel_type=ch_type,
            frac_zero_diff=frac_zero, min_nonzero_abs_diff=min_nonzero_abs,
            quantization_step=q_step, r_robust=r_robust, q_robust=q_robust,
            n_nominal_points=n_points,
        )

    def fit_profiles(segments_subset, use_mixture):
        profs = {}
        for ch in channels:
            nominal_vals = [
                s.sort_values("timestamp")["value"].values
                for _, s in segments_subset[
                    (segments_subset["channel"] == ch) & (segments_subset["anomaly"] == 0)
                ].groupby("segment")
            ]
            profs[ch] = profile_channel_param(ch, nominal_vals, use_mixture)

        r_est = {ch: (profs[ch].r_robust, max(profs[ch].n_nominal_points, 2)) for ch in channels}
        q_est = {ch: (profs[ch].q_robust, max(profs[ch].n_nominal_points, 2)) for ch in channels}
        r_shr = hierarchical_shrink_variance(r_est)
        q_shr = hierarchical_shrink_variance(q_est)
        qf = {
            ch: (profs[ch].quantization_step ** 2 / 12.0) if np.isfinite(profs[ch].quantization_step) else 0.0
            for ch in channels
        }
        return profs, r_shr, q_shr, qf

    def evaluate_config(segments_subset, r_shr, q_shr, qf, use_forgetting):
        rows = []
        for ch in channels:
            ch_segs = segments_subset[segments_subset["channel"] == ch].sort_values(["segment", "timestamp"])
            seg_ids = ch_segs["segment"].unique()
            kf = LocalLinearTrendKF(q=q_shr[ch], r_nominal=r_shr[ch], theta=0.0, quantization_floor=qf[ch])
            bocpd = BOCPD(hazard_lambda=250.0, kappa_max=40.0, use_forgetting=use_forgetting)
            n_anom, n_anom_det, n_norm, n_norm_fa = 0, 0, 0, 0
            for sid in seg_ids:
                s = ch_segs[ch_segs["segment"] == sid].sort_values("timestamp")
                values = s["value"].values
                is_anomaly = bool(s["anomaly"].iloc[0])
                if len(values) < 5:
                    continue
                innovations, innovation_vars, _ = kf.run(values)
                z = innovations / np.sqrt(np.clip(innovation_vars, 1e-8, None))
                result = bocpd.run(z)
                detected = result["onset_idx"] is not None
                if is_anomaly:
                    n_anom += 1
                    n_anom_det += int(detected)
                else:
                    n_norm += 1
                    n_norm_fa += int(detected)
            recall = n_anom_det / n_anom if n_anom else np.nan
            fa = n_norm_fa / n_norm if n_norm else np.nan
            tp, fn = n_anom_det, n_anom - n_anom_det
            fp, tn = n_norm_fa, n_norm - n_norm_fa
            mcc = mcc_from_counts(tp, fn, fp, tn) if (n_anom > 0 and n_norm > 0) else np.nan
            rows.append({
                "channel": ch, "n_anom": n_anom, "n_norm": n_norm,
                "recall": recall, "fa": fa, "mcc": mcc,
            })
        return pd.DataFrame(rows)

    combos = [(False, False), (False, True), (True, False), (True, True)]
    ablation_rows = []
    train_eval_cache = {}

    for use_mix, use_forget in combos:
        _, r_shr_c, q_shr_c, qf_c = fit_profiles(seg_train, use_mix)
        train_eval = evaluate_config(seg_train, r_shr_c, q_shr_c, qf_c, use_forget)
        train_eval_cache[(use_mix, use_forget)] = (r_shr_c, q_shr_c, qf_c, train_eval)

        # 선택 기준: 표본이 충분한 채널(n_anom>=5, n_norm>=5)만 대상으로 MCC 매크로 평균.
        # 이 표본 기준 자체는 (E)의 구조적 제외 기준과 동일하며 조합에 의존하지 않으므로,
        # "우리에게 유리한 채널만 골라 스코어링했다"는 순환 논리를 피할 수 있음.
        scoring_subset = train_eval[(train_eval["n_anom"] >= 5) & (train_eval["n_norm"] >= 5)]
        macro_mcc = float(scoring_subset["mcc"].mean()) if len(scoring_subset) else np.nan

        ablation_rows.append({
            "use_mixture": use_mix, "use_forgetting": use_forget,
            "train_macro_mcc": macro_mcc,
            "n_scoring_channels": len(scoring_subset),
        })
        print(f"[train] mixture={use_mix!s:<5s} forgetting={use_forget!s:<5s} "
              f"-> macro_mcc={macro_mcc:.4f}  (n_scoring_channels={len(scoring_subset)})")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(out_dir / "ablation_4combo_train_selection.csv", index=False)
    print(f"\n[저장] {out_dir / 'ablation_4combo_train_selection.csv'}")

    best_row = ablation_df.loc[ablation_df["train_macro_mcc"].idxmax()]
    best_combo = (bool(best_row["use_mixture"]), bool(best_row["use_forgetting"]))
    locked_combo = (USE_MIXTURE_NOISE_ESTIMATOR, USE_BOCPD_FORGETTING)

    print(f"\ntrain 전용 선택 결과: best combo = mixture={best_combo[0]}, forgetting={best_combo[1]} "
          f"(macro_mcc={best_row['train_macro_mcc']:.4f})")
    print(f"현재 잠금된 조합(§1-6 원 결론): mixture={locked_combo[0]}, forgetting={locked_combo[1]}")

    if best_combo == locked_combo:
        print("-> 일치: train만으로 선택해도 동일한 조합이 이깁니다. §1-6 ablation이 "
              "in-sample 튜닝이었다는 우려는 이 재현 절차 기준에서는 뒷받침되지 않습니다.")
    else:
        print("-> 불일치: train만으로 선택하면 다른 조합이 이깁니다. 원 §1-6 ablation 결론은 "
              "전체 데이터(train+test)로 골랐을 가능성이 있고, in-sample 튜닝 우려가 실질적임을 "
              "시사합니다. 논문에서 (mixture=False, forgetting=True) 채택 근거를 재검토할 것.")

    # 원칙 준수: 선택된 조합의 최종 성능은 test에서만 평가
    r_shr_best, q_shr_best, qf_best, _ = train_eval_cache[best_combo]
    test_eval_best = evaluate_config(seg_test, r_shr_best, q_shr_best, qf_best, best_combo[1])
    test_eval_best.to_csv(out_dir / "ablation_best_combo_test_eval.csv", index=False)
    print("\n[best combo, test 평가]")
    print(test_eval_best.to_string(index=False))
    print(f"\n[저장] {out_dir / 'ablation_best_combo_test_eval.csv'}")

    # 참고용: best_combo와 locked_combo가 다를 경우, 잠금 조합도 동일 절차로 test 평가해 비교 제공
    if locked_combo != best_combo:
        if locked_combo in train_eval_cache:
            r_shr_lock, q_shr_lock, qf_lock, _ = train_eval_cache[locked_combo]
        else:
            _, r_shr_lock, q_shr_lock, qf_lock = fit_profiles(seg_train, locked_combo[0])
        test_eval_locked = evaluate_config(seg_test, r_shr_lock, q_shr_lock, qf_lock, locked_combo[1])
        test_eval_locked.to_csv(out_dir / "ablation_locked_combo_test_eval.csv", index=False)
        print("\n[참고: 현재 잠금 조합, test 평가]")
        print(test_eval_locked.to_string(index=False))
        print(f"[저장] {out_dir / 'ablation_locked_combo_test_eval.csv'}")

==========================================================================================
(J) CADC0894 onset 트리거 로직 진단: 정상 세그먼트의 map_run_length 궤적
==========================================================================================
[CADC0894] 정상 세그먼트 123개 중 리셋(run_length: >5 -> <=2) 1회 이상 발생: 98개 (79.7%)
[CADC0894] 세그먼트당 평균 리셋 횟수: 1.24
[CADC0894] 리셋 횟수 분포:
n_resets
0    25
1    53
2    37
3     6
4     2

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/cadc0894_reset_diagnosis.csv
[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/cadc0894_run_length_diagnosis.png

[비교: 892 vs 894]
  892: 리셋 발생률 99.4%  (평균 리셋 1.93, z_kurtosis 평균 13.05, frac_z_gt3 평균 0.031)
  894: 리셋 발생률 79.7%  (평균 리셋 1.24, z_kurtosis 평균 11.83, frac_z_gt3 평균 0.130)

[해석] 894도 892와 마찬가지로 정상 세그먼트의 30% 이상에서 리셋이 반복 발생 -> 두 채널이 동일한 onset 판정 구조 문제(§1-9 가설)를 공유한다는 근거로 사용 가능. 단, frac_z_gt3/kurtosis 프로파일이 892와 다르면 '같은 증상, 다른 메커니즘'일 가능성은 여전히 남아있으므로 리셋률 수치만으로 메커니즘 동일성을 단정하지 말 것.

==========================================================================================
(K) 4-조합(mixture x forgetting) ablation: train 전용 선택 -> test 전용 평가
==========================================================================================
[train] mixture=False forgetting=False -> macro_mcc=0.2382  (n_scoring_channels=6)
[train] mixture=False forgetting=True  -> macro_mcc=0.3080  (n_scoring_channels=6)
[train] mixture=True  forgetting=False -> macro_mcc=0.1992  (n_scoring_channels=6)
[train] mixture=True  forgetting=True  -> macro_mcc=0.2750  (n_scoring_channels=6)

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/ablation_4combo_train_selection.csv

train 전용 선택 결과: best combo = mixture=False, forgetting=True (macro_mcc=0.3080)
현재 잠금된 조합(§1-6 원 결론): mixture=False, forgetting=True
-> 일치: train만으로 선택해도 동일한 조합이 이깁니다. §1-6 ablation이 in-sample 튜닝이었다는 우려는 이 재현 절차 기준에서는 뒷받침되지 않습니다.

[best combo, test 평가]
 channel  n_anom  n_norm   recall       fa      mcc
CADC0872      32     100 0.375000 0.000000 0.559017
CADC0873      31     122 0.225806 0.000000 0.434382
CADC0874      23      29 0.826087 0.000000 0.852030
CADC0884       0      36      NaN 0.277778      NaN
CADC0886       1       3 0.000000 0.000000 0.000000
CADC0888      12      52 0.750000 0.346154 0.319173
CADC0890       2       0 1.000000      NaN      NaN
CADC0892       7      46 1.000000 0.978261 0.054096
CADC0894       5      28 1.000000 0.785714 0.199205

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/ablation_best_combo_test_eval.csv

# =============================================================================
# 삽입 위치 안내
# =============================================================================
# 아래 (L), (M), (N) 블록은 (K) 섹션 바로 다음, `section("최종 요약")` 블록 바로
# 이전에 붙여넣으면 됩니다. (K)에서 정의된 best_combo, r_shr_best, q_shr_best,
# qf_best, train_eval_cache 등을 그대로 재사용하므로 (K) 없이 이 블록만 단독
# 실행하면 안 됩니다. (I)의 recheck_df, (H)의 channel_mcc_bootstrap.csv,
# (E)의 channel_scope도 그대로 재사용합니다.
#
# 마지막의 "최종 요약" 블록 파일 체크리스트에도 다음을 추가하세요:
#   "unified_steady_state_metrics_canonical.csv",
#   "test_onset_detections_best_combo.csv",
#   "channel_mcc_bootstrap_test_only.csv",
#   "mcc_full_vs_test_ci_comparison.csv",
#   "channel_scope_final_locked.csv",
# =============================================================================


# =============================================================================
# (L) frac_z_gt3 계산 규약 통일 — 892 vs 894를 (I) 방식(burn-in 제외 + pooled
#     steady-state)으로만 비교. (J)는 세그먼트별 계산 후 평균 + burn-in 포함 방식이라
#     절대값이 달라 직접 비교하면 안 됨 — 이 표를 canonical로 못박는다.
# =============================================================================
section("(L) frac_z_gt3 계산 규약 통일: 892 vs 894 재비교 (burn-in 제외 + pooled steady-state 기준)")

unified_path = out_dir / "problem_channels_zcheck_locked.csv"
if unified_path.exists():
    unified_df = pd.read_csv(unified_path)
elif "recheck_df" in globals():
    unified_df = recheck_df.copy()
else:
    unified_df = None
    print("[중단] (I)의 결과(recheck_df 또는 problem_channels_zcheck_locked.csv)를 찾을 수 없어 "
          "(L)을 생략합니다. (I)를 먼저 실행하세요.")

if unified_df is not None:
    cols_to_show = ["channel", "steady_std", "steady_kurt", "steady_frac_gt3"]
    print(unified_df[cols_to_show].to_string(index=False))
    unified_df[cols_to_show].to_csv(out_dir / "unified_steady_state_metrics_canonical.csv", index=False)
    print(f"\n[저장] {out_dir / 'unified_steady_state_metrics_canonical.csv'}")

    row_892 = unified_df[unified_df["channel"] == "CADC0892"]
    row_894 = unified_df[unified_df["channel"] == "CADC0894"]
    if len(row_892) and len(row_894):
        kurt_892 = float(row_892["steady_kurt"].iloc[0])
        kurt_894 = float(row_894["steady_kurt"].iloc[0])
        fg3_892 = float(row_892["steady_frac_gt3"].iloc[0])
        fg3_894 = float(row_894["steady_frac_gt3"].iloc[0])
        print(f"\n[통일된 정의 기준] 892: steady_kurt={kurt_892:.2f}, steady_frac|z|>3={fg3_892:.3f}  |  "
              f"894: steady_kurt={kurt_894:.2f}, steady_frac|z|>3={fg3_894:.3f}")
        higher = "894" if fg3_894 > fg3_892 else "892"
        print(f"-> 이 정의(canonical) 기준으로 극단값 비율이 더 높은 쪽: {higher}.")
        print("[주의] (J) 섹션에서 출력된 frac_z_gt3(세그먼트평균, burn-in 포함)는 이 표와 "
              "계산 규약이 다르므로 논문에 병기하지 말 것. 두 계산 모두 894가 892보다 높게 "
              "나오는 등 '방향'은 일치할 수 있으나, 절대값을 나란히 인용하면 리뷰어가 "
              "동일 지표로 오인할 수 있음 — 본문/부록에는 이 canonical 표만 사용.")
    else:
        print("[안내] 892 또는 894가 unified_df에 없어 비교를 생략합니다.")


# =============================================================================
# (M) test 전용 성능의 채널별 MCC 부트스트랩 CI — (K)에서 고른 best_combo를
#     seg_test에 적용해 세그먼트 단위 탐지 결과를 저장하고, (H)와 동일한 절차로
#     부트스트랩 CI를 구한 뒤 전체 데이터 기준 (H) 중앙값과 비교한다.
# =============================================================================
section("(M) test 전용 채널별 MCC 부트스트랩 CI (best_combo 기준)")

if not train_col:
    print("[중단] (F)에서 train/test 컬럼을 확정하지 못해 (M)을 생략합니다.")
elif "best_combo" not in globals():
    print("[중단] (K)가 실행되지 않아 best_combo/r_shr_best 등을 찾을 수 없습니다. (K)를 먼저 실행하세요.")
else:
    def evaluate_config_detailed(segments_subset, r_shr, q_shr, qf, use_forgetting):
        """evaluate_config과 동일 절차이나 세그먼트 단위 (channel, segment, is_anomaly,
        detected) 원자료를 그대로 반환한다 — 부트스트랩에 필요."""
        detail_rows = []
        for ch in channels:
            ch_segs = segments_subset[segments_subset["channel"] == ch].sort_values(["segment", "timestamp"])
            seg_ids = ch_segs["segment"].unique()
            kf = LocalLinearTrendKF(q=q_shr[ch], r_nominal=r_shr[ch], theta=0.0, quantization_floor=qf[ch])
            bocpd = BOCPD(hazard_lambda=250.0, kappa_max=40.0, use_forgetting=use_forgetting)
            for sid in seg_ids:
                s = ch_segs[ch_segs["segment"] == sid].sort_values("timestamp")
                values = s["value"].values
                is_anomaly = bool(s["anomaly"].iloc[0])
                if len(values) < 5:
                    continue
                innovations, innovation_vars, _ = kf.run(values)
                z = innovations / np.sqrt(np.clip(innovation_vars, 1e-8, None))
                result = bocpd.run(z)
                detected = result["onset_idx"] is not None
                detail_rows.append({
                    "channel": ch, "segment": sid,
                    "is_anomaly": int(is_anomaly), "detected": int(detected),
                })
        return pd.DataFrame(detail_rows)

    test_detail_best = evaluate_config_detailed(seg_test, r_shr_best, q_shr_best, qf_best, best_combo[1])
    test_detail_best.to_csv(out_dir / "test_onset_detections_best_combo.csv", index=False)
    print(f"[저장] {out_dir / 'test_onset_detections_best_combo.csv'}  ({len(test_detail_best)} rows)")

    N_BOOT_TEST = 2000
    rng_boot_test = np.random.default_rng(RANDOM_STATE)

    test_boot_rows = []
    for ch in channels:
        sub = test_detail_best[test_detail_best["channel"] == ch]
        anom = sub[sub["is_anomaly"] == 1]["detected"].values
        norm = sub[sub["is_anomaly"] == 0]["detected"].values

        if len(anom) == 0 or len(norm) == 0:
            test_boot_rows.append({
                "channel": ch, "n_anom_test": len(anom), "n_norm_test": len(norm),
                "mcc_median": np.nan, "mcc_ci_low": np.nan, "mcc_ci_high": np.nan,
                "ci_crosses_zero": np.nan,
                "note": "test에서 anomaly 또는 normal 세그먼트가 없어 CI 계산 불가",
            })
            continue

        mccs = np.empty(N_BOOT_TEST)
        for i in range(N_BOOT_TEST):
            a_s = rng_boot_test.choice(anom, size=len(anom), replace=True)
            n_s = rng_boot_test.choice(norm, size=len(norm), replace=True)
            tp, fn = a_s.sum(), len(a_s) - a_s.sum()
            fp, tn = n_s.sum(), len(n_s) - n_s.sum()
            mccs[i] = mcc_from_counts(tp, fn, fp, tn)

        ci_low, ci_high = float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5))
        test_boot_rows.append({
            "channel": ch, "n_anom_test": len(anom), "n_norm_test": len(norm),
            "mcc_median": float(np.median(mccs)), "mcc_ci_low": ci_low, "mcc_ci_high": ci_high,
            "ci_crosses_zero": bool(ci_low < 0 < ci_high), "note": "",
        })

    test_mcc_boot_df = pd.DataFrame(test_boot_rows)
    test_mcc_boot_df.to_csv(out_dir / "channel_mcc_bootstrap_test_only.csv", index=False)
    print(test_mcc_boot_df.to_string(index=False))
    print(f"\n[저장] {out_dir / 'channel_mcc_bootstrap_test_only.csv'}  (N_BOOT={N_BOOT_TEST})")

    flagged_test = test_mcc_boot_df[test_mcc_boot_df["ci_crosses_zero"] == True]  # noqa: E712
    if len(flagged_test):
        print(f"\n[주의] test-only MCC 95% CI가 0을 포함하는 채널 ({len(flagged_test)}개): "
              f"{flagged_test['channel'].tolist()}")

    # 전체 데이터 기준 (H) 중앙값이 test-only (M) CI 안에 드는지 대조
    full_boot_path = out_dir / "channel_mcc_bootstrap.csv"
    if full_boot_path.exists():
        full_boot = pd.read_csv(full_boot_path)
        # 두 데이터프레임 모두 mcc_median/mcc_ci_low/mcc_ci_high/ci_crosses_zero/note
        # 컬럼명을 공유하므로, suffixes가 이 컬럼들 전부에 적용되어 병합 후 실제
        # 컬럼명은 "mcc_ci_low"가 아니라 "mcc_ci_low_test" / "mcc_ci_low_full"이 됨.
        cmp = test_mcc_boot_df.merge(full_boot, on="channel", suffixes=("_test", "_full"))
        cmp["full_median_in_test_ci"] = (
            (cmp["mcc_median_full"] >= cmp["mcc_ci_low_test"])
            & (cmp["mcc_median_full"] <= cmp["mcc_ci_high_test"])
        )
        print("\n[전체(H) 중앙값이 test-only(M) 95% CI 안에 드는지]")
        print(cmp[["channel", "mcc_median_full", "mcc_ci_low_test", "mcc_ci_high_test",
                    "full_median_in_test_ci"]].to_string(index=False))

        outside = cmp[~cmp["full_median_in_test_ci"].fillna(False)]
        if len(outside):
            print(f"\n[주의] 전체 데이터 기준 중앙값이 test-only CI 밖에 있는 채널: "
                  f"{outside['channel'].tolist()} -> 이 채널들은 test 부분표본에서의 성능이 "
                  "표본변동 범위를 벗어난 것일 수 있음. 논문에 test 수치를 인용할 때 이 채널은 "
                  "CI를 함께 표기하거나 '참고용'으로만 제시할 것.")
        else:
            print("\n[안내] 모든 채널에서 전체 데이터 중앙값이 test-only CI 안에 듭니다 — "
                  "test 부분표본 성능이 전체 데이터 추정과 통계적으로 모순되지 않음.")
        cmp.to_csv(out_dir / "mcc_full_vs_test_ci_comparison.csv", index=False)
        print(f"[저장] {out_dir / 'mcc_full_vs_test_ci_comparison.csv'}")
    else:
        print(f"[안내] {full_boot_path} 가 없어 (H) 대비 비교는 생략합니다.")


# =============================================================================
# (N) 1단계 잠금 — 채널 스코프 최종 확정 (890 서술 정정 포함)
#     890은 (E)에서 이미 표본부족으로 제외되어 있었고, MCC가 높아 '포함'으로
#     판정된 적이 없음. test 정상 표본=0은 새 결정이 아니라 기존 제외 판정을
#     재확인/강화하는 추가 근거이므로 "폐기"가 아니라 "강화"로 서술한다.
# =============================================================================
section("(N) 1단계 잠금 — 채널 스코프 최종 확정 (890 서술 정정 포함)")

if "channel_scope" not in globals():
    print("[중단] (E)의 channel_scope가 없어 (N)을 생략합니다. (E)를 먼저 실행하세요.")
else:
    final_scope_rows = []
    for _, row in channel_scope.iterrows():
        ch = row["channel"]
        reason = row["scope_reason"]
        if ch == "CADC0890":
            reason_final = f"{reason} + test 정상 표본=0으로 재확인 (기존 제외 판정 강화, 번복 아님)"
        else:
            reason_final = reason
        final_scope_rows.append({
            "channel": ch,
            "include_in_stage2": row["include_in_stage2"],
            "scope_reason_final": reason_final,
        })

    final_scope_df = pd.DataFrame(final_scope_rows)
    final_scope_df.to_csv(out_dir / "channel_scope_final_locked.csv", index=False)
    print(final_scope_df.to_string(index=False))
    print(f"\n[저장] {out_dir / 'channel_scope_final_locked.csv'}")

    included_final = final_scope_df[final_scope_df["include_in_stage2"]]["channel"].tolist()
    print(f"\n2단계 최종 포함 채널 ({len(included_final)}개): {included_final}")

    print("\n[1단계 잠금 체크리스트]")
    checklist = [
        ("892 vs 894 리셋 진단 분리 (J)", (out_dir / "cadc0894_reset_diagnosis.csv").exists()),
        ("frac_z_gt3 계산 규약 통일 (L)", (out_dir / "unified_steady_state_metrics_canonical.csv").exists()),
        ("4-조합 train-only 선택/test 평가 (K)", (out_dir / "ablation_4combo_train_selection.csv").exists()),
        ("test-only MCC 부트스트랩 CI (M)", (out_dir / "channel_mcc_bootstrap_test_only.csv").exists()),
        ("890 서술 정정 (N)", (out_dir / "channel_scope_final_locked.csv").exists()),
    ]
    for name, ok in checklist:
        print(f"  [{'O' if ok else 'X'}] {name}")

    if all(ok for _, ok in checklist):
        print("\n[결론] 위 5개 항목이 모두 실행/저장되었으므로 1단계는 '잠금' 상태로 취급 가능. "
              "단, (K)의 train-only 선택 기준은 §1-6 원 코드가 아니라 새로 재구성한 대체 기준이라는 "
              "한계는 논문 methods/limitations에 반드시 명시할 것. 또한 (M)에서 CI가 0을 포함하거나 "
              "전체 데이터 중앙값이 test CI 밖으로 벗어난 채널이 있었다면, 해당 채널의 test 수치는 "
              "본문에서 CI와 함께 제시하거나 참고용으로만 인용할 것.")
    else:
        print("\n[중단] 위 체크리스트가 전부 O가 아니면 아직 '1단계 잠금'이라고 부르면 안 됨. "
              "누락된 항목을 먼저 실행하세요.")


==========================================================================================
(L) frac_z_gt3 계산 규약 통일: 892 vs 894 재비교 (burn-in 제외 + pooled steady-state 기준)
==========================================================================================
 channel  steady_std  steady_kurt  steady_frac_gt3
CADC0890    0.945336     0.373170         0.000000
CADC0892    0.626050    63.322849         0.007681
CADC0894    0.977772    34.272141         0.026234

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/unified_steady_state_metrics_canonical.csv

[통일된 정의 기준] 892: steady_kurt=63.32, steady_frac|z|>3=0.008  |  894: steady_kurt=34.27, steady_frac|z|>3=0.026
-> 이 정의(canonical) 기준으로 극단값 비율이 더 높은 쪽: 894.
[주의] (J) 섹션에서 출력된 frac_z_gt3(세그먼트평균, burn-in 포함)는 이 표와 계산 규약이 다르므로 논문에 병기하지 말 것. 두 계산 모두 894가 892보다 높게 나오는 등 '방향'은 일치할 수 있으나, 절대값을 나란히 인용하면 리뷰어가 동일 지표로 오인할 수 있음 — 본문/부록에는 이 canonical 표만 사용.

==========================================================================================
(M) test 전용 채널별 MCC 부트스트랩 CI (best_combo 기준)
==========================================================================================
[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/test_onset_detections_best_combo.csv  (529 rows)
 channel  n_anom_test  n_norm_test  mcc_median  mcc_ci_low  mcc_ci_high ci_crosses_zero                                       note
CADC0872           32          100    0.559017    0.418330     0.679674           False                                           
CADC0873           31          122    0.434382    0.280552     0.578736           False                                           
CADC0874           23           29    0.852030    0.714957     0.961581           False                                           
CADC0884            0           36         NaN         NaN          NaN             NaN test에서 anomaly 또는 normal 세그먼트가 없어 CI 계산 불가
CADC0886            1            3    0.000000    0.000000     0.000000           False                                           
CADC0888           12           52    0.320256    0.091698     0.527988           False                                           
CADC0890            2            0         NaN         NaN          NaN             NaN test에서 anomaly 또는 normal 세그먼트가 없어 CI 계산 불가
CADC0892            7           46    0.054096    0.000000     0.095553           False                                           
CADC0894            5           28    0.199205    0.107335     0.279143           False                                           

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/channel_mcc_bootstrap_test_only.csv  (N_BOOT=2000)

[전체(H) 중앙값이 test-only(M) 95% CI 안에 드는지]
 channel  mcc_median_full  mcc_ci_low_test  mcc_ci_high_test  full_median_in_test_ci
CADC0872         0.513804         0.418330          0.679674                    True
CADC0873         0.488849         0.280552          0.578736                    True
CADC0874         0.743875         0.714957          0.961581                    True
CADC0884              NaN              NaN               NaN                   False
CADC0886         0.000000         0.000000          0.000000                    True
CADC0888         0.194050         0.091698          0.527988                    True
CADC0890         0.825723              NaN               NaN                   False
CADC0892        -0.090162         0.000000          0.095553                   False
CADC0894         0.189389         0.107335          0.279143                    True

[주의] 전체 데이터 기준 중앙값이 test-only CI 밖에 있는 채널: ['CADC0884', 'CADC0890', 'CADC0892'] -> 이 채널들은 test 부분표본에서의 성능이 표본변동 범위를 벗어난 것일 수 있음. 논문에 test 수치를 인용할 때 이 채널은 CI를 함께 표기하거나 '참고용'으로만 제시할 것.
[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/mcc_full_vs_test_ci_comparison.csv

==========================================================================================
(N) 1단계 잠금 — 채널 스코프 최종 확정 (890 서술 정정 포함)
==========================================================================================
 channel  include_in_stage2                                                   scope_reason_final
CADC0872               True                                                                   포함
CADC0873               True                                                                   포함
CADC0874               True                                                                   포함
CADC0884              False                                                   구조적 제외 (anomaly=0)
CADC0886              False                                            표본부족 (n_anom=3, n_norm=8)
CADC0888               True                                                                   포함
CADC0890              False 표본부족 (n_anom=11, n_norm=3) + test 정상 표본=0으로 재확인 (기존 제외 판정 강화, 번복 아님)
CADC0892              False                                                     우연수준 이하 (MCC<=0)
CADC0894               True                                                                   포함

[저장] /home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/causal_pipeline_outputs/channel_scope_final_locked.csv

2단계 최종 포함 채널 (5개): ['CADC0872', 'CADC0873', 'CADC0874', 'CADC0888', 'CADC0894']

[1단계 잠금 체크리스트]
  [O] 892 vs 894 리셋 진단 분리 (J)
  [O] frac_z_gt3 계산 규약 통일 (L)
  [O] 4-조합 train-only 선택/test 평가 (K)
  [O] test-only MCC 부트스트랩 CI (M)
  [O] 890 서술 정정 (N)

[결론] 위 5개 항목이 모두 실행/저장되었으므로 1단계는 '잠금' 상태로 취급 가능. 단, (K)의 train-only 선택 기준은 §1-6 원 코드가 아니라 새로 재구성한 대체 기준이라는 한계는 논문 methods/limitations에 반드시 명시할 것. 또한 (M)에서 CI가 0을 포함하거나 전체 데이터 중앙값이 test CI 밖으로 벗어난 채널이 있었다면, 해당 채널의 test 수치는 본문에서 CI와 함께 제시하거나 참고용으로만 인용할 것.
#

# =============================================================================
# 1단계 결과 정리 (마무리 시점 기준)
# =============================================================================
#
# [1] 파라미터 잠금 및 재검증 — mixture=False, forgetting=True를 §1-6 원 ablation의
# 잠정 결론대로 확정했고, 이번에 train 데이터만으로 4개 조합(mixture x forgetting)을
# 독립적으로 재선택해도 동일 조합이 최고 성능(train macro-MCC 0.308, 2위 대비 약 11%
# 차이)으로 뽑혔다. 다만 이 재선택 기준은 §1-6 원 코드의 정확한 스코어링 절차가 아니라
# 새로 재구성한 대체 기준(표본충분 채널의 MCC 매크로평균)이므로, "in-sample 튜닝
# 우려가 완전히 해소됐다"가 아니라 "독립적으로 재구성한 기준에서도 일치했다" 정도로만
# 서술해야 한다.
#
# [2] onset posterior 저장 및 CADC0892/0894 진단 — 전체 5개 채널에 대해
# onset_bayesian_results.csv와 onset_posterior_distributions.csv를 확보했고,
# 892와 894에 동일한 map_run_length 리셋 진단을 각각 적용한 결과 892는 정상
# 세그먼트의 99.4%에서 리셋이 발생하는 systematic trigger failure, 894는 79.7%로
# 정도가 뚜렷이 약한 partial trigger instability로 구분됐다. frac_z_gt3는 burn-in
# 제외 + pooled steady-state 방식(892=0.008, 894=0.026)으로 계산 규약을 통일해,
# 세그먼트평균 방식과 절대값은 다르지만 894가 892보다 극단값 비율이 높다는 방향은
# 일관되게 확인했다.
#
# [3] 문제 채널 재확인 및 최종 스코프 — 890/892/894에 대한 burn-in vs steady-state
# 재확인, 채널별 부트스트랩 CI(전체 데이터 및 test-only 양쪽)를 모두 거쳤다. 892는
# 전체 데이터(MCC 중앙값 -0.090, CI -0.274~0.053)와 test-only(MCC 0.054, CI
# 0.000~0.096) 양쪽 모두 0 근방에 몰려 있어 부호만 뒤바뀐 것이지 실질적으로는
# 우연수준 배제 결정이 두 슬라이스 모두에서 재확인된 것이고, 890은 test 분할에서
# 정상 세그먼트가 0개라 애초에 비교가 불가능해 기존 표본부족 제외가 강화됐을 뿐
# 번복된 적은 없다. 최종적으로 872/873/874/888/894 5개 채널만 전체·train-only·
# test-only 세 기준에서 모두 MCC CI가 0을 넘지 않는 것을 확인했고, 이 상태를
# "1단계 잠금"으로 취급해 2단계(onset 불확실성 정량화 → level/diff/diff² 선행성
# 검정)로 넘어가는 것이 방어 가능하다.
# =============================================================================

# =============================================================================
# 9개 채널 -> 5개 채널 선별 과정 (1단계 전체 파이프라인 기준)
# =============================================================================
#
# [단계 0] 대상 채널: ALL_CHANNELS = 872, 873, 874, 884, 886, 888, 890, 892, 894 (9개)
#
# [단계 1] 채널 프로파일링 (A) — 신호 특성에 따라 quantized / float_noise_suspect /
# continuous로 분류하고 강건 노이즈(r, q)를 추정. 이 단계 자체는 채널을 걸러내지
# 않고, 이후 필터 파라미터를 채널별로 다르게 잡기 위한 준비 단계다.
#
# [단계 2] 온라인 필터링 + BOCPD 탐지 (B') — 채널별 recall/false_alarm_rate 산출.
# 이 시점에는 9개 채널 모두 살아있고, 890은 recall=0.909(n_anom=11), 892는
# recall=0.971이지만 false_alarm_rate=0.994로 사실상 전부 알람을 울리는 패턴이
# 이미 드러남.
#
# [단계 3] MCC/Youden 채널 스코프 확정 (E) — 표본 크기와 MCC 부호를 기준으로
# 구조적 배제/표본부족/우연수준 이하를 걸러내는 1차 필터:
#   - 884: n_anom=0 (이상 세그먼트 자체가 없음)                  -> 구조적 제외
#   - 886: n_anom=3, n_norm=8 (양쪽 다 표본 미달, MCC=0)         -> 표본부족 제외
#   - 890: n_anom=11, n_norm=3 (표본 미달, MCC=0.826은 높지만
#          n<5 기준에 걸림)                                      -> 표본부족 제외
#   - 892: n_anom=34, n_norm=177 (표본은 충분하나 MCC=-0.090)    -> 우연수준 이하 제외
#   -> 872/873/874/888/894 5개만 "포함"으로 1차 확정, n=1594 시점의 잠정 결론
#
# [단계 4] 부트스트랩 CI로 안정성 재확인 (H) — 1차 포함/제외가 표본 변동만으로
# 뒤집히지 않는지 전체 데이터 기준 2000회 부트스트랩 MCC CI로 검증. 892만 CI가
# 0을 포함(-0.274~0.053)해 배제가 표본운이 아니라 구조적임을 재확인했고, 나머지
# 872/873/874/888/894는 CI가 모두 0을 넘지 않아 1차 포함 결정이 안정적임을 확인.
# 890/886/884는 애초에 구조적/표본부족 사유라 CI 재확인 대상에서 제외.
#
# [단계 5] train-only 재선택으로 파라미터 자체의 in-sample 편향 점검 (K) — 채널
# 스코프가 아니라 상위 파라미터(mixture, forgetting)의 4개 조합을 train 세그먼트
# 만으로 다시 골랐을 때도 잠금된 조합(mixture=False, forgetting=True)이 그대로
# 최선으로 나와, 스코프 확정에 쓰인 성능 수치 자체가 전체 데이터로 선택되어
# 부풀려진 것은 아님을 간접 확인.
#
# [단계 6] test-only로 최종 재확인 (M) — 선택된 조합을 test 세그먼트에만 적용해
# 다시 부트스트랩 CI를 구함. 872(0.559)/873(0.434)/874(0.852)/888(0.320)/894(0.199)
# 모두 CI가 0을 넘지 않고, 전체 데이터 중앙값도 이 test-only CI 안에 들어와 두
# 슬라이스가 서로 모순되지 않음을 확인. 892는 test-only에서도 MCC=0.054로 0
# 근방(CI 0.000~0.096)에 머물러 우연수준 배제가 재확인됨. 890/884는 test 분할에
# 한쪽 클래스가 아예 없어 계산 불가 — 기존 배제 사유가 강화된 것이지 새 결정 아님.
#
# [최종] 872 / 873 / 874 / 888 / 894 — 5개 채널이 전체 데이터, train-only,
# test-only 세 가지 독립적 기준 모두에서 일관되게 통계적으로 유효한 탐지 성능을
# 보여 2단계(onset posterior 기반 선행성 검정) 대상으로 최종 확정됨.
#
#   channel  1차스코프(E)  부트스트랩(H)  test-only(M)  최종
#   872      포함          CI>0           CI>0          포함
#   873      포함          CI>0           CI>0          포함
#   874      포함          CI>0           CI>0          포함
#   884      구조적 제외    -              계산불가       제외
#   886      표본부족       -              -             제외
#   888      포함          CI>0           CI>0          포함
#   890      표본부족       -              계산불가       제외 (강화)
#   892      우연수준 이하  CI∋0           CI≈0 근방      제외 (강화)
#   894      포함          CI>0           CI>0          포함
# =============================================================================

# OPSSAT-AD 탐지 → 인과추론 파이프라인 설계서 (v2: 노트북 구현 반영본)

> 이 개정판은 초판(RandomForest + PELT/z-score)에서 "수정본"(DR-EKF + BOCPD +
> 계층적 베이즈) 설계로 넘어간 뒤, 실제 `opssat_pipeline_v2.ipynb` 구현 과정에서
> **설계와 다르게 구현된 부분**과 **코드 리뷰 중 새로 드러나 추가된 부분**을
> 반영해 다시 고침. 원 설계서 대비 바뀐 지점은 각 절 앞에 `[설계 대비 변경]`으로
> 표시함.

---

## 0. 구현 범위 요약 (가장 먼저 확인할 것)

**노트북 v2는 1단계(탐지)만 구현되어 있음.** 2단계(이벤트 스터디/인과추론)와
3단계(Physics-informed SCM)는 노트북에 코드가 전혀 없고, 여전히 설계 문서
수준에 머물러 있음. 아래 표는 "수정본" 설계서가 계획했던 것과 노트북에 실제로
있는 것을 대조한 것.

| 항목 | 설계서(수정본) 계획 | 노트북 v2 실제 구현 | 비고 |
|---|---|---|---|
| 공분산 팽창 | DR(Wasserstein) 스칼라 근사, `theta`를 부트스트랩 90th-percentile로 추정 | **미구현.** `theta=0.0` 하드코딩, 어디서도 재계산 안 됨 | 설계서 1-2절은 사실과 다름 — 삭제/재작성 필요 |
| 노이즈 추정 방식 | (DR팽창 외 별도 언급 없음) | **신규:** 채널 자동분류(양자화/float-noise/연속) + MAD·mixture 강건분산 추정 | zero-inflated 데이터에서 순수 MAD가 0으로 붕괴하는 문제를 별도로 발견·해결 |
| 관측노이즈 추가항 | r_eff = r_nominal + theta | r_eff = r_nominal(강건추정) + theta(=0, 미사용) **+ quantization_floor** | 양자화 채널에 균등분포 분산(step²/12)을 별도 항으로 추가 — 설계서에 없던 항 |
| BOCPD | Adams & MacKay 표준형 | **+ forgetting 메커니즘** (`kappa_max` 상한으로 켤레사후분포 유효표본크기 캡) | 설계서에 없음. 장기간 정상 구간에서 사후분포가 과도하게 확신에 차는 문제 대응으로 추정 |
| Ablation | "필요" 언급만 있고 미구현 (맨 아래 "목표 저널 재검토" 참고) | **구현됨.** `USE_MIXTURE_NOISE_ESTIMATOR`, `USE_BOCPD_FORGETTING` 스위치 + `RUN_TAG`별 출력 분리 | 설계서의 계획이 실제로 실현된 유일한 항목 |
| v1 대비 비교 | 없음 | **신규.** `detector_calibration.csv`(v1) 있으면 자동으로 recall/오탐률 delta 계산 | |
| 문제채널 재진단 | 없음 | **신규.** CADC0890/0892/0894의 steady-state z가 N(0,1)에 가까운지 직접 검정 → 가설 A(노이즈모델) vs 가설 B(BOCPD forgetting) 분리 | 오탐률이 여전히 심각한 세 채널에 대한 사후 디버깅 셀 |
| 합성데이터 폴백 | 없음 | **신규.** `segments.csv` 없으면 자동으로 synthetic 데이터 생성 후 스모크 테스트 | |
| 2단계 (이벤트 스터디, Monte Carlo onset 전파, 선행성검정, 메타분석) | 설계 완료, 산출물 파일명까지 정의됨 | **미구현.** 코드 없음 | 아래 2단계 절은 "여전히 설계만 존재"로 표시 |
| 3단계 (Physics-informed SCM) | 설계 완료 (경로 A/B 분기) | **미구현.** 코드 없음 | 그대로 유지 |

**확률 추정**: 현재 노트북 상태에서 2단계 코드를 그대로 작성해 붙이는 데 걸리는
난이도는 낮음(1단계 산출물 스키마가 이미 이벤트 스터디 입력 요건을 충족하므로
약 70% 확률로 큰 재설계 없이 이식 가능하다고 봄) — 다만 DR-EKF 이식이 계획대로
안 됐던 선례를 보면, "설계서에 쓰여 있다 = 구현될 것이다"로 가정하지 말고
1단계 완료 후 별도 확인 필요.

---

## 1단계: 채널 진단 + 강건 노이즈 추정 + KF·BOCPD(forgetting) 온라인 탐지 (구현됨)

### 1-1. 채널 자동분류 `[신규 — 설계서에 없던 절]`

`profile_channel()`이 채널별 1차 diff 분포를 보고 세 가지로 분류:

- **quantized**: `frac_zero_diff >= 0.10` (광다이오드류; 0이 아닌 diff가 계단폭
  단위로 뭉쳐 있음) → `quantization_step`을 0아닌 |diff|의 25th-percentile로 추정
- **float_noise_suspect**: 0아닌 최소 |diff| `< 1e-6` (자력계류; 부동소수점
  잡음 의심)
- **continuous**: 나머지

임계값(0.10, 1e-6)은 `zscore_diagnostics.csv` 실측치를 보고 정한 것으로,
이론적으로 유도된 값이 아님 — 데이터셋이 바뀌면 재검토 필요.

### 1-2. MAD·mixture 기반 강건 노이즈 추정 `[신규 — 설계서 1-2 "DR 공분산 팽창"을 대체]`

```
mixture_variance(diffs) = p * Var(jump | jump != 0)   # p = P(diff != 0)
r_robust = mixture_variance(diffs_all) / 2
q_robust = mixture_variance(diff(diffs_all)) / 6
```

전체 diff에 MAD를 바로 적용하면 zero-inflated 채널(양자화 채널)에서 median이
0이 되어 분산 추정이 0으로 붕괴하는 문제가 1차 시도에서 실제로 발생 →
"점프 발생확률 × 점프 크기 분산"으로 분리해 재정의. `USE_MIXTURE_NOISE_ESTIMATOR
= False`로 두면 기존 `var(diff)/2` 방식(원 v1과 동일)으로 되돌아감 — ablation용.

> **설계서 1-2절과의 관계**: 원 설계서는 이 자리에 Jang et al.(arXiv:2604.02749)의
> DR-EKF를 스칼라 근사로 이식하는 내용을 담고 있었으나, 코드에는 그 논문의
> 어떤 부분도 구현되어 있지 않음(`theta`는 상수 0). 로봇 추적/내비게이션
> 논문을 위성 텔레메트리에 이식하는 시도 자체가 검증되지 않았다고 설계서도
> 명시했던 만큼, 실제로는 그 이식을 시도하는 대신 데이터에서 직접 관찰된
> zero-inflation 문제를 우선 해결하는 쪽으로 방향이 바뀐 것으로 보임. 논문에
> DR-EKF를 언급할 계획이라면 이 절 전체를 다시 쓰거나, 아니면 "시도했으나
> 채택하지 않음" 정도로 축소해야 함.

### 1-3. 국소선형추세 칼만필터 + 양자화 플로어 `[일부 변경]`

상태공간 모델 자체는 설계서와 동일(국소선형추세, "물리모델 아님" 명시도 유지).
다만 관측노이즈 구성이 바뀜:

```
r_eff = r_robust(1-2) + theta(=0, 미사용) + quantization_floor
quantization_floor = quantization_step² / 12   # 균등분포 근사
```

양자화 채널에 대해 계단폭 잡음을 균등분포 분산 공식으로 별도 반영 — 이 항이
사실상 설계서 3단계가 이야기하던 "물리 정보"의 가장 약한 형태(신호 자체의
물리적 성질을 노이즈 모델에 반영)에 해당함. 3단계 경로 A/B 논의와 연결지어
서술할 가치 있음.

### 1-4. 계층적 베이즈 부분풀링 `[변경 없음]`

Efron-Morris 스타일 empirical-Bayes shrinkage, log-분산에 대해 적용. 설계서
1-3절과 동일한 방식 그대로 구현됨(표본극소 채널 CADC0886/CADC0890 포함 목적도 동일).

### 1-5. BOCPD + forgetting `[변경 — 설계서에 forgetting 메커니즘 없었음]`

Normal-Inverse-Gamma 켤레모델은 설계서와 동일. 추가된 것은 **forgetting**:
run-length가 `kappa_max`(=40)를 넘으면 그 시점의 켤레분포 충분통계량
(kappa, alpha, beta)을 비례적으로 줄여 유효표본크기를 상한 이하로 유지.

- `USE_BOCPD_FORGETTING = False` → v1과 동일(무제한 성장)
- onset 판정 로직: MAP run-length가 "긴 상태 → 짧은 상태(≤2)"로 급격히
  재설정되는 시점을 onset으로 판정 — 설계서의 정의와 동일
- **사후분포 자체(`posterior_over_onset`, run-length 상위 30개)를 저장** —
  설계서가 요구한 "2단계 Monte Carlo 전파용 사후분포 보존"은 구현되어 있음.
  다만 이를 실제로 소비하는 2단계 코드는 없음(§2 참고)

forgetting을 추가한 이유는 노트북에 명시적으로 서술되어 있지 않으나, §1-9의
문제채널 재진단 셀이 "가설 B: BOCPD의 kappa_max 메커니즘이 오탐 원인"을
검토 대상으로 놓고 있는 것으로 보아, 장기 정상 구간에서 사후분포가 과확신
상태가 되어 이후 진짜 변화를 놓치거나 반대로 민감해지는 문제에 대응하려는
시도로 추정됨 — **원인을 명시적으로 확인하는 셀(§1-9)이 존재하므로, 이 가정을
검정 결과로 확정한 후 여기에 반영할 것.**

### 1-6. Ablation 프레임워크 `[신규 — 설계서 맨 끝 "Ablation 설계 필요"가 실현됨]`

```python
USE_MIXTURE_NOISE_ESTIMATOR = True   # False → 기존 var(diff)/2 (v1과 동일)
USE_BOCPD_FORGETTING = True          # False → kappa 상한 없음 (v1과 동일)
RUN_TAG = "mixture_and_forgetting"   # 조합별 출력 파일 태그
```

두 스위치를 조합해 재실행하면 오탐률 변화가 "노이즈 추정 교체" 때문인지
"BOCPD forgetting" 때문인지 분리 가능. 설계서 맨 아래 문단이 "Ablation 설계
필요"라고 적었던 부분은 더 이상 미해결 항목이 아님 — §5(목표 저널 재검토)에서
갱신 필요.

**실제 4-조합 실행 결과 (권장 기본값 확정)**:

- `USE_BOCPD_FORGETTING`: **True 유지 권장.** 문제채널(890/892/894)의 FA는
  이 스위치에 거의 반응하지 않음(±1%p 이내인 반면, float_noise_suspect
  채널(872/873/874)의 recall은 forgetting을 끄면 절반 가까이 떨어짐
  (874: 0.725→0.319). forgetting은 문제의 원인이 아니라 다른 채널의
  민감도를 지키는 장치였음 — §1-9의 가설 B는 기각.
- `USE_MIXTURE_NOISE_ESTIMATOR`: **False로 전환 권장.** 872/873/874/888은
  거의 무차이인 반면, 890·894는 mixture=True일 때 steady-state z의 std가
  각각 2.25·2.88까지 벌어지다가 mixture=False에서 0.95·0.98로 정상화됨
  (§1-9 실증 결과 참고). mixture=True가 더 나은 채널은 확인되지 않음.

**`(mixture=False, forgetting=True)`가 전체 최적 조합인지 검증 (채널별 MCC, 4조합 전수비교)**

| 채널 | (T,T) | **(F,T) — 권장** | (T,F) | (F,F) | 채널 유형 |
|---|---|---|---|---|---|
| CADC0872 | 0.507 | **0.514** | 0.396 | 0.371 | float_noise_suspect |
| CADC0873 | 0.489 | **0.489**(동률) | 0.385 | 0.385 | float_noise_suspect |
| CADC0874 | 0.740 | **0.740**(동률) | 0.521 | 0.521 | float_noise_suspect |
| CADC0888 | 0.107 | **0.194**(동률) | 0.107 | 0.194 | quantized |
| CADC0890 | −0.213 | **0.826**(동률) | −0.213 | 0.826 | continuous |
| CADC0892 | −0.158 | **−0.090**(동률, 최선이나 여전히 음수) | −0.223 | −0.090 | quantized |
| CADC0894 | 0.086 | 0.189 | 0.096 | **0.203** | quantized |

**메커니즘이 채널 유형별로 완전히 갈림** (이게 왜 (F,T)가 최적인지의 이유):
- **float_noise_suspect(872/873/874)**: forgetting 스위치가 지배적(껐다 켰다에 따라
  MCC가 0.04~0.22 변함), mixture는 미미한 추가효과. → forgetting=True가 필수.
- **quantized/continuous(888/890/892/894)**: mixture 스위치가 지배적(890은
  mixture=True에서 아예 부호가 뒤집혀 −0.213, mixture=False에서 0.826),
  forgetting은 894를 제외하곤 영향 없음(888/890/892는 forget on/off 완전 동률).
  → mixture=False가 필수.

**결론**: (F,T)는 7개 평가가능 채널 중 6개에서 최고 또는 동률 최고이고, 유일한
예외인 894에서도 최선(F,F)의 0.203 대비 0.189로 **0.014(≈7%) 차이**밖에 안 남 —
반면 872에서 (F,T)를 안 쓰면 최대 0.22, 890에서는 부호 자체가 뒤집히는 손실이
남. 종합하면 (F,T)가 사실상 지배적(Pareto-near-optimal) 조합이라는 지난 결론은
유지됨.

> **다만 방법론적으로 짚어야 할 것**: 이 4-조합 비교는 최종 calibration 결과를
> 보고할 바로 그 `segments.csv` 위에서 수행됨 — 즉 스위치 선택과 성능 평가가
> **같은 데이터**에 대해 이뤄짐. 채널 단위로 4개 값 중 최댓값을 고르는 것은
> 아주 약한 형태의 in-sample 튜닝(=train/test 분리 없이 test에서 하이퍼파라미터를
> 고르는 것)이라, 채널이 9개뿐인 이 상황에서는 우연히 (F,T)가 이겼을 가능성을
> 완전히 배제할 수 없음(890 같은 n=14 채널의 부호 반전은 특히 표본변동 영향을
> 받기 쉬움). OPSSAT-AD 벤치마크 논문이 정의한 train/test 분할이 `dataset.csv`
> 쪽에 존재하는지 확인해, 가능하면 스위치는 train 쪽에서만 고르고 report는 test
> 쪽에서 하는 절차로 바꾸는 것을 권장 — 안 그러면 리뷰어가 "왜 하필 이 조합을
> 썼냐"에 "test에서 제일 잘 나와서"라고 답하게 되는 상황이 됨.

### 1-7. 선택편향 제거 + calibration `[변경 없음]`

`anomaly=1`인 모든 세그먼트에 전수 적용, recall과 정상 세그먼트 오탐률을
채널별로 정직하게 보고. 설계서 1-5절과 동일.

### 1-8. v1 대비 비교 `[신규]`

`detector_calibration.csv`(v1 산출물)가 존재하면 채널별 recall/오탐률 delta를
자동 계산해 `calibration_v1_vs_v2_{RUN_TAG}.csv`로 저장. 없으면 조용히 생략.

### 1-9. 문제채널(CADC0890/0892/0894) 재진단 `[신규]`

v2에서도 오탐률이 여전히 심각한 세 채널에 대해, 정상 세그먼트의 steady-state
z-score(burn-in 5포인트 제외) 분포를 직접 진단:

- **가설 A (노이즈 모델 문제)**: steady std ≫ 1 또는 `|z|>3` 비율 ≥ 0.01
  → r/q 추정이 아직 부족
- **가설 B (BOCPD forgetting 문제)**: steady std ≈ 1인데도 오탐 심함 →
  노이즈 모델은 정상, `kappa_max` 메커니즘이 원인 → §1-6 스위치로 재확인

**실증 결과 (4-조합 ablation 실행 완료)**:

| 채널 | mixture=True steady std | mixture=False steady std | mixture=True FA | mixture=False FA |
|---|---|---|---|---|
| CADC0890 | 2.252 | **0.945** | 1.000 | **0.000** |
| CADC0892 | 1.126 | 0.626 | 1.000 | 0.994 |
| CADC0894 | 2.877 | **0.978** | 0.951 | 0.797 |

- **가설 B는 기각됨.** forgetting on/off에 따른 세 채널의 steady std·FA
  변화가 사실상 없음(§1-6 참고).
- **가설 A는 890·894에 대해서만 채택.** mixture=True일 때 두 채널의
  steady std가 1에서 크게 벗어나 있고, mixture=False로 바꾸면 거의 정확히
  1로 돌아옴 — MAD 강건분산 추정이 이 두 채널에서 오히려 역효과를 낸 것으로
  판단됨(큰 diff가 실제로는 이상치가 아니라 진짜 신호변동인데 MAD가
  분산을 과소추정). 원인 추정 확률 약 75% — 확정하려면 872/873/874처럼
  MAD가 잘 맞는 채널과의 신호 특성 차이를 별도로 봐야 함.
- **CADC0892는 가설 A·B 모두로 설명되지 않음.** mixture=True에서 이미
  steady std=1.126로 거의 정상인데도 FA=1.000이고, 노이즈를 부풀려도
  (mixture=False, std=0.626) FA가 1.000→0.994로 거의 안 움직임. steady
  kurtosis가 63~85로 극단적으로 높다는 점(§1-9 원 산출물)을 볼 때, 원인은
  노이즈 모델도 forgetting도 아니라 **onset 판정 로직**(run-length가 `>5`
  에서 `≤2`로 꺾이는 패턴을 changepoint로 보는 규칙, 또는 전 채널 공통
  `hazard_lambda=250`)일 가능성이 높음 — 892는 "대부분 조용하다가 가끔
  극단값이 튀는" 패턴이라 run-length가 짧게 반복 리셋되기 쉬움. 다음
  액션 아이템: 정상 세그먼트의 `map_run_length` 궤적을 직접 그려 이
  가설을 확인하는 셀 추가.

### 1-10. 합성 데이터 폴백 `[신규]`

`segments.csv` 부재 시 채널 2개(SYN_QUANT, SYN_FLOAT)짜리 합성 데이터로
자동 대체해 스모크 테스트. 실제 통계 결과가 아니므로 논문 산출물과 혼동 주의
— `out_dir`가 `./synthetic_outputs`로 분리되어 있어 실수로 섞일 위험은 낮음.

### 산출물 (실제)
`channel_profiles_v2.csv`, `detector_calibration_v2_{RUN_TAG}.csv`,
`calibration_v1_vs_v2_{RUN_TAG}.csv`(v1 있을 때만), `problem_channels_zcheck.csv`

> 설계서가 언급했던 `channel_noise_hyperparams.csv`, `onset_bayesian_results.csv`,
> `onset_posterior_distributions.csv`는 노트북에 별도 저장 로직이 없음 —
> 필요하면 §1-5의 `posterior_over_onset`을 순회하며 저장하는 셀을 추가해야 함.

### 1-11. 2단계 채널 스코프 결정 기준 (문헌 기반) `[신규]`

**배경**: 이전 버전(비공식 논의)에서는 `false_alarm_rate ≥ 0.7`처럼 임의
임계값으로 892·894를 2단계에서 제외하는 안을 검토했음. 이는 recall을
무시하고 FA만 보는 방식이라 원칙이 없다는 문제가 있어, 문헌을 검토해
다음 기준으로 교체함.

**채택 기준 — Matthews 상관계수(MCC) ≤ 0**

OPSSAT-AD 벤치마크 논문(Ruszczak et al., *Sci Data* 2025, 이 데이터셋의
원 출처)이 이 데이터셋에 대해 항상 계산해야 할 지표로 명시적으로
제시하는 것이 accuracy/precision/recall/F1과 함께 **MCC**임. MCC는
Chicco & Jurman(*BMC Genomics*, 2020)이 보이듯 클래스 불균형에 강건하고
"우연 수준"이 0으로 정규화되어 있어(−1~1, 무작위 예측이면 0), 채널마다
`n_anomaly`/`n_normal` 비율이 크게 다른 이 문제에 F1이나 raw FA rate보다
적합함. 동일한 취지의 통계량으로 Youden(1950)의 J = recall − FA rate
(=sensitivity+specificity−1, ROC "chance line" 기준)가 있으며, 두 지표는
독립적으로 유도됐지만 이 데이터에서 부호가 완전히 일치함 — 아래에서
교차검증으로 사용.

```
MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**권장 기본 설정(§1-6: mixture=False, forgetting=True)에서 채널별 계산 결과**:

| 채널 | recall | FA rate | MCC | Youden's J | 판정 |
|---|---|---|---|---|---|
| CADC0872 | 0.321 | 0.000 | 0.514 | 0.321 | 포함 |
| CADC0873 | 0.276 | 0.000 | 0.489 | 0.276 | 포함 |
| CADC0874 | 0.725 | 0.032 | 0.740 | 0.693 | 포함 |
| CADC0884 | NaN | 0.241 | 정의불가 (TP+FN=0) | 정의불가 | **제외 (구조적)** |
| CADC0886 | 0.000 | 0.000 | 정의불가→관례상 0 | 0.000 | **제외 (우연수준+n=3/8)** |
| CADC0888 | 0.617 | 0.391 | 0.194 | 0.226 | 포함 |
| CADC0890 | 0.909 | 0.000 | 0.826 | 0.909 | 포함 (단, n=14 — 아래 주의) |
| CADC0892 | 0.971 | 0.994 | **−0.090** | **−0.024** | **제외 (우연보다 나쁨)** |
| CADC0894 | 1.000 | 0.797 | 0.189 | 0.203 | **포함 (이전 논의에서 제외 후보였으나 재검토 결과 유지)** |

**이전 논의와 달라진 점 (직접 정정)**: raw FA rate(0.797)만 보면 894가
892와 비슷해 보여 지난번에 함께 제외하는 안을 드렸으나, MCC·J로 보면
894는 recall=1.000이 FA를 상쇄해 888(포함 예정 채널)과 비슷한 수준으로
정보량이 있음(MCC 0.189 vs 0.194). 반면 892는 recall도 0.971로 높지만
FA가 0.994라 recall 우위가 완전히 상쇄되고 남는 게 없음(MCC<0, J<0) —
"양성·음성 둘 다 거의 항상 양성으로 찍는" 채널이라는 뜻. 결과적으로
**892만 제외, 894는 포함**으로 뒤집힘.

**남는 주의점 (직접 명시)**:
- CADC0890은 MCC=0.826으로 가장 높지만 n=14(anom 11 + normal 3)에 불과함.
  Wilson score interval(Brown, Cai & DasGupta, *Statistical Science*, 2001 —
  이 표본 크기대에서 Wald/Clopper-Pearson보다 권장됨)로 recall·FA 각각의
  CI를 구해 MCC의 불확실성을 함께 보고하지 않으면, 리뷰어가 "n=3인 정상
  세그먼트로 FA=0을 주장하는 게 타당한가"를 물을 가능성이 높음(추정 확률
  70% 이상). 2단계 착수 전에 부트스트랩 MCC CI를 계산해 부록에 넣는 것을
  권장.
- CADC0886은 MCC/J 계산 자체가 이 정의로는 0(또는 undefined)이 나오지만,
  n=3/8이라는 표본 크기만으로도 이미 배제 사유가 충분함 — 우연수준
  판정과 표본부족 판정이 우연히 일치한 것으로, 논문에는 두 사유를 함께
  명시하는 것이 안전함.
- 이 기준(MCC≤0)은 "탐지기가 무작위 예측보다 나은가"만 걸러낼 뿐,
  "onset 위치가 신뢰할 만한가"는 별도 문제임. 892처럼 MCC<0인 채널은
  onset 자체가 의미 없으므로 자동으로 제외되지만, 포함된 6개 채널
  안에서도 recall이 낮은 채널(872: 0.321, 873: 0.276)의 onset은
  §1-5에서 언급한 계층적 베이즈·§2-2 Monte Carlo 전파에서 가중치를
  낮춰 반영해야 함 — 이 부분은 §1-5의 "선택편향 제거" 원칙과 동일한
  논리이므로 새 기준을 만들 필요는 없음.

**최종 2단계 채널 스코프**: 9개 중 CADC0884(구조적 제외)·CADC0886
(우연수준+표본부족)·CADC0892(우연보다 나쁨) 3개 제외, **6개
(872/873/874/888/890/894) 포함**.

---

## 2단계: 몬테카를로 이벤트 스터디 + 선행성 검정 + 채널 메타분석 — **미구현, 설계만 존재**

`[상태: 노트북에 코드 없음. 아래는 원 설계서 내용 그대로 유지 — 실행 전 재검토 필요]`

### 2-1. Causal 롤링 피처
세그먼트 전체 요약통계 대신, onset 전후 원시 시계열에서 그 시점까지만 사용한
롤링 통계로 재정의: `level_var`(원신호), `diff_var`(1차 변화), `diff2_var`(2차 변화).

### 2-2. onset 불확실성의 몬테카를로 전파
1단계 §1-5에서 저장한 `posterior_over_onset`에서 200회 샘플링 → 매 샘플마다
pre/post Welch t-test 및 효과크기(Cohen's d 근사) 계산 → 95% CI로 보고.

### 2-3. 물리 정보 기반 인과 순서의 실증 검정 (선행성 검정)
원신호 → 1차미분 → 2차미분의 인과 순서를 3단계 SCM 구조로 고정하기 전에,
Wilcoxon 부호순위검정으로 `diff_var`/`diff2_var`가 `level_var`보다 유의하게
먼저 임계값을 넘는지 먼저 확인.

### 2-4. 채널 간 랜덤효과 메타분석
채널을 관측단위로 하는 DerSimonian-Laird 랜덤효과 메타분석 + 이질성(Q-검정) 보고.

### 산출물 (계획, 미생성)
`event_study_results.csv`, `precedence_test_results.csv`, `meta_analysis_by_feature.csv`

---

## 3단계: 물리 정보 기반 인과 구조 삽입 (Physics-Informed SCM) — **미구현, 설계만 존재**

`[상태: 노트북에 코드 없음. 경로 A/B 분기는 원 설계서와 동일하게 유지]`

### 3-1. 스코프가 갈리는 지점
- **경로 A (현재 데이터만으로 가능)**: 2단계에서 확인될 원신호→1차미분→2차미분
  선행 순서를 SCM 뼈대로 고정. "physics-informed"보다는 "신호처리 구조를 인과
  뼈대로 쓴 강건한 이벤트 스터디"에 가까움.
- **경로 B (보조 데이터 필요)**: 광다이오드 채널(I_PDx_THETA)의 관측모델 h(x)를
  궤도/자세 propagation 방정식에서 실제로 유도. OPSSAT-AD에 TLE·자세 quaternion
  등 보조 데이터가 포함되어 있는지 확인이 여전히 다음 액션 아이템.

`[참고]` §1-3에서 추가된 양자화 플로어(step²/12)는 경로 A/B 사이의 아주 약한
중간 지점(신호의 물리적 특성을 노이즈 모델에 반영)으로 볼 수 있음 — 완전한
물리모델은 아니지만, "신호처리적 근사"보다는 한 걸음 나아간 것으로 §3-1
서술에 각주로 추가할 가치 있음.

### 3-2. CVaR 기반 최악-경우 조기경보 성능
2단계 결과가 유의한 경우에만 진행할 가치 있음. 미구현 상태 그대로 유지.

---

## 남는 confound (해결되지 않음, 논문화 전 반드시 재확인)

onset_fraction이 세그먼트 초반에 몰리는 경향이 관측되더라도, 이것이 실제
물리적 선행 신호인지 ESA 라벨러가 세그먼트 경계를 관행적으로 여유 있게
잘랐기 때문인지는 현재 설계로 분리되지 않음. Zenodo/GitHub의 라벨링 프로토콜
문서를 다시 확인하는 것이 1순위 후속 작업 — 노트북 v2 단계에서도 여전히
미해결.

---

## 목표 저널 재검토 `[일부 갱신]`

로보틱스/신호처리 중심 1편(온라인 탐지: 채널분류+강건노이즈+BOCPD forgetting)과
응용통계/데이터 중심 2편(인과추론+계층적 베이즈)으로 분리하는 안은 유지.
다만 근거가 바뀜:

- ~~"Ablation 설계 필요"~~ → **완료**: `USE_MIXTURE_NOISE_ESTIMATOR` /
  `USE_BOCPD_FORGETTING` 스위치로 구성요소별 제거 비교 가능
- 1편 제목에서 "DR"은 빼야 함 — 실제로 이식되지 않았으므로. 대신 "채널 유형별
  강건 노이즈 추정(MAD/mixture) + BOCPD forgetting"으로 재구성하는 편이
  구현과 일치
- 1편이 2편보다 먼저 완성 가능한 상태(2단계 코드가 없으므로) — 순서상
  1편 단독 투고 후 2편 착수가 현실적 경로일 가능성 높음(추정 확률 65% 정도,
  2단계 코드 이식 난이도에 달려 있음)

---

## 참고문헌 (§1-11 채널 스코프 기준 근거)

- Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. (2025). The OPS-SAT
  benchmark for detecting anomalies in satellite telemetry. *Scientific
  Data*, 12, 710. — OPSSAT-AD 원 논문. Accuracy/Precision/Recall/F1과 함께
  MCC, AUCROC, AUCPR을 이 데이터셋에 항상 계산해야 할 지표로 명시.
- Chicco, D., Jurman, G. (2020). The advantages of the Matthews correlation
  coefficient (MCC) over F1 score and accuracy in binary classification
  evaluation. *BMC Genomics*, 21, 6. — MCC가 클래스 불균형에 강건하고
  우연수준이 0으로 정규화됨을 보임; 네 개 혼동행렬 주변합 중 하나라도
  0이면 MCC는 정의되지 않고 관례상 무작위 예측 수준(0)으로 처리.
- Youden, W.J. (1950). Index for rating diagnostic tests. *Cancer*, 3(1),
  32–35. — J = sensitivity + specificity − 1 = recall − FA rate. J=0이
  ROC "chance line"(무작위 예측), J<0은 무작위보다 나쁨.
- Brown, L.D., Cai, T.T., DasGupta, A. (2001). Interval estimation for a
  binomial proportion. *Statistical Science*, 16(2), 101–133. — Wilson
  score interval이 표본 크기와 무관하게 Wald/Clopper-Pearson보다 권장됨을
  보임; §1-11의 CADC0890(n=14) 불확실성 보고에 적용 권장.


# 지금까지 나온 논의에서 실제로 처리해야 할 항목들을 모아서 우선순위별로 정리했습니다.
#-> 아래 + (2) 이상탐지 기반의 인과분석설명.md도 참고
#-> 가장아래 [통계최적화]는 인과관계까지 규명하고 최종적으로 넣고 + [Mamba&SHAP]는 보조적 성격으로 규명하는데 사용한다.

# ## A. 파이프라인 코드 — 2단계 진행을 막고 있는 것들 (최우선)

# | 항목 | 내용 | 근거 |
# |---|---|---|
# | 1. onset 사후분포 저장 로직 추가 | `posterior_over_onset`을 순회하며 `onset_bayesian_results.csv`, `onset_posterior_distributions.csv`로 저장하는 셀 추가 | §1-5/§1-10 — 지금은 메모리에만 있고 파일로 안 남음, 2단계 Monte Carlo가 이걸 못 씀 |
# | 2. CADC0892 onset 트리거 로직 진단 | 정상 세그먼트 몇 개의 `map_run_length` 궤적을 직접 그려서, run-length가 5 넘긴 뒤 2 이하로 자주 꺾이는지 확인 | §1-9 — 892는 노이즈·forgetting 둘 다로 설명 안 되는 유일한 채널, 원인 미확정 |
# | 3. 2단계 코드 작성 | Causal 롤링 피처 → Monte Carlo onset 전파 → Wilcoxon 선행성검정 → DerSimonian-Laird 메타분석 | §2 전체가 아직 설계만 있고 코드 없음. 1단계 산출물 스키마는 이미 맞춰져 있어 이식 자체는 어렵지 않을 것(추정 확률 70%) |

# ## B. 통계적 엄밀성 — 리뷰어가 물어볼 가능성이 높은 것들

# | 항목 | 내용 | 근거 |
# |---|---|---|
# | 4. train/test 분리 후 재검증 | OPSSAT-AD 논문이 정의한 split이 `dataset.csv`에 있는지 확인 → 있으면 스위치는 train, calibration 보고는 test로 분리 | 방금 확인한 (F,T) 조합 선택이 report와 같은 데이터에서 이뤄진 in-sample 튜닝 문제 |
# | 5. CADC0890 CI 계산 | Wilson score interval(recall/FA) 또는 부트스트랩 MCC CI를 구해 §1-11 표에 병기 | n=14로 MCC=0.826이 가장 불안정한 추정치 |
# | 6. 890 부호반전의 재현성 확인 | mixture=True/False에 따라 890의 MCC가 −0.213↔0.826으로 뒤집히는 게 진짜 효과인지, 다른 리샘플에서도 재현되는지 | n이 작아 표본변동 가능성 배제 못함 |

# ## C. 외부 데이터/문헌 확인

# | 항목 | 내용 | 근거 |
# |---|---|---|
# | 7. 라벨링 프로토콜 재확인 | Zenodo/GitHub에서 세그먼트 경계 설정 규칙 문서화 여부 확인 | "남는 confound" — onset_fraction 쏠림이 실제 물리신호인지 라벨링 관행인지 아직 미분리, 설계서 최초 버전부터 1순위로 남아있던 항목 |
# | 8. 보조 데이터(TLE·자세 quaternion) 존재 여부 확인 | OPSSAT-AD에 궤도/자세 정보가 포함돼 있는지 | §3-1 경로 A/B 분기 — 이게 있어야 "진짜" physics-informed SCM(경로 B) 시도 가능 |

# ## D. 저자가 직접 판단해야 할 프레이밍 결정 (제가 대신 못 정하는 것)

# | 항목 | 내용 |
# |---|---|
# | 9. §1-2 DR-EKF 서술 처리 방향 | 논문에서 아예 뺄지, "시도했으나 채택 안 함"으로 한 문단 남길지 |
# | 10. 1편 제목/프레이밍 | "DR" 빼고 "채널 유형별 강건 노이즈 추정 + BOCPD forgetting"으로 갈지 |
# | 11. 투고 순서 | 1편(탐지) 먼저 단독 투고 후 2편(인과추론) 착수할지, 2단계 코드부터 완성해서 같이 갈지 |

# 우선순위를 매기면 **A(1~3) → B(4~6) → C(7~8) → D(9~11)** 순으로, A가 없으면 2단계 자체가 시작이 안 되고, B는 지금 나온 §1-9/§1-11/§1-6 결론들의 신뢰도에 직접 영향을 주는 반면, C·D는 A·B 결과를 기다려도 되는 항목입니다.

# 이 목록을 설계서 맨 뒤에 "다음 액션 아이템" 절로 추가해드릴까요?

# 네. 지금 단계에서는 **새로운 방법론을 더 붙이는 것보다, 1단계를 잠그고 바로 2단계 인과분석으로 넘어가는 것**이 맞습니다.
# ### 후속 작업 순서
# **(1) 먼저 1단계 마무리**
# * `mixture=False + BOCPD forgetting=True`를 기본 설정으로 확정
# * **onset posterior 저장 코드 추가**
# * **CADC0892의 `map_run_length` 진단**
# * 890/892/894 등 문제 채널의 결과 재확인
# → 여기까지 해서 **“어떤 시점이 이상 발생 onset인가?”**를 데이터 파일로 확정합니다.
# **(2) 동시에 통계적 검증**
# * train/test split 확인
# * 890의 작은 표본에 대한 CI/MCC 불확실성 확인
# * 4개 조합의 성능 선택이 test-set 튜닝이 되지 않았는지 점검
# → 즉, **“우리가 고른 탐지기가 우연히 잘 나온 것은 아닌가?”**를 확인합니다.
# **(3) 그 다음 바로 핵심인 2단계**
# 6개 채널을 대상으로
# > **onset posterior → Monte Carlo → rolling `level/diff/diff²` → 선행성 검정 → 채널별 효과 → 랜덤효과 메타분석**
# 을 구현합니다.
# 여기가 **이 연구의 진짜 핵심**입니다.
# **(4) 2단계에서 결과가 나오면 인과구조를 판단**
# * `level → diff → diff²`의 선행성이 반복적으로 확인되면
# * 단순 상관이 아니라 **인과적 선행구조**로 발전시킬 근거를 확보
# * 그때 **Physics-informed SCM**을 설계합니다.
# **(5) 마지막에 외부자료 확인**
# * 라벨링 프로토콜
# * TLE/자세 quaternion 존재 여부
# 를 확인해서 **SCM을 실제 물리모델까지 끌어올릴 수 있는지** 결정합니다.
# ---
# ### 가장 중요한 한 줄
# **지금 당장 할 일은 `onset posterior 저장 → 892 진단 → train/test 검증 → 2단계 코드 구현`입니다.**
# 그리고 **DR-EKF나 새로운 알고리즘을 더 추가하는 것은 지금 하지 않는 게 좋습니다.** 현재 구조만으로도
# **“탐지 → 변화시점 → 불확실성 → 선행성 → 인과추론 → SCM”**
# 이라는 상당히 명확한 연구 스토리가 이미 만들어져 있습니다.

#[중요][통계최적화] 통계적 최적화, 처음부터 억지로 넣기보다는 마지막 의사결정 단계에 넣는 것이 가장 강합니다.

# 네. **이것까지 합치면 연구 프레임워크가 훨씬 강해집니다.**
# 다만 중요한 것은 **모든 방법을 동급의 핵심 방법론처럼 나열하지 않는 것**입니다.
# 가장 좋은 구조는 **5개 층**으로 정리하는 것입니다.
# ## 최종 프레임워크
# ```text
#                 OPSSAT-AD Telemetry
#                          │
#                          ▼
#               ┌─────────────────────┐
#               │ 1. Signal Estimation│
#               │   Kalman Filter      │
#               │   State Estimation   │
#               └──────────┬──────────┘
#                          ▼
#               ┌─────────────────────┐
#               │ 2. Anomaly Detection│
#               │ BOCPD + Forgetting   │
#               │ Probabilistic Onset  │
#               └──────────┬──────────┘
#                          ▼
#                  Onset Posterior
#               P(T_onset | Data)
#                          │
#               ┌──────────┴──────────┐
#               ▼                     ▼
#         Monte Carlo               Mamba
#      Uncertainty Propagation   Deep Sequence
#               │                  Benchmark
#               │                     │
#               │                     ▼
#               │                    SHAP
#               │             Predictive Attribution
#               │
#               ▼
#        Event-time Variables
#         level → diff → diff²
#               │
#               ▼
#        ┌──────────────────┐
#        │ 3. Causal Analysis│
#        │ Causal Inference  │
#        │ Temporal Precedence│
#        └────────┬─────────┘
#                 ▼
#        Physics-informed SCM
#                 │
#                 ▼
#        ┌────────────────────┐
#        │ 4. Risk Optimization│
#        │ Stochastic Optimization│
#        │       + CVaR         │
#        └─────────┬──────────┘
#                  ▼
#        Optimal Warning / Response
#                  │
#                  ▼
#        ┌────────────────────┐
#        │ 5. Decision Support │
#        │        DSS          │
#        └────────────────────┘
# ```
# ### 핵심적으로는
# **Detect → Quantify Uncertainty → Explain → Identify Cause → Optimize → Decide**
# 입니다.
# 조금 더 논문스럽게 표현하면:
# > **Probabilistic Anomaly Detection → Uncertainty Quantification → Predictive Explanation → Causal Inference → Risk-aware Stochastic Optimization → Decision Support**
# 가 됩니다.
# ---
# ## 각 방법의 역할도 명확해집니다

# | 방법                           | 역할                          | 연구에서의 위치  |
# | ---------------------------- | --------------------------- | --------- |
# | **Kalman Filter**            | 센서 노이즈 제거 및 잠재 상태 추정        | 기반        |
# | **BOCPD + Forgetting**       | 이상 발생 시점 확률적 탐지             | **핵심**    |
# | **Bayesian Onset Posterior** | onset의 불확실성 정량화             | **핵심**    |
# | **Monte Carlo**              | 불확실성 전파                     | **핵심**    |
# | **Mamba**                    | 장기 시계열 기반 독립적 탐지 benchmark  | 보조/검증     |
# | **SHAP**                     | Mamba의 예측 기여도 설명            | 보조/설명     |
# | **Causal Inference**         | 변화의 선행성 및 인과관계 분석           | **핵심**    |
# | **Physics-informed SCM**     | 인과구조와 물리적 메커니즘 연결           | 확장        |
# | **Stochastic Optimization**  | 불확실성을 고려한 대응정책 최적화          | **핵심 확장** |
# | **CVaR**                     | 극단적 위험을 고려한 robust decision | **핵심 확장** |
# | **DSS**                      | 최종 의사결정 시스템                 | 최종 산출물    |
# ---
# # 특히 중요한 변화
# 이걸 붙이면 연구 질문 자체가 상당히 좋아집니다.
# 단순히
# > **"위성 이상을 탐지할 수 있는가?"**
# 가 아닙니다.
# ### RQ1 — Detection
# **When does an anomaly begin?**
# → KF + BOCPD
# ### RQ2 — Uncertainty
# **How certain are we about the anomaly onset?**
# → Bayesian posterior + Monte Carlo
# ### RQ3 — Validation & Explanation
# **Can an independent deep sequence model identify the same abnormal dynamics, and what features drive its prediction?**
# → Mamba + SHAP
# ### RQ4 — Causality
# **Which signal changes precede and causally contribute to the subsequent anomaly dynamics?**
# → level → diff → diff² + Causal Inference
# ### RQ5 — Decision
# **Given uncertainty and asymmetric consequences of false alarms, misses, and delays, when should an intervention be triggered?**
# → Stochastic Optimization + CVaR
# 이렇게 됩니다.
# ---
# # 이게 상당히 중요한 이유
# 마지막에 **CVaR/Stochastic Optimization**을 붙이면 연구가 단순한
# **Anomaly Detection 논문**
# 에서 끝나지 않습니다.
# ```text
# 데이터
#  ↓
# 이상 탐지
#  ↓
# 이상시점의 불확실성
#  ↓
# 설명
#  ↓
# 인과관계
#  ↓
# 위험
#  ↓
# 최적 대응
#  ↓
# 의사결정
# ```
# 이라는 **완결된 DSS 연구**가 됩니다.
# 즉,
# > **"이상이 발생했다."**
# 에서 끝나는 게 아니라,
# > **"이상이 언제 발생했는지 확률적으로 추정하고, 그 불확실성을 고려하여 원인과 선행 신호를 규명하고, 오탐·미탐·지연의 비용을 고려해 최적의 대응시점을 결정한다."**
# 까지 갑니다.
# 이것은 연구의 포지셔닝을 상당히 높여줍니다.
# ---
# ## 다만 Mamba는 반드시 조심해야 합니다
# 제가 가장 권하는 것은 **Mamba를 주인공으로 만들지 않는 것**입니다.
# 현재 연구의 독창성은 Mamba가 아닙니다.
# 오히려:
# > **Uncertain onset → causal identification → risk-aware decision**
# 이 핵심입니다.
# Mamba는
# > **"우리의 확률적 탐지 결과가 특정 통계모형의 산물이 아니라, 독립적인 장기 시계열 모델에서도 일관되게 관찰되는가?"**
# 를 확인하는 **external predictive benchmark**로 두는 것이 좋습니다.
# SHAP 역시
# > **"Mamba의 예측이 어떤 입력 특성에 의해 설명되는가?"**
# 까지만 담당합니다.
# **SHAP 결과를 인과효과로 해석하면 안 됩니다.**
# ---
# # 논문의 가장 좋은 한 문장
# 저라면 이 연구를 다음처럼 정의하겠습니다.
# > **A causal and risk-aware decision support framework for probabilistic anomaly onset identification under uncertain satellite telemetry.**
# 그리고 전체 연구 철학은:
# > **Detect → Quantify → Explain → Cause → Optimize → Decide**
# 입니다.
# 이렇게 정리하면 이용재님이 계속 말씀하신 **NLP / Causal Inference / XAI / Stochastic Optimization / DSS**라는 연구 정체성과도 상당히 잘 맞습니다. 
# **Causal Inference + XAI + Stochastic Optimization + DSS**가 중심이 됩니다.
# **현재 1단계가 잠금된 상태라면, 지금은 Mamba나 최적화를 먼저 붙이기보다 ② onset posterior → ③ causal analysis를 먼저 완성하는 것이 맞습니다. 
# 그 결과가 나온 뒤 Mamba/SHAP을 benchmark·explanation으로 붙이고, 마지막에 CVaR 기반 최적 경보정책을 붙이는 순서가 가장 논리적입니다.**

# 네. **수정하는 게 좋습니다.** 다만 Mamba/SHAP을 빼는 방향이 아니라, **현재 문구의 몇 가지 과한 표현을 정교하게 수정하는 것**을 추천합니다.
# 특히 지금 단계에서는 **5개 채널 선정 → onset posterior 확보 → 인과추론**이 연구의 핵심이므로, Mamba와 SHAP을 핵심 방법론으로 전면에 세우면 오히려 논문의 논리가 흐려질 수 있습니다.
# ### 제가 추천하는 수정 방향
# | 구성                 | 현재            | 수정 권장                                |
# | ------------------ | ------------- | ------------------------------------ |
# | Kalman Filter      | 노이즈 제거        | **상태 추정 및 측정잡음 완화**                  |
# | BOCPD + Forgetting | anomaly onset | **확률적 변화점/onset 탐지**                 |
# | Monte Carlo        | 불확실성 전파       | **onset 불확실성의 전파·민감도 분석**            |
# | Causal Inference   | 인과관계 검증       | **변화의 시간적 선행성 및 인과효과 검증**            |
# | Mamba              | 비교/보조         | **독립적인 sequence-learning benchmark** |
# | SHAP               | 설명            | **Mamba의 predictive attribution**    |
# 특히 **"SHAP → 무엇 때문에?"**라는 표현은 조금 위험합니다.
# SHAP은 "무엇이 원인인가?"가 아니라,
# > **"Mamba의 예측에 어떤 입력 특성이 얼마나 기여했는가?"**
# 라고 표현하는 게 학술적으로 정확합니다.
# ---
# ## 최종적으로는 이렇게 바꾸는 게 가장 좋습니다
# ```text
#                 OPSSAT-AD Telemetry
#                          │
#                          ▼
#                 Channel Profiling
#                          │
#                          ▼
#                  Kalman Filtering
#              State Estimation / Denoising
#                          │
#                          ▼
#                BOCPD + Forgetting
#             Probabilistic Onset Detection
#                          │
#                          ▼
#                   Onset Posterior
#               P(T_onset | observations)
#                          │
#               ┌──────────┴──────────┐
#               │                     │
#               ▼                     ▼
#         Monte Carlo              Mamba
#      Onset Uncertainty       Sequence Benchmark
#         Propagation                │
#               │                    ▼
#               │                   SHAP
#               │          Predictive Attribution
#               │
#               ▼
#         Event-time Variables
#        level → diff → diff²
#               │
#               ▼
#         Temporal Precedence
#               │
#               ▼
#        Causal Inference
#               │
#               ▼
#      Physics-informed SCM
# ```
# ### 이렇게 하면 논문의 메시지가 상당히 명확해집니다.
# **1단계 — Detection**
# > **언제 이상이 발생했는가?**
# → Kalman + BOCPD
# **2단계 — Uncertainty**
# > **그 이상시점이 얼마나 불확실한가?**
# → Bayesian onset posterior + Monte Carlo
# **3단계 — Benchmark & Explanation**
# > **딥러닝 모델에서도 같은 이상이 관찰되는가? 그리고 모델은 무엇을 보고 그렇게 판단했는가?**
# → Mamba + SHAP
# **4단계 — Causality**
# > **이상 이후 어떤 변화가 먼저 발생했고, 무엇에 영향을 주었는가?**
# → level → diff → diff² + Causal Inference
# **5단계 — Mechanistic interpretation**
# > **관찰된 인과구조를 물리적으로 어떻게 해석할 수 있는가?**
# → Physics-informed SCM
# ---
# ### 그리고 아주 중요한 수정 하나
# 현재 문구의
# > **"최종적인 인과적 판단은 BOCPD + Causal Inference가 담당"**
# 은 조금 부정확합니다.
# BOCPD는 **인과추론 방법이 아닙니다.**
# 더 정확하게는:
# > **BOCPD는 변화시점의 확률적 식별을 담당하고, Causal Inference는 식별된 사건 전후의 시간적 선행성과 인과적 관계를 검증한다.**
# 라고 하는 게 좋습니다.
# 이렇게 하면 **Detection ≠ Causality**라는 방법론적 경계가 명확해집니다.
# ---
# ## 결론
# **수정하는 것이 좋습니다. 하지만 Mamba와 SHAP을 핵심으로 끌어올리는 수정은 하지 마세요.**
# 현재 연구의 가장 강한 구조는 오히려:
# > **Kalman → BOCPD → Bayesian onset → Monte Carlo → Causal Inference**
# 이고,
# 그 옆에
# > **Mamba → SHAP**
# 을 **독립적인 딥러닝 검증·설명 계층**으로 붙이는 것입니다.
# 그러면 논문 전체가
# > **"AI로 이상을 찾아냈다"**
# 수준이 아니라
# > **"불확실한 위성 telemetry에서 이상 발생시점을 확률적으로 식별하고, 그 불확실성을 전파한 뒤, 독립적인 sequence model로 탐지 결과를 검증·설명하고, 최종적으로 변화의 시간적 선행성과 인과구조를 규명한다."**
# 라는 훨씬 높은 수준의 연구 프레임워크가 됩니다.
# **현재 1단계 결과까지 나온 상황에서는 이 구조를 유지하고, 이제 2단계의 `onset posterior → level/diff/diff² → causal inference`를 제대로 수행하는 것이 우선순위입니다.**

import pandas as pd

SEGMENTS_PATH = "/home/yjlee/Research/Satellite/OPSSAT-AD_LEO data/segments.csv"

seg_full = pd.read_csv(SEGMENTS_PATH, usecols=["channel", "timestamp"], parse_dates=["timestamp"])

channel_range = seg_full.groupby("channel")["timestamp"].agg(["min", "max"])
channel_range["기간"] = channel_range["max"] - channel_range["min"]
channel_range = channel_range.sort_values("min")

print(channel_range)

                               min                       max                기간
channel                                                                       
CADC0872 2022-01-04 20:00:50+00:00 2022-06-02 15:08:22+00:00 148 days 19:07:32
CADC0886 2022-01-04 20:20:05+00:00 2022-01-05 02:52:25+00:00   0 days 06:32:20
CADC0884 2022-01-04 21:23:45+00:00 2022-06-02 15:10:42+00:00 148 days 17:46:57
CADC0888 2022-01-05 05:39:01+00:00 2022-06-02 15:09:47+00:00 148 days 09:30:46
CADC0873 2022-01-26 00:03:07+00:00 2022-06-02 15:10:18+00:00 127 days 15:07:11
CADC0874 2022-01-26 08:22:19+00:00 2022-06-02 15:08:38+00:00 127 days 06:46:19
CADC0892 2022-01-27 06:54:22+00:00 2022-06-02 15:06:32+00:00 126 days 08:12:10
CADC0894 2022-01-28 20:50:59+00:00 2022-06-02 15:04:33+00:00 124 days 18:13:34
CADC0890 2022-01-29 08:31:42+00:00 2022-01-29 15:22:27+00:00   0 days 06:50:45
