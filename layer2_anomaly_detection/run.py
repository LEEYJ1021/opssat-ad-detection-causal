"""
layer2_anomaly_detection.run
=============================
층2(Anomaly Detection) 전체 파이프라인 엔트리 포인트.

    python -m layer2_anomaly_detection.run [--data PATH] [--out PATH]

data/raw/segments.csv (layer1_signal_estimation이 산출한, 또는 원본
OPS-SAT-AD segments.csv)가 있으면 그것으로 실제 파이프라인을 처음부터
끝까지 실행하고, 없으면 합성 데이터로 스모크 테스트만 수행한다.

단계:
    (A) 채널 프로파일링 + 계층적 베이즈 수축         -> _shared.fit_channel_profiles
    (B) BOCPD(forgetting) 전 채널 탐지                -> bocpd_forgetting.detect_all_channels
    (C) MCC/Youden 채널 스코프 확정                   -> channel_scoping.build_channel_scope
    (D) 전체 데이터 MCC 부트스트랩 CI                 -> bootstrap_ci.bootstrap_channel_mcc
    (E) 4-조합 ablation (train 선택 -> test 평가)      -> ablation_mixture_forgetting.run_ablation
    (F) train 재적합 -> test 재평가                    -> triple_verification.train_fit_test_eval
    (G) test-only MCC 부트스트랩 CI + 전체 대비 CI 대조 -> triple_verification.compare_full_vs_test_ci
    (H) 890 서술 정정 + 최종 스코프 잠금                -> triple_verification.lock_channel_scope

results/layer2/ 아래에 다음을 저장한다:
    channel_scope.json          — (C)+(H) 최종 결정표 + 5채널 스코프
    ablation_grid.csv           — (E) 4-조합 grid
    bootstrap_ci.csv            — (D) 전체 데이터 MCC 부트스트랩 CI
    bootstrap_ci_test_only.csv  — (G) test-only MCC 부트스트랩 CI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _shared import fit_channel_profiles, load_segments, section
from bocpd_forgetting import detect_all_channels
from channel_scoping import build_channel_scope, to_json_summary
from bootstrap_ci import bootstrap_channel_mcc
from ablation_mixture_forgetting import run_ablation
from triple_verification import train_fit_test_eval, compare_full_vs_test_ci, lock_channel_scope


def main(data_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    section("0. 데이터 로드")
    seg, channels, is_synthetic = load_segments(data_path)
    print(f"{'[합성 데이터 — 스모크 테스트]' if is_synthetic else '[실 데이터]'} "
          f"segments={seg.shape}, channels={channels}")

    section("(A)+(B) 채널 프로파일링 + BOCPD(forgetting) 탐지")
    profiles, r_shrunk, q_shrunk, quant_floor = fit_channel_profiles(seg, channels)
    detect_out = detect_all_channels(seg, channels, r_shrunk, q_shrunk, quant_floor, use_forgetting=True)
    calib = detect_out["calibration"]
    print(calib.to_string(index=False))

    section("(C) MCC/Youden 채널 스코프")
    # channel_scoping은 n_anomaly_segments/n_normal_segments 컬럼명을 기대함
    calib_for_scope = calib.rename(columns={})
    scope = build_channel_scope(calib_for_scope)
    print(scope.to_string(index=False))

    section("(D) 전체 데이터 MCC 부트스트랩 CI")
    boot_full = bootstrap_channel_mcc(detect_out["onset_results"], channels)
    print(boot_full.to_string(index=False))
    boot_full.to_csv(out_dir / "bootstrap_ci.csv", index=False)
    print(f"\n[저장] {out_dir / 'bootstrap_ci.csv'}")

    if "train" not in seg.columns:
        print("\n[안내] 'train' 컬럼이 없어 (E)~(H) train/test 3중 검증 단계는 생략합니다 "
              "(합성 스모크 데이터에는 포함되어 있으므로, 보통 실 데이터에 스키마 문제가 있을 때만 발생).")
        locked = scope.assign(scope_reason_final=scope["scope_reason"])
    else:
        section("(E) 4-조합 ablation (train 선택 -> test 평가)")
        ablation_out = run_ablation(seg, channels)
        print(ablation_out["ablation_grid"].to_string(index=False))
        print(f"\nbest_combo (train-only) = {ablation_out['best_combo']}")
        ablation_out["ablation_grid"].to_csv(out_dir / "ablation_grid.csv", index=False)
        print(f"[저장] {out_dir / 'ablation_grid.csv'}")

        section("(F) train 재적합 -> test 재평가")
        train_test_compare = train_fit_test_eval(seg, channels, calib)
        print(train_test_compare.to_string(index=False))

        section("(G) test-only MCC 부트스트랩 CI")
        test_mask = ~seg["train"].astype(bool)
        test_onset = detect_out["onset_results"].merge(
            seg.loc[test_mask, ["segment"]].drop_duplicates(), on="segment", how="inner"
        )
        boot_test = bootstrap_channel_mcc(test_onset, channels)
        print(boot_test.to_string(index=False))
        boot_test = boot_test.rename(columns={"n_anomaly": "n_anom_test", "n_normal": "n_norm_test"})
        boot_test.to_csv(out_dir / "bootstrap_ci_test_only.csv", index=False)
        print(f"[저장] {out_dir / 'bootstrap_ci_test_only.csv'}")

        cmp = compare_full_vs_test_ci(boot_full, boot_test)
        print("\n[전체 vs test-only CI 대조]")
        print(cmp[["channel", "mcc_median_full", "mcc_ci_low_test", "mcc_ci_high_test",
                    "full_median_in_test_ci"]].to_string(index=False))

        section("(H) 890 서술 정정 + 최종 스코프 잠금")
        locked = lock_channel_scope(scope)
        print(locked.to_string(index=False))

    section("결과 저장")
    summary = to_json_summary(scope)
    summary["locked_scope_reasons"] = locked.set_index("channel")["scope_reason_final"].to_dict()
    with open(out_dir / "channel_scope.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[저장] {out_dir / 'channel_scope.json'}")

    scope.to_csv(out_dir / "channel_scope_full_table.csv", index=False)
    print(f"[저장] {out_dir / 'channel_scope_full_table.csv'}")

    included = scope[scope["include_in_stage2"]]["channel"].tolist()
    section("완료")
    print(f"층3(causal_analysis)으로 넘길 최종 채널 스코프 ({len(included)}개): {included}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="layer2_anomaly_detection full pipeline")
    parser.add_argument("--data", type=Path, default=Path("data/raw/segments.csv"),
                         help="OPS-SAT-AD segments.csv 경로 (없으면 합성 데이터로 스모크 테스트)")
    parser.add_argument("--out", type=Path, default=Path("../results/layer2"),
                         help="결과 저장 디렉토리")
    args = parser.parse_args()
    main(args.data, args.out)
