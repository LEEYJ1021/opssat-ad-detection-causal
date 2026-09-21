#!/usr/bin/env bash
#
# download_smap_msl.sh — fetches the SMAP/MSL (Telemanom) spacecraft
# telemetry anomaly-detection benchmark and lays it out under
# data/raw/smap_msl/ in the layout external_validation/loaders.py expects.
#
# Source: Hundman et al., KDD 2018 (Telemanom).
# This script does not redistribute the dataset — it downloads directly
# from the dataset's public release and does not commit any of it to
# version control (data/raw/ is gitignored).
#
# Usage: bash data/download_smap_msl.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/raw/smap_msl"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

# Public archive containing train/ and test/ .npy files (as used by the
# Telemanom repository's own setup instructions). If this URL becomes
# unavailable, consult https://github.com/khundman/telemanom for the
# current canonical location — the maintainers have occasionally moved
# the hosted archive.
DATA_ARCHIVE_URL="https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABELS_URL="https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"

echo "=== Downloading SMAP/MSL (Telemanom) benchmark ==="
echo "Target directory: ${RAW_DIR}"

mkdir -p "${RAW_DIR}"

# --- 1. Ground-truth anomaly labels -----------------------------------
echo "[1/3] Fetching labeled_anomalies.csv ..."
if ! curl -fsSL -o "${RAW_DIR}/labeled_anomalies.csv" "${LABELS_URL}"; then
    echo "ERROR: failed to download labeled_anomalies.csv from:"
    echo "  ${LABELS_URL}"
    echo "The upstream repository may have moved or renamed this file."
    echo "Check https://github.com/khundman/telemanom for the current path."
    exit 1
fi

# --- 2. Train/test telemetry archive ------------------------------------
echo "[2/3] Fetching train/test telemetry archive (this may take a few minutes) ..."
if ! curl -fSL --progress-bar -o "${TMP_DIR}/data.zip" "${DATA_ARCHIVE_URL}"; then
    echo "ERROR: failed to download the SMAP/MSL data archive from:"
    echo "  ${DATA_ARCHIVE_URL}"
    echo "If this URL is no longer valid, check"
    echo "https://github.com/khundman/telemanom for the current hosted"
    echo "location, then update DATA_ARCHIVE_URL in this script."
    exit 1
fi

echo "[3/3] Extracting to ${RAW_DIR} ..."
unzip -q "${TMP_DIR}/data.zip" -d "${TMP_DIR}/extracted"

# The upstream archive layout has historically been:
#   data/train/<chan_id>.npy
#   data/test/<chan_id>.npy
# but has also appeared without the top-level "data/" wrapper depending on
# release version, so we search for it rather than assuming a fixed depth.
SRC_TRAIN_DIR="$(find "${TMP_DIR}/extracted" -type d -name train | head -n 1)"
SRC_TEST_DIR="$(find "${TMP_DIR}/extracted" -type d -name test | head -n 1)"

if [[ -z "${SRC_TRAIN_DIR}" || -z "${SRC_TEST_DIR}" ]]; then
    echo "ERROR: could not locate train/ and test/ subdirectories inside"
    echo "the downloaded archive. The upstream release layout may have"
    echo "changed. Inspect ${TMP_DIR}/extracted manually and update this"
    echo "script's directory-discovery logic accordingly."
    exit 1
fi

mkdir -p "${RAW_DIR}/train" "${RAW_DIR}/test"
cp "${SRC_TRAIN_DIR}"/*.npy "${RAW_DIR}/train/"
cp "${SRC_TEST_DIR}"/*.npy "${RAW_DIR}/test/"

# --- Verification --------------------------------------------------------
N_TRAIN=$(find "${RAW_DIR}/train" -name '*.npy' | wc -l | tr -d ' ')
N_TEST=$(find "${RAW_DIR}/test" -name '*.npy' | wc -l | tr -d ' ')
N_LABELS=$(($(wc -l < "${RAW_DIR}/labeled_anomalies.csv") - 1))  # minus header

echo ""
echo "=== SMAP/MSL download complete ==="
echo "  train/ : ${N_TRAIN} channel files"
echo "  test/  : ${N_TEST} channel files"
echo "  labeled_anomalies.csv : ${N_LABELS} labeled channel-rows"

if [[ "${N_TEST}" -lt 82 || "${N_LABELS}" -lt 82 ]]; then
    echo ""
    echo "WARNING: expected 82 channels (55 SMAP + 27 MSL) but found fewer."
    echo "This may indicate a partial download or an upstream layout change."
    echo "Do not proceed with external_validation until this is resolved."
    exit 1
fi

echo ""
echo "Please cite the Telemanom paper (Hundman et al., KDD 2018) directly"
echo "in addition to this repository's paper. See data/README.md for the"
echo "full citation and license notes."
