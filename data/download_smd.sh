#!/usr/bin/env bash
#
# download_smd.sh — fetches the SMD (Server Machine Dataset) industrial
# telemetry anomaly-detection benchmark and lays it out under
# data/raw/smd/ in the layout external_validation/loaders.py expects.
#
# Source: Su et al., KDD 2019 (OmniAnomaly).
# This script does not redistribute the dataset — it downloads directly
# from the dataset's public release and does not commit any of it to
# version control (data/raw/ is gitignored).
#
# Usage: bash data/download_smd.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/raw/smd"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

# SMD ("ServerMachineDataset") is distributed inside the OmniAnomaly
# repository rather than as a standalone archive. We fetch a tarball of
# the repository at a pinned ref for reproducibility and extract only the
# ServerMachineDataset/ subtree. If this ref becomes unavailable, consult
# https://github.com/NetManAIOps/OmniAnomaly for the current default
# branch name and update REPO_REF below.
REPO_OWNER="NetManAIOps"
REPO_NAME="OmniAnomaly"
REPO_REF="master"
ARCHIVE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/archive/refs/heads/${REPO_REF}.tar.gz"

echo "=== Downloading SMD (Server Machine Dataset / OmniAnomaly repo) ==="
echo "Target directory: ${RAW_DIR}"

mkdir -p "${RAW_DIR}"

echo "[1/2] Fetching ${REPO_OWNER}/${REPO_NAME}@${REPO_REF} archive ..."
if ! curl -fSL --progress-bar -o "${TMP_DIR}/repo.tar.gz" "${ARCHIVE_URL}"; then
    echo "ERROR: failed to download the OmniAnomaly repository archive from:"
    echo "  ${ARCHIVE_URL}"
    echo "If the default branch has been renamed (e.g. master -> main),"
    echo "update REPO_REF in this script. Otherwise check"
    echo "https://github.com/NetManAIOps/OmniAnomaly directly."
    exit 1
fi

echo "[2/2] Extracting ServerMachineDataset/ ..."
tar -xzf "${TMP_DIR}/repo.tar.gz" -C "${TMP_DIR}"

SRC_DIR="$(find "${TMP_DIR}" -maxdepth 2 -type d -name "ServerMachineDataset" | head -n 1)"
if [[ -z "${SRC_DIR}" ]]; then
    echo "ERROR: could not locate ServerMachineDataset/ inside the"
    echo "downloaded archive. The upstream repository layout may have"
    echo "changed. Inspect ${TMP_DIR} manually and update this script's"
    echo "directory-discovery logic accordingly."
    exit 1
fi

for sub in train test test_label interpretation_label; do
    if [[ -d "${SRC_DIR}/${sub}" ]]; then
        mkdir -p "${RAW_DIR}/${sub}"
        cp "${SRC_DIR}/${sub}"/*.txt "${RAW_DIR}/${sub}/" 2>/dev/null || true
    fi
done

# --- Verification --------------------------------------------------------
N_TRAIN=$(find "${RAW_DIR}/train" -name 'machine-*.txt' 2>/dev/null | wc -l | tr -d ' ')
N_TEST=$(find "${RAW_DIR}/test" -name 'machine-*.txt' 2>/dev/null | wc -l | tr -d ' ')
N_LABELS=$(find "${RAW_DIR}/test_label" -name 'machine-*.txt' 2>/dev/null | wc -l | tr -d ' ')
N_INTERP=$(find "${RAW_DIR}/interpretation_label" -name 'machine-*.txt' 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "=== SMD download complete ==="
echo "  train/                 : ${N_TRAIN} machine files"
echo "  test/                  : ${N_TEST} machine files"
echo "  test_label/             : ${N_LABELS} machine files"
echo "  interpretation_label/   : ${N_INTERP} machine files"

if [[ "${N_TRAIN}" -lt 28 || "${N_TEST}" -lt 28 || "${N_LABELS}" -lt 28 ]]; then
    echo ""
    echo "WARNING: expected 28 machines but found fewer in one or more"
    echo "subdirectories. This may indicate a partial download or an"
    echo "upstream layout change. Do not proceed with external_validation"
    echo "until this is resolved."
    exit 1
fi

echo ""
echo "Please cite the OmniAnomaly paper (Su et al., KDD 2019) directly"
echo "in addition to this repository's paper. See data/README.md for the"
echo "full citation and license notes."
