#!/usr/bin/env bash
# Download the exact corrected release. Does not select images or run a model.
set -euo pipefail
root="${1:?Usage: bash hpc/download_dermamnist.sh DATASET_ROOT}"
if [[ "$root" == /scratch/* ]]; then
  [[ -n "${SLURM_JOB_ID:-}" && "$(hostname -s)" =~ ^bun[0-9]{3}$ ]] || {
    echo 'Bunya downloads and file processing require Slurm.' >&2; exit 2;
  }
fi
umask 077
mkdir -p "$root"/{raw,provenance,manifests,prepared}
cd "$root"
log="provenance/download-$(date -u +%Y%m%dT%H%M%SZ).log"
fetch() {
  local name="$1" checksum="$2" code
  if [[ -e "raw/$name" ]]; then
    printf '%s  %s\n' "$checksum" "raw/$name" | md5sum -c - | tee -a "$log"
    return
  fi
  if curl --location --fail --show-error --silent --retry 2 --retry-delay 20 \
      --connect-timeout 20 --max-time 1800 --speed-limit 1024 --speed-time 60 \
      --continue-at - --output "raw/$name.part" \
      "https://zenodo.org/records/12739457/files/$name?download=1" 2>>"$log"; then
    printf '%s  %s\n' "$checksum" "raw/$name.part" | md5sum -c - | tee -a "$log"
    mv "raw/$name.part" "raw/$name"
  else
    code=$?
    printf 'DOWNLOAD_FAILED file=%s curl_exit=%s; partial file retained; dataset not ready\n' "$name" "$code" | tee -a "$log" >&2
    return "$code"
  fi
}
fetch DermaMNIST-C.csv 1c54bc9f483e97a9d7dbbde076a45900
fetch dermamnist_corrected_224.npz 84920fb70c83b234c295b6f0d4ae2bc0
sha256sum raw/DermaMNIST-C.csv raw/dermamnist_corrected_224.npz > provenance/SHA256SUMS
printf 'DOWNLOAD_VERIFIED; run prepare_dermamnist.py without --metadata-only next.\n' | tee -a "$log"
