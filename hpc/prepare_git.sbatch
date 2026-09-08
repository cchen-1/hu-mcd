#!/usr/bin/env bash
#SBATCH --job-name=humcd-git-prepare
#SBATCH --account=a_ai_collab
#SBATCH --partition=general
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/user/uqcche38/humcd-git-prepare-%j.out
#SBATCH --error=/scratch/user/uqcche38/humcd-git-prepare-%j.err
set -euo pipefail
sha="${1:-${HUMCD_PREPARE_SHA:-}}"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected full lowercase commit SHA' >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname -s)" =~ ^bun[0-9]{3}$ ]] || {
  echo 'Requires a Slurm allocation on a Bunya compute node' >&2; exit 2;
}
repo=https://github.com/cchen-1/hu-mcd.git
root=/scratch/user/uqcche38/hu-mcd-git
mkdir -p "$root/releases"
# Serialize publication; never update an existing release in place.
exec 9>"$root/prepare.lock"
flock -w 60 9
release="$root/releases/$sha"
if [[ -e "$release" ]]; then
  echo "Release already exists; refusing to replace it: $release" >&2
  exit 3
fi
tmp="$(mktemp -d "$root/releases/.prepare.XXXXXXXX")"
trap 'rm -rf -- "$tmp"' EXIT
export GIT_TERMINAL_PROMPT=0
export GIT_LFS_SKIP_SMUDGE=1
git init "$tmp/code"
git -C "$tmp/code" remote add origin "$repo"
timeout 180s git -C "$tmp/code" fetch --no-tags origin refs/heads/reproduction
git -C "$tmp/code" cat-file -e "$sha^{commit}"
git -C "$tmp/code" merge-base --is-ancestor "$sha" FETCH_HEAD
git -C "$tmp/code" checkout --detach "$sha"
[[ "$(git -C "$tmp/code" rev-parse HEAD)" == "$sha" ]]
[[ -z "$(git -C "$tmp/code" status --porcelain)" ]]
# Dependencies, submodules, LFS assets and experiment execution are separate steps.
printf 'repository=%s\nbranch=reproduction\ncommit=%s\njob_id=%s\nhost=%s\nprepared_utc=%s\n' \
  "$repo" "$sha" "$SLURM_JOB_ID" "$(hostname -s)" "$(date -u +%FT%TZ)" > "$tmp/provenance.txt"
mv -- "$tmp" "$release"
trap - EXIT
cat "$release/provenance.txt"
echo "PASS: prepared $release/code"
