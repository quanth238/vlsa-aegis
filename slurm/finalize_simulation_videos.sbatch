#!/usr/bin/env bash
#SBATCH --job-name=vlsa-video-finalize
#SBATCH --partition=main
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --exclude=worker-0,worker-3
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/quanth/slurm_logs/%x-%j.out

set -euo pipefail

VIDEO_ROOT=${VIDEO_ROOT:?VIDEO_ROOT is required}
OUTPUT_DIR_NAME=${OUTPUT_DIR_NAME:-portable-v1}
FFMPEG_BIN=${FFMPEG_BIN:-/mnt/data/quanth/venvs/safety_vla/main/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2}

die() { printf '%s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || die "video finalization requires Slurm"
case "${SLURMD_NODENAME:-}" in login*|login-restricted*) die "login node forbidden" ;; esac
case "$VIDEO_ROOT" in /mnt/data/quanth/experiments/*) ;; *) die "unsafe VIDEO_ROOT" ;; esac
case "$OUTPUT_DIR_NAME" in ""|*/*|*[!A-Za-z0-9._-]*) die "unsafe OUTPUT_DIR_NAME" ;; esac
[[ -x "$FFMPEG_BIN" ]] || die "ffmpeg binary is missing"
gpu_line=$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader,nounits)
[[ $(printf '%s\n' "$gpu_line" | wc -l) -eq 1 && "$gpu_line" == *H100* ]] || die "requires exactly one H100"

OUTPUT_ROOT=$VIDEO_ROOT/$OUTPUT_DIR_NAME
mkdir "$OUTPUT_ROOT" || die "immutable portable-video root exists"
MANIFEST=$OUTPUT_ROOT/manifest.txt
temporary=$(mktemp "$OUTPUT_ROOT/.manifest.XXXXXX")
trap 'rm -f "$temporary"' EXIT

count=0
for case_root in "$VIDEO_ROOT"/case-*; do
  [[ -d "$case_root" && ! -L "$case_root" ]] || continue
  input=$case_root/episode.mp4
  [[ -f "$input" && ! -L "$input" ]] || die "source episode video is missing"
  case_name=$(basename "$case_root")
  output=$OUTPUT_ROOT/$case_name.mp4
  [[ ! -e "$output" ]] || die "portable output already exists"
  "$FFMPEG_BIN" -nostdin -hide_banner -loglevel error \
    -i "$input" \
    -c:v libx264 -preset medium -crf 28 \
    -pix_fmt yuv420p -movflags +faststart -an \
    "$output"
  # A successful full decode is mandatory; container existence is insufficient.
  "$FFMPEG_BIN" -nostdin -v error -i "$output" -f null -
  bytes=$(wc -c <"$output" | tr -d ' ')
  digest=$(sha256sum "$output" | awk '{print $1}')
  printf '%s\t%s\t%s\n' "$case_name.mp4" "$bytes" "$digest" >>"$temporary"
  count=$((count + 1))
done

[[ $count -gt 0 ]] || die "no case videos were found"
mv "$temporary" "$MANIFEST"
trap - EXIT
cat "$MANIFEST"
