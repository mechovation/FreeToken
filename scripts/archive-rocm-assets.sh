#!/usr/bin/env bash
set -euo pipefail

action="${1:-export}"
archive_dir="${2:-./archives/rocm}"
image="${ROCM_DEPS_IMAGE:-freetoken:rocm-deps}"
archive="${archive_dir}/freetoken-rocm-deps.tar.zst"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

case "$action" in
  export)
    mkdir -p "$archive_dir"
    if [[ -e "$archive" ]]; then
      printf 'Archive already exists: %s\n' "$archive" >&2
      exit 1
    fi
    docker build --target rocm-deps -f "$root/docker/Dockerfile.rocm" -t "$image" "$root"
    temp_archive="$(mktemp "${archive}.XXXXXX")"
    trap 'rm -f "$temp_archive"' EXIT
    docker image save "$image" | zstd -T2 -3 -o "$temp_archive" -f
    mv "$temp_archive" "$archive"
    trap - EXIT
    (cd "$archive_dir" && sha256sum freetoken-rocm-deps.tar.zst > SHA256SUMS)
    docker image inspect "$image" > "${archive_dir}/image.json"
    printf 'Archived %s to %s\n' "$image" "$archive"
    ;;
  restore)
    (cd "$archive_dir" && sha256sum -c SHA256SUMS)
    zstd -dc "$archive" | docker image load
    printf 'Build with ROCM_DEPS_IMAGE=%s to use the restored dependencies.\n' "$image"
    ;;
  *)
    printf 'Usage: bash %s {export|restore} [archive-directory]\n' "$0" >&2
    exit 2
    ;;
esac
