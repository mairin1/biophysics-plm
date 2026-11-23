#!/usr/bin/env bash
# Download and optionally extract the METL data release from Zenodo record 24039302.
#
# Usage:
#   scripts/metl_global/fetch_zenodo_24039302.sh bootstrap --extract
#   scripts/metl_global/fetch_zenodo_24039302.sh raw-global --extract
#   scripts/metl_global/fetch_zenodo_24039302.sh processed-global --extract
#   scripts/metl_global/fetch_zenodo_24039302.sh reproduce --extract

set -Eeuo pipefail

RECORD="24039302"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
ARCHIVE_DIR="${ROOT}/external/zenodo-${RECORD}/archives"
EXTRACT_DIR="${ROOT}/external/zenodo-${RECORD}/extracted"

MODE="bootstrap"
if (($# > 0)); then
    MODE="$1"
    shift
fi

EXTRACT=false
while (($# > 0)); do
    case "$1" in
        --extract) EXTRACT=true ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

case "$MODE" in
    bootstrap)
        FILES=(
            "clean_pdb_2025-02-13.tar.gz"
            "metl-sim_2025-02-13.tar.gz"
        )
        ;;
    raw-global)
        FILES=("raw-global.tar.gz")
        ;;
    processed-global)
        FILES=("processed-global.tar.gz")
        ;;
    reproduce)
        FILES=(
            "clean_pdb_2025-02-13.tar.gz"
            "metl-sim_2025-02-13.tar.gz"
            "raw-global.tar.gz"
            "processed-global.tar.gz"
        )
        ;;
    *)
        cat >&2 <<USAGE
Usage:
  $0 bootstrap --extract
  $0 raw-global --extract
  $0 processed-global --extract
  $0 reproduce --extract
USAGE
        exit 2
        ;;
esac

expected_md5() {
    case "$1" in
        clean_pdb_2025-02-13.tar.gz) echo "e3a73b6567d0b2c4f2c3268f583447bc" ;;
        metl-sim_2025-02-13.tar.gz) echo "7ea8b7dd3ba1b005e71dc500b00f2248" ;;
        raw-global.tar.gz) echo "74aad528e294433b0c55e0b8c5b75213" ;;
        processed-global.tar.gz) echo "02221f43363c23b06b156900fbd86957" ;;
        *)
            echo "No checksum registered for $1" >&2
            exit 2
            ;;
    esac
}

download_one() {
    local name="$1"
    local expected
    local archive
    local partial
    local observed
    local url

    expected="$(expected_md5 "$name")"
    archive="${ARCHIVE_DIR}/${name}"
    partial="${archive}.part"
    url="https://zenodo.org/api/records/${RECORD}/files/${name}/content"

    mkdir -p "$ARCHIVE_DIR"

    if [[ -f "$archive" ]]; then
        observed="$(md5sum "$archive" | awk '{print $1}')"
        if [[ "$observed" == "$expected" ]]; then
            echo "Verified existing archive: $archive"
            return
        fi

        echo "Removing invalid archive: $archive" >&2
        rm -f "$archive"
    fi

    echo "Downloading: $name"
    echo "URL: $url"

    # Continue a prior incomplete transfer when <archive>.part exists.
    curl \
        --fail \
        --location \
        --retry 8 \
        --retry-delay 5 \
        --continue-at - \
        --output "$partial" \
        "$url"

    observed="$(md5sum "$partial" | awk '{print $1}')"
    if [[ "$observed" != "$expected" ]]; then
        echo "Checksum mismatch for $name" >&2
        echo "Expected: $expected" >&2
        echo "Observed: $observed" >&2
        echo "Deleting corrupt partial download." >&2
        rm -f "$partial"
        exit 1
    fi

    mv "$partial" "$archive"
    echo "Verified: $archive"
}

extract_one() {
    local name="$1"
    local archive="${ARCHIVE_DIR}/${name}"
    local stem="${name%.tar.gz}"
    local destination="${EXTRACT_DIR}/${stem}"
    local marker="${destination}/.zenodo-${RECORD}-extracted"

    mkdir -p "$EXTRACT_DIR"

    echo "Recording archive contents: ${archive}.contents.txt"
    tar -tzf "$archive" > "${archive}.contents.txt"

    if [[ -f "$marker" ]]; then
        echo "Already extracted: $destination"
        return
    fi

    echo "Extracting: $archive"
    mkdir -p "$destination"
    tar \
        --extract \
        --gzip \
        --file "$archive" \
        --directory "$destination" \
        --no-same-owner \
        --no-same-permissions

    touch "$marker"
    echo "Extracted: $destination"
}

for file in "${FILES[@]}"; do
    download_one "$file"

    if [[ "$EXTRACT" == true ]]; then
        extract_one "$file"
    fi
done

echo
echo "Archives:  $ARCHIVE_DIR"
echo "Extracted: $EXTRACT_DIR"
