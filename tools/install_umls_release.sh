#!/bin/sh
# Install a UMLS full-release archive into an RRF subset (SAB in {CPT, HCPT, HCPCS,
# SNOMEDCT_US}) via NLM's own MetamorphoSys tool, run in its documented headless
# "Batch Subset" mode. tools/build_umls_crosswalk.py consumes the RRF output this
# script produces; it never reads the raw archive or the proprietary .nlm files
# directly -- MetamorphoSys is NLM's own official, licensed tool for that, and the
# only supported path.
#
# Usage: tools/install_umls_release.sh <archive.zip> <scratch_dir>
#
# <scratch_dir>/rrf_output/ holds the result (MRCONSO.RRF etc.) when this exits 0.
# Everything else under <scratch_dir> is disposable working state this script cleans
# up on success -- the archive itself is the only thing that must survive, and this
# script never touches it beyond reading it.
#
# Reverse-engineered from MetamorphoSys's own compiled classes and config files
# (its batch-mode system properties are not documented in the release's own README):
#   -Djpf.boot.config=etc/subset.boot.properties   selects the BATCH plugin, not the
#                                                    GUI one boot.properties defaults to
#   -Dinput.uri / -Doutput.uri / -Dmmsys.config.uri  are PLAIN FILESYSTEM PATHS, not
#                                                    file:// URIs (confirmed by reading
#                                                    AbstractPropertyFileConfiguration.
#                                                    open() / NLMFileMetamorphoSysInput
#                                                    Stream.validateUri(), which call
#                                                    `new File(uriString)` directly)
# and one real bug in MetamorphoSys itself: RRFMetadataInputStream.openSourceFile()
# calls .listFiles() on every installPaths[i] entry when asked for "release.dat",
# including .nlm archive-file entries (not directories), throwing a
# NullPointerException. Worked around by placing a copy of the release's own
# config/<release>/release.dat directly in the source directory, so the directory
# branch satisfies the lookup before the loop ever reaches a .nlm entry.
set -eu

ARCHIVE="${1:?usage: install_umls_release.sh <archive.zip> <scratch_dir>}"
SCRATCH="${2:?usage: install_umls_release.sh <archive.zip> <scratch_dir>}"

mkdir -p "$SCRATCH"
SCRATCH="$(cd "$SCRATCH" && pwd)"

# The archive's own top-level directory name (e.g. "2026AA-full") -- read from the
# zip, never assumed, so this script works against any release edition unchanged.
RELEASE_DIR="$(python3 -c "
import zipfile
z = zipfile.ZipFile('$ARCHIVE')
top = sorted({n.split('/')[0] for n in z.namelist() if '/' in n})
print(top[0])
")"

SRC="$SCRATCH/$RELEASE_DIR/source"
MMSYS="$SCRATCH/$RELEASE_DIR/mmsys"
OUT="$SCRATCH/rrf_output"
# Idempotent on re-run against a REUSED scratch dir: a stale MRCONSO.RRF (or partial
# source/mmsys extraction) left behind by an earlier failed or older-release run must
# never be silently read as this run's output, or a genuine failure this time could
# be masked by yesterday's success.
rm -rf "$SRC" "$MMSYS" "$OUT"
mkdir -p "$SRC" "$MMSYS" "$OUT"

echo "Extracting mmsys.zip and the release's .nlm files ..."
python3 -c "
import zipfile
z = zipfile.ZipFile('$ARCHIVE')
prefix = '$RELEASE_DIR/'
members = [n for n in z.namelist()
          if n.startswith(prefix) and (n.endswith('.nlm') or n.endswith('mmsys.zip')
                                       or n.endswith('release.dat'))]
z.extractall('$SCRATCH/_extract', members=members)
"
mv "$SCRATCH/_extract/$RELEASE_DIR"/*.nlm "$SRC"/ 2>/dev/null || true
mv "$SCRATCH/_extract/$RELEASE_DIR"/release.dat "$SRC"/ 2>/dev/null || true
unzip -q "$SCRATCH/_extract/$RELEASE_DIR/mmsys.zip" -d "$MMSYS"
rm -rf "$SCRATCH/_extract"

# The config subdirectory is named after the RELEASE VERSION (e.g. "2026AA"), which
# is not necessarily the archive's own top-level folder name ("2026AA-full") -- found
# dynamically, excluding mmsys.zip's other known fixed subdirectories.
CONFIG_DIR="$(find "$MMSYS/config" -mindepth 1 -maxdepth 1 -type d \
             ! -name dict ! -name icons | head -1)"
if [ -z "$CONFIG_DIR" ]; then
    echo "Could not locate the release config directory under $MMSYS/config" >&2
    exit 1
fi
if [ ! -f "$SRC/release.dat" ] && [ -f "$CONFIG_DIR/release.dat" ]; then
    # Workaround for the release.dat NPE described above -- the release's own
    # bundled config copy is authoritative for this purpose.
    cp "$CONFIG_DIR/release.dat" "$SRC/release.dat"
fi

echo "Building the batch subset config (Level 0 + SNOMEDCT_US preset, corrected to " \
     "INCLUDE rather than exclude its selected_sources) ..."
python3 -c "
from pathlib import Path
preset = Path('$CONFIG_DIR/user.b.prop').read_text()
lines = []
for line in preset.splitlines():
    if line.startswith('gov.nih.nlm.umls.mmsys.filter.SourceListFilter.remove_selected_sources='):
        lines.append('gov.nih.nlm.umls.mmsys.filter.SourceListFilter.remove_selected_sources=false')
    elif line.startswith('gov.nih.nlm.umls.mmsys.filter.SourceListFilter.selected_sources='):
        lines.append('gov.nih.nlm.umls.mmsys.filter.SourceListFilter.selected_sources='
                     'CPT|CPT;CPTSP|CPT;HCPT|CPT;HCDT|HCPCS;HCPCS|HCPCS;SNOMEDCT_US|SNOMEDCT')
    elif line.startswith('meta_destination_uri=') or line.startswith('umls_destination_uri='):
        continue   # set via -Doutput.uri instead
    else:
        lines.append(line)
Path('$SCRATCH/subset.prop').write_text('\n'.join(lines) + '\n')
"

echo "Running MetamorphoSys (headless batch subset) ..."
cd "$MMSYS"
JAVA_HOME="$MMSYS/jre/linux"
CLASSPATH=".:lib/jpf-boot.jar"
export CLASSPATH
"$JAVA_HOME/bin/java" \
    -Dfile.encoding=UTF-8 -Xms1000M -Xmx2000M -client \
    -Djava.awt.headless=true \
    -Dscript_type=.sh -Dunzip.native=true -Dunzip.path=/usr/bin/unzip \
    -Djpf.boot.config=etc/subset.boot.properties \
    -Dinput.uri="$SRC/" \
    -Doutput.uri="$OUT/" \
    -Dmmsys.config.uri="$SCRATCH/subset.prop" \
    org.java.plugin.boot.Boot

if [ ! -s "$OUT/MRCONSO.RRF" ]; then
    echo "MetamorphoSys did not produce a non-empty MRCONSO.RRF -- failing loudly." >&2
    exit 1
fi

echo "Cleaning up intermediate extraction (source .nlm files, MetamorphoSys install) ..."
rm -rf "$SCRATCH/$RELEASE_DIR"

echo "RRF subset ready at $OUT (run tools/build_umls_crosswalk.py --release '$OUT' next)."
