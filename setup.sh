#!/usr/bin/env bash
#
# setup.sh — one-command environment bootstrap for the Podiatry Clean-Claim Scrubber.
#
# It intelligently checks what's already installed and only installs what's missing,
# then starts the pipeline. Two modes:
#
#   Docker mode (default) : ensures Docker + Compose, then `docker compose up --build`.
#                           The container image provisions Python 3.11, poppler, all
#                           libraries, pre-downloads models, and Qdrant runs alongside.
#   Native mode (--native): installs Python + libs + poppler on the host, runs Qdrant
#                           locally (or on-disk store), and runs the pipeline directly.
#
# Usage:
#   ./setup.sh                 # Docker mode: build + start processing
#   ./setup.sh --native        # host-native install + run (no Docker)
#   ./setup.sh --check         # detect & report only — installs nothing, starts nothing
#   ./setup.sh --no-start      # set everything up but don't start processing
#   ./setup.sh --rebuild       # force-rebuild the Qdrant index on start
#   ./setup.sh --notes DIR     # process PDFs from DIR (native mode)
#   ./setup.sh -h | --help
#
# Safe to re-run: every step is idempotent. Compatible with macOS (bash 3.2) and Linux.

set -euo pipefail

# ───────────────────────── pretty logging ─────────────────────────
if [ -t 1 ]; then
  C_RESET=$'\033[0m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_R=$'\033[31m'; C_B=$'\033[36m'
else
  C_RESET=''; C_G=''; C_Y=''; C_R=''; C_B=''
fi
step() { printf "\n%s==>%s %s\n" "$C_B" "$C_RESET" "$1"; }
ok()   { printf "  %s[ok]%s   %s\n" "$C_G" "$C_RESET" "$1"; }
info() { printf "  %s[..]%s   %s\n" "$C_B" "$C_RESET" "$1"; }
warn() { printf "  %s[warn]%s %s\n" "$C_Y" "$C_RESET" "$1"; }
die()  { printf "  %s[err]%s  %s\n" "$C_R" "$C_RESET" "$1" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# ───────────────────────── flags ─────────────────────────
MODE="docker"; DO_START=1; REBUILD=0; CHECK_ONLY=0; NOTES_DIR_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --native)   MODE="native" ;;
    --docker)   MODE="docker" ;;
    --check)    CHECK_ONLY=1 ;;
    --no-start) DO_START=0 ;;
    --rebuild)  REBUILD=1 ;;
    --notes)    shift; NOTES_DIR_OVERRIDE="${1:-}" ;;
    -h|--help)  grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1 (use -h for help)" ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ───────────────────────── OS / package manager ─────────────────────────
OS="$(uname -s)"; ARCH="$(uname -m)"; PKG=""; SUDO=""
detect_os() {
  step "Detecting platform"
  case "$OS" in
    Darwin) PKG="brew" ;;
    Linux)
      if   have apt-get; then PKG="apt"
      elif have dnf;     then PKG="dnf"
      elif have yum;     then PKG="yum"
      else PKG="none"; fi ;;
    *) die "unsupported OS: $OS" ;;
  esac
  if [ "$(id -u)" != "0" ] && have sudo; then SUDO="sudo"; fi
  ok "OS=$OS  arch=$ARCH  package-manager=$PKG"
}

pkg_install() {  # pkg_install <brew-name> <apt-name> [dnf/yum-name]
  local brewp="$1" aptp="$2" rpmp="${3:-$2}"
  case "$PKG" in
    brew) brew install "$brewp" ;;
    apt)  $SUDO apt-get update -y && $SUDO apt-get install -y "$aptp" ;;
    dnf)  $SUDO dnf install -y "$rpmp" ;;
    yum)  $SUDO yum install -y "$rpmp" ;;
    *)    return 1 ;;
  esac
}

# ───────────────────────── .env ─────────────────────────
ensure_env() {
  step "Checking .env configuration"
  if [ ! -f .env ]; then
    if [ -f .env.example ]; then
      cp .env.example .env
      warn ".env created from .env.example — you MUST add your ANTHROPIC_API_KEY"
    else
      die ".env and .env.example both missing"
    fi
  else
    ok ".env present"
  fi
  # Warn (don't fail) if the API key still looks like a placeholder
  if grep -qE '^(ANTHROPIC_API_KEY|OPENAI_API_KEY)=.*REPLACE_ME' .env; then
    warn "an API key in .env is still a placeholder — edit .env before processing"
    ENV_KEY_OK=0
  else
    ok "API key looks set"
    ENV_KEY_OK=1
  fi
}

ensure_dirs() {
  mkdir -p output/results logs data/result_cache
  ok "output/, logs/, cache dirs ready"
}

# ───────────────────────── Docker path ─────────────────────────
COMPOSE=""
detect_compose() {
  if docker compose version >/dev/null 2>&1; then COMPOSE="docker compose"
  elif have docker-compose;              then COMPOSE="docker-compose"
  else COMPOSE=""; fi
}

ensure_docker() {
  step "Checking Docker"
  if have docker; then
    ok "docker CLI present ($(docker --version 2>/dev/null | head -1))"
  else
    info "docker not found — installing"
    if [ "$OS" = "Linux" ] && [ "$PKG" != "none" ]; then
      info "installing Docker Engine via get.docker.com"
      curl -fsSL https://get.docker.com | $SUDO sh || die "Docker install failed"
      $SUDO systemctl enable --now docker 2>/dev/null || true
    elif [ "$OS" = "Darwin" ]; then
      if have brew; then
        info "installing Docker Desktop via Homebrew (cask)"
        brew install --cask docker || die "Docker Desktop install failed — install it manually from docker.com"
        warn "Docker Desktop installed — OPEN the Docker app once to start the daemon, then re-run this script"
        exit 0
      else
        die "install Docker Desktop from https://docker.com/products/docker-desktop then re-run"
      fi
    else
      die "cannot auto-install Docker on this platform — install it manually, or use: ./setup.sh --native"
    fi
  fi

  # daemon running?
  if docker info >/dev/null 2>&1; then
    ok "docker daemon is running"
  else
    info "docker daemon not running — attempting to start"
    if [ "$OS" = "Darwin" ]; then
      open -a Docker 2>/dev/null || true
    else
      $SUDO systemctl start docker 2>/dev/null || $SUDO service docker start 2>/dev/null || true
    fi
    # wait up to ~60s
    local n=0
    while ! docker info >/dev/null 2>&1; do
      n=$((n+1)); [ "$n" -gt 30 ] && die "docker daemon did not start — start Docker manually and re-run"
      sleep 2
    done
    ok "docker daemon is now running"
  fi

  detect_compose
  [ -n "$COMPOSE" ] || die "Docker Compose not available — install the compose plugin"
  ok "compose: $COMPOSE"
}

run_docker() {
  if [ "$CHECK_ONLY" = "1" ]; then
    detect_compose            # pure detection, no side effects
    report_check "docker"
    return
  fi
  ensure_docker
  ensure_env
  ensure_dirs
  step "Building and starting the stack (Qdrant + app)"
  info "this builds the app image (Python 3.11 + poppler + libs + models) and starts Qdrant"
  if [ "$DO_START" = "1" ]; then
    if [ "$REBUILD" = "1" ]; then
      $COMPOSE up --build --abort-on-container-exit
    else
      $COMPOSE up --build --abort-on-container-exit
    fi
  else
    $COMPOSE build
    ok "images built — start later with: $COMPOSE up"
  fi
}

# ───────────────────────── Native path ─────────────────────────
PYBIN=""
ensure_python() {
  step "Checking Python 3.11+"
  for cand in python3.12 python3.11 python3; do
    if have "$cand"; then
      ver="$($cand -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo 0.0)"
      major="${ver%%.*}"; minor="${ver##*.}"
      if [ "${major:-0}" -ge 3 ] && [ "${minor:-0}" -ge 11 ]; then
        PYBIN="$cand"; ok "using $cand (Python $ver)"; return
      fi
    fi
  done
  info "Python 3.11+ not found — installing"
  pkg_install "python@3.11" "python3.11" "python3.11" || die "could not install Python 3.11"
  have python3.11 && PYBIN="python3.11" || PYBIN="python3"
  ok "installed $($PYBIN --version)"
}

ensure_poppler() {
  step "Checking poppler (pdf2image dependency)"
  if have pdftoppm; then ok "poppler present"; return; fi
  info "installing poppler"
  pkg_install "poppler" "poppler-utils" "poppler-utils" || warn "install poppler manually if PDF parsing fails"
  have pdftoppm && ok "poppler installed" || warn "pdftoppm still not found"
}

ensure_venv_deps() {
  step "Setting up Python virtualenv + dependencies"
  [ -d .venv ] || { info "creating .venv"; "$PYBIN" -m venv .venv; }
  # shellcheck disable=SC1091
  . .venv/bin/activate
  info "installing requirements (pip)"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
  ok "dependencies installed in .venv"
}

ensure_models() {
  step "Pre-downloading embedding models (first run only)"
  # Mirrors the Dockerfile: fetch the FastEmbed models now so the first pipeline
  # run doesn't stall on a throttled HuggingFace download mid-processing.
  if python - <<'PY'
from fastembed import TextEmbedding, SparseTextEmbedding
TextEmbedding("BAAI/bge-base-en-v1.5")
SparseTextEmbedding("Qdrant/bm25")
print("ok")
PY
  then ok "embedding models cached"
  else warn "model pre-download failed — the first run will retry the download"
  fi
}

ensure_qdrant_native() {
  step "Checking Qdrant"
  if curl -sf http://localhost:6333/readyz >/dev/null 2>&1; then
    ok "Qdrant already running on :6333"; QMODE="server"; return
  fi
  if have docker && docker info >/dev/null 2>&1; then
    info "starting Qdrant in Docker (qdrant/qdrant on :6333)"
    docker rm -f podiatry-qdrant >/dev/null 2>&1 || true
    docker run -d --name podiatry-qdrant -p 6333:6333 -v podiatry_qdrant:/qdrant/storage qdrant/qdrant >/dev/null
    local n=0
    while ! curl -sf http://localhost:6333/readyz >/dev/null 2>&1; do
      n=$((n+1)); [ "$n" -gt 30 ] && { warn "Qdrant slow to start — falling back to on-disk store"; QMODE="local"; return; }
      sleep 2
    done
    ok "Qdrant running in Docker"; QMODE="server"
  else
    warn "no Docker — using the on-disk local Qdrant store (no server needed)"
    QMODE="local"
  fi
}

run_native() {
  detect_os
  if [ "$CHECK_ONLY" = "1" ]; then
    detect_python_only        # pure detection, no installs
    detect_qdrant_only
    report_check "native"
    return
  fi
  ensure_python
  ensure_poppler
  ensure_env
  ensure_dirs
  ensure_venv_deps
  ensure_models
  ensure_qdrant_native

  step "Starting the pipeline"
  # If Qdrant server isn't up, force on-disk mode by blanking QDRANT_URL for this run.
  RUN_ENV=""
  [ "${QMODE:-local}" = "local" ] && RUN_ENV="QDRANT_URL="
  [ -n "$NOTES_DIR_OVERRIDE" ] && RUN_ENV="$RUN_ENV NOTES_DIR=$NOTES_DIR_OVERRIDE"
  local args=""
  [ "$REBUILD" = "1" ] && args="--rebuild-index"
  if [ "$DO_START" = "1" ]; then
    if [ "${ENV_KEY_OK:-1}" = "0" ]; then
      warn "skipping run: add a real API key to .env, then: source .venv/bin/activate && python run.py"
      return
    fi
    # shellcheck disable=SC2086
    env $RUN_ENV python run.py $args
  else
    ok "setup complete — run with: source .venv/bin/activate && python run.py"
  fi
}

# ───────────────────────── pure detectors (for --check) ─────────────────────────
detect_python_only() {
  for cand in python3.12 python3.11 python3; do
    if have "$cand"; then
      ver="$($cand -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo 0.0)"
      major="${ver%%.*}"; minor="${ver##*.}"
      if [ "${major:-0}" -ge 3 ] && [ "${minor:-0}" -ge 11 ]; then PYBIN="$cand"; return; fi
    fi
  done
  PYBIN=""
}
detect_qdrant_only() {
  if curl -sf http://localhost:6333/readyz >/dev/null 2>&1; then QMODE="server"; else QMODE="local (on-disk) or docker"; fi
}

# ───────────────────────── check-only report ─────────────────────────
report_check() {
  step "Readiness report (mode: $1) — nothing was installed or started"
  printf "  %-22s %s\n" "OS / arch"        "$OS / $ARCH"
  printf "  %-22s %s\n" "package manager"  "$PKG"
  if [ "$1" = "docker" ]; then
    have docker && printf "  %-22s %s\n" "docker" "present" || printf "  %-22s %s\n" "docker" "MISSING"
    (docker info >/dev/null 2>&1) && printf "  %-22s %s\n" "docker daemon" "running" || printf "  %-22s %s\n" "docker daemon" "not running"
    printf "  %-22s %s\n" "compose" "${COMPOSE:-none}"
  else
    printf "  %-22s %s\n" "python"  "${PYBIN:-none}"
    have pdftoppm && printf "  %-22s %s\n" "poppler" "present" || printf "  %-22s %s\n" "poppler" "MISSING"
    printf "  %-22s %s\n" "qdrant plan" "${QMODE:-unknown}"
    [ -d .venv ] && printf "  %-22s %s\n" ".venv" "present" || printf "  %-22s %s\n" ".venv" "not created"
  fi
  printf "  %-22s %s\n" ".env" "$([ -f .env ] && echo present || echo missing)"
  ok "check complete"
}

# ───────────────────────── main ─────────────────────────
printf "%s╔══════════════════════════════════════════════╗%s\n" "$C_B" "$C_RESET"
printf "%s║  Podiatry Clean-Claim Scrubber — setup.sh    ║%s\n" "$C_B" "$C_RESET"
printf "%s╚══════════════════════════════════════════════╝%s\n" "$C_B" "$C_RESET"
printf "mode=%s  check-only=%s  start=%s\n" "$MODE" "$CHECK_ONLY" "$DO_START"

if [ "$MODE" = "docker" ]; then
  detect_os
  run_docker
else
  run_native
fi

step "Done."
