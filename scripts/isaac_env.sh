# Shared Isaac Lab runtime for binary Isaac Sim and pip installations.
# Override ISAACLAB_PATH, ISAAC_PYTHON, ISAAC_VENV or ISAAC_SIM_PATH as needed.

_q1_pick_isaaclab_path() {
  local cand
  for cand in "${ISAACLAB_PATH:-}" "${HOME}/IsaacLab" "${HOME}/isaac/IsaacLab"; do
    [[ -n "$cand" && -d "${cand}/source/isaaclab" ]] || continue
    printf '%s\n' "$cand"
    return 0
  done
  return 1
}

ISAACLAB_PATH="$(_q1_pick_isaaclab_path)" || {
  echo "ERROR: Isaac Lab tree not found. Set ISAACLAB_PATH." >&2
  return 1 2>/dev/null || exit 1
}

_q1_pick_isaac_python() {
  local cand
  # An explicit interpreter is authoritative (including binary python.sh wrappers).
  if [[ -n "${ISAAC_PYTHON:-}" ]]; then
    [[ -x "$ISAAC_PYTHON" ]] || return 1
    printf '%s\n' "$ISAAC_PYTHON"
    return 0
  fi
  if [[ -n "${ISAAC_SIM_PATH:-}" ]]; then
    [[ -x "${ISAAC_SIM_PATH}/python.sh" ]] || return 1
    printf '%s\n' "${ISAAC_SIM_PATH}/python.sh"
    return 0
  fi
  if [[ -n "${ISAAC_VENV:-}" ]]; then
    cand="${ISAAC_VENV}/bin/python"
    [[ -x "$cand" ]] || return 1
    "$cand" -c 'import importlib.util as u; raise SystemExit(not all(u.find_spec(p) for p in ("torch", "isaacsim")))' >/dev/null 2>&1 || return 1
    printf '%s\n' "$cand"
    return 0
  fi
  for cand in \
    "${ISAACLAB_PATH}/_isaac_sim/python.sh" \
    "${HOME}/isaacsim/python.sh" "${HOME}/isaac/isaacsim/python.sh"; do
    [[ -n "$cand" && -x "$cand" ]] || continue
    printf '%s\n' "$cand"
    return 0
  done
  # The pip Isaac Sim runtime used by DGX Spark is self-contained in its venv.
  # Test package presence without importing Kit before AppLauncher starts it.
  for cand in \
    "${ISAAC_VENV:+${ISAAC_VENV}/bin/python}" \
    "${ISAACLAB_PATH}/env_isaaclab/bin/python" \
    "${HOME}/isaac/env_isaaclab/bin/python" "${HOME}/env_isaaclab/bin/python" \
    "${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}"; do
    [[ -n "$cand" && -x "$cand" ]] || continue
    if "$cand" -c 'import importlib.util as u; raise SystemExit(not all(u.find_spec(p) for p in ("torch", "isaacsim")))' >/dev/null 2>&1; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

ISAAC_PYTHON="$(_q1_pick_isaac_python)" || {
  echo "ERROR: No Isaac Sim runtime found. Set ISAAC_PYTHON to python.sh or a pip Isaac Sim venv interpreter." >&2
  return 1 2>/dev/null || exit 1
}
if [[ "$ISAAC_PYTHON" == */python.sh ]]; then
  ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-$(dirname "$ISAAC_PYTHON")}"
else
  ISAAC_VENV="${ISAAC_VENV:-$(dirname "$(dirname "$ISAAC_PYTHON")")}"
fi

# Isaac Sim's Linux ARM bootstrap requires OpenMP preloaded even for headless
# runs. A systemd user service does not inherit the interactive shell's preload.
# Discover the same system/PyTorch libraries as the installed Isaac Sim checker;
# keep this process-local and leave x86 RTX environments untouched.
case "$(uname -sm)" in
  "Linux aarch64"|"Linux arm64")
    _q1_torch_omp="$("$ISAAC_PYTHON" -c 'import glob, importlib.util; s=importlib.util.find_spec("torch"); print("\n".join(p for root in (s.submodule_search_locations if s else []) for p in glob.glob(root + ".libs/libgomp*")))')"
    while IFS= read -r _q1_omp; do
      [[ -f "$_q1_omp" ]] || continue
      _q1_preloads="${LD_PRELOAD:-}"
      case ":${_q1_preloads// /:}:" in
        *":${_q1_omp}:"*) ;;
        *) export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${_q1_omp}" ;;
      esac
    done < <(printf '%s\n' /lib/*/libgomp.so.1 "$_q1_torch_omp")
    unset _q1_torch_omp _q1_omp _q1_preloads
    ;;
esac

_Q1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ISAAC_SRC_PATH=""
for _pkg in isaaclab isaaclab_tasks isaaclab_rl isaaclab_assets isaaclab_mimic isaaclab_contrib; do
  if [[ -d "${ISAACLAB_PATH}/source/${_pkg}" ]]; then
    _ISAAC_SRC_PATH="${_ISAAC_SRC_PATH:+${_ISAAC_SRC_PATH}:}${ISAACLAB_PATH}/source/${_pkg}"
  fi
done
export PYTHONPATH="${_Q1_ROOT}/source/leg_humanoid_lab:${_ISAAC_SRC_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export Q1_LEG_ROOT="${Q1_LEG_ROOT:-${_Q1_ROOT}}"
export ISAACLAB_PATH ISAAC_SIM_PATH ISAAC_VENV ISAAC_PYTHON
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

_q1_display_alive() {
  if command -v xdpyinfo >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
    timeout 2 xdpyinfo -display "$1" >/dev/null 2>&1
  else
    [[ "$1" == :* && -S "/tmp/.X11-unix/X${1#:}" ]]
  fi
}
# Respect a working display; otherwise discover :1 on Spark and :0 on RTX desktops.
if [[ -z "${DISPLAY:-}" ]] || ! _q1_display_alive "$DISPLAY"; then
  for _display in :0 :1 :2; do
    if _q1_display_alive "$_display"; then
      export DISPLAY="$_display"
      break
    fi
  done
fi
# Keep headless execution valid even when no X server is running.
if [[ -n "${DISPLAY:-}" ]]; then export DISPLAY; fi
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
# Laptop Optimus routing is opt-in; discrete RTX and GB10 need no PRIME override.
if [[ "${Q1_PRIME_OFFLOAD:-0}" == 1 ]]; then
  export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
  export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
  export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"
fi
