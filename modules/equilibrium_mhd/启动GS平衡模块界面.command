#!/bin/zsh -f
MODULE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$MODULE_DIR/../.." || exit 1

mkdir -p runs/logs
LOG="runs/logs/equilibrium_mhd_desktop_launcher.log"
: > "$LOG"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TK_SILENCE_DEPRECATION=1

echo "[launcher] $(date)" >> "$LOG"
echo "[launcher] cwd=$(pwd)" >> "$LOG"

close_launcher_terminal() {
  [[ "${XIRONG_KEEP_TERMINAL:-}" == "1" ]] && return 0
  local launcher_tty
  launcher_tty="$(tty 2>/dev/null || true)"
  [[ -n "$launcher_tty" && "$launcher_tty" != "not a tty" ]] || return 0
  (
    sleep 0.4
    /usr/bin/osascript >/dev/null 2>&1 <<OSA
tell application "Terminal"
  repeat with w in windows
    repeat with t in tabs of w
      if tty of t is "$launcher_tty" then
        close w
        return
      end if
    end repeat
  end repeat
end tell
OSA
  ) >/dev/null 2>&1 &
}

PY_CANDIDATES=(
  "/Library/Frameworks/Python.framework/Versions/Current/bin/python3"
  "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
  "/usr/bin/python3"
  "$(command -v python3 2>/dev/null)"
)

for PY in "${PY_CANDIDATES[@]}"; do
  [[ -n "$PY" && -x "$PY" ]] || continue
  echo "[launcher] testing Tk interface with $PY" >> "$LOG"
  "$PY" -B -u - <<'PYTEST' >> "$LOG" 2>&1
import sys
print("python", sys.executable)
import tkinter as tk
root = tk.Tk()
root.withdraw()
root.update_idletasks()
root.destroy()
import modules.equilibrium_mhd.equilibrium_mhd_desktop
print("equilibrium mhd desktop import ok")
PYTEST
  if [[ $? -eq 0 ]]; then
    echo "[launcher] starting GS equilibrium desktop with $PY" >> "$LOG"
    nohup "$PY" -B modules/equilibrium_mhd/equilibrium_mhd_desktop.py >> "$LOG" 2>&1 &
    echo "[launcher] desktop pid=$!" >> "$LOG"
    close_launcher_terminal
    exit 0
  fi
  echo "[launcher] failed: $PY" >> "$LOG"
done

echo "[launcher] no usable Python/Tk runtime found" >> "$LOG"
open -a TextEdit "$LOG"
exit 1
