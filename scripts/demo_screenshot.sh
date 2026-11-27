#!/bin/bash
# Regenerates docs/demo.png from a real run of the README examples.
# Needs freeze (https://github.com/charmbracelet/freeze). Only the severity tags are coloured; the text is the CLI's own output.
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(mktemp)
run() { printf '\033[1;32m$\033[0m %s\n' "cachecanary $*"; cachecanary "$@" || true; echo; }
{
  run lint examples/request_bad.json
  run diff examples/request_bad.json examples/request_next.json
  run lint examples/request_good.json --model claude-haiku-4-5
} | sed -e $'s/^\\[warn\\]/\033[33m[warn]\033[0m/' -e $'s/^\\[error\\]/\033[31m[error]\033[0m/' -e $'s/^system-changed:/\033[31msystem-changed:\033[0m/' > "$out"
freeze --execute "cat $out" --window --theme dracula --font.size 14 --padding 20,24,8,24 --margin 0 --wrap 104 --border.radius 8 -o docs/demo.png
sips -Z 1700 docs/demo.png >/dev/null
rm -f "$out"
