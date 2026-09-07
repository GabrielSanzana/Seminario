#!/usr/bin/env bash
# Rebasea la rama actual sobre origin/main. Si hay conflicto, lo reporta y
# deja el rebase en curso para que el agente lo resuelva.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1
rama=$(git rev-parse --abbrev-ref HEAD)

if [ "$rama" = "main" ]; then
  git pull -q --ff-only origin main && echo "OK: main al dia" && exit 0
  echo "FALLO: main divergio del remoto" >&2; exit 1
fi

sucio=$(git status --porcelain | grep -v '^?? ' || true)
if [ -n "$sucio" ]; then
  echo "FALLO: hay cambios sin commitear. Commitear antes de sincronizar:" >&2
  printf '%s\n' "$sucio" >&2
  exit 1
fi

git fetch -q origin || exit 1

if git rebase origin/main >/dev/null 2>&1; then
  echo "OK: $rama rebaseada sobre origin/main"
  exit 0
fi

conflictos=$(git diff --name-only --diff-filter=U)
echo "CONFLICTO en $rama. Archivos:" >&2
printf '%s\n' "$conflictos" | sed 's/^/  /' >&2
echo >&2
if [ "$(printf '%s\n' "$conflictos" | tr -d '[:space:]')" = "tesis/TASKS.md" ]; then
  echo "Solo TASKS.md: conservar AMBOS lados. Son filas distintas de la misma tabla," >&2
  echo "no versiones rivales del mismo hecho." >&2
else
  echo "Conflicto en contenido. Resolver preservando el trabajo de las dos ramas;" >&2
  echo "si las dos editaron la misma seccion, es un fallo del protocolo de reserva:" >&2
  echo "detenerse y avisar al autor humano." >&2
fi
echo >&2
echo "Al terminar: git add <archivos> && git rebase --continue" >&2
echo "Para abortar:  git rebase --abort" >&2
exit 1
