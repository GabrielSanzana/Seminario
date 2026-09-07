#!/usr/bin/env bash
# Compila la tesis y falla si hay errores de LaTeX.
# Es la compuerta que reemplaza la revision humana: ningun merge sin esto en verde.
set -uo pipefail

export PATH="/c/Users/patru/AppData/Roaming/TinyTeX/bin/windows:$PATH"
cd "$(dirname "$0")/../tesis" || exit 1

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "FALLO: pdflatex no esta en el PATH. TinyTeX no instalado?" >&2
  exit 1
fi

# Dos pasadas: la segunda resuelve indice y referencias cruzadas.
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
out=$(pdflatex -interaction=nonstopmode main.tex 2>&1)

errores=$(printf '%s\n' "$out" | grep -E '^!' || true)
if [ -n "$errores" ]; then
  echo "FALLO: errores de LaTeX" >&2
  printf '%s\n' "$errores" >&2
  exit 1
fi

paginas=$(printf '%s\n' "$out" | sed -n 's/.*Output written on main\.pdf (\([0-9]*\) pages.*/\1/p')
if [ -z "$paginas" ]; then
  echo "FALLO: no se genero main.pdf" >&2
  exit 1
fi

# Referencias sin resolver: no rompen la compilacion pero salen como ?? en el PDF.
colgantes=$(printf '%s\n' "$out" | grep -cE 'LaTeX Warning: (Reference|Citation).*undefined' || true)

echo "OK: main.pdf compilado, $paginas paginas"
if [ "$colgantes" -gt 0 ]; then
  echo "ADVERTENCIA: $colgantes referencias o citas sin resolver (saldran como ?? en el PDF)"
  printf '%s\n' "$out" | grep -E 'LaTeX Warning: (Reference|Citation).*undefined' | sed 's/^/  /'
  exit 2
fi
exit 0
