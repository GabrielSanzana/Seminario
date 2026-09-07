#!/usr/bin/env bash
# Cliente minimo de la API de GitHub para el flujo de dos agentes.
# Toma el token del credential helper de git en cada llamada; nunca lo guarda
# en disco ni lo imprime.
#
# Uso:
#   ./gh.sh pr-abrir "<titulo>" <archivo-con-cuerpo.md>   abre PR de la rama actual hacia main
#   ./gh.sh pr-listar                                     PR abiertos, con autor y estado de merge
#   ./gh.sh pr-ver <n>                                    detalle de un PR
#   ./gh.sh pr-comentar <n> <archivo-con-comentario.md>   deja un comentario
#   ./gh.sh pr-mergear <n>                                mergea (rechaza si hay conflictos)
set -uo pipefail

REPO="GabrielSanzana/Seminario"
API="https://api.github.com/repos/$REPO"
cd "$(git rev-parse --show-toplevel)" || exit 1

token() {
  printf 'protocol=https\nhost=github.com\n\n' | git credential fill 2>/dev/null | sed -n 's/^password=//p'
}

TOKEN=$(token)
if [ -z "$TOKEN" ]; then
  echo "FALLO: sin credencial de GitHub. Hacer un 'git push' manual una vez para guardarla." >&2
  exit 1
fi

api() {
  local metodo="$1" ruta="$2" datos="${3:-}"
  local args=(-s -X "$metodo"
    -H "Authorization: Bearer $TOKEN"
    -H "Accept: application/vnd.github+json")
  [ -n "$datos" ] && args+=(--data-binary "@$datos")
  curl "${args[@]}" "$API$ruta"
}

tmp() { mktemp "${TMPDIR:-/tmp}/gh.XXXXXX"; }

case "${1:-}" in

pr-abrir)
  titulo="${2:?falta el titulo}"
  cuerpo="${3:?falta el archivo con el cuerpo}"
  rama=$(git rev-parse --abbrev-ref HEAD)
  if [ "$rama" = "main" ]; then
    echo "FALLO: no se abre PR desde main. Cambiar a la rama del rol." >&2
    exit 1
  fi
  git push -q origin "$rama" || exit 1
  payload=$(tmp)
  python -c "
import json, sys
json.dump({'title': sys.argv[1], 'head': sys.argv[2], 'base': 'main',
           'body': open(sys.argv[3], encoding='utf-8').read()},
          open(sys.argv[4], 'w', encoding='utf-8'))
" "$titulo" "$rama" "$cuerpo" "$payload" || exit 1
  resp=$(tmp)
  api POST /pulls "$payload" > "$resp"
  python -c "
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
if 'html_url' in d:
    print('PR #%s abierto: %s' % (d['number'], d['html_url']))
else:
    print('FALLO:', d.get('message'), file=sys.stderr)
    for e in d.get('errors', []): print(' ', e.get('message') or e, file=sys.stderr)
    sys.exit(1)
" "$resp"
  ;;

pr-listar)
  resp=$(tmp); api GET "/pulls?state=open&per_page=30" > "$resp"
  python -c "
import json, sys
prs = json.load(open(sys.argv[1], encoding='utf-8'))
if not prs: print('sin PR abiertos')
for p in prs:
    print('PR #%-3s %-22s -> %s  autor:%s' % (p['number'], p['head']['ref'], p['base']['ref'], p['user']['login']))
    print('   %s' % p['title'])
" "$resp"
  ;;

pr-ver)
  n="${2:?falta el numero de PR}"
  resp=$(tmp); api GET "/pulls/$n" > "$resp"
  python -c "
import json, sys
p = json.load(open(sys.argv[1], encoding='utf-8'))
print('PR #%s: %s' % (p['number'], p['title']))
print('  %s -> %s | estado: %s | mergeable: %s (%s)' % (
    p['head']['ref'], p['base']['ref'], p['state'], p.get('mergeable'), p.get('mergeable_state')))
print('  commits=%s archivos=%s +%s/-%s' % (p['commits'], p['changed_files'], p['additions'], p['deletions']))
" "$resp"
  resp2=$(tmp); api GET "/pulls/$n/files?per_page=100" > "$resp2"
  python -c "
import json, sys
print('  archivos tocados:')
for f in json.load(open(sys.argv[1], encoding='utf-8')):
    print('    %-44s %s +%s/-%s' % (f['filename'], f['status'], f['additions'], f['deletions']))
" "$resp2"
  ;;

pr-comentar)
  n="${2:?falta el numero de PR}"
  cuerpo="${3:?falta el archivo con el comentario}"
  payload=$(tmp)
  python -c "
import json, sys
json.dump({'body': open(sys.argv[1], encoding='utf-8').read()}, open(sys.argv[2], 'w', encoding='utf-8'))
" "$cuerpo" "$payload" || exit 1
  resp=$(tmp); api POST "/issues/$n/comments" "$payload" > "$resp"
  python -c "
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
print('comentario publicado:', d['html_url']) if 'html_url' in d else (print('FALLO:', d.get('message'), file=sys.stderr), sys.exit(1))
" "$resp"
  ;;

pr-mergear)
  n="${2:?falta el numero de PR}"
  resp=$(tmp); api GET "/pulls/$n" > "$resp"
  estado=$(python -c "
import json, sys
p = json.load(open(sys.argv[1], encoding='utf-8'))
print('%s|%s|%s' % (p['state'], p.get('mergeable'), p['head']['ref']))
" "$resp")
  IFS='|' read -r st mergeable rama <<< "$estado"
  if [ "$st" != "open" ]; then
    echo "FALLO: PR #$n no esta abierto (estado: $st)" >&2; exit 1
  fi
  if [ "$mergeable" = "False" ]; then
    echo "FALLO: PR #$n tiene conflictos. El AUTOR debe resolverlos:" >&2
    echo "  git checkout $rama && git fetch origin && git rebase origin/main" >&2
    echo "  (resolver, luego) git push --force-with-lease origin $rama" >&2
    exit 1
  fi
  payload=$(tmp)
  python -c "
import json, sys
json.dump({'merge_method': 'merge'}, open(sys.argv[1], 'w', encoding='utf-8'))
" "$payload"
  resp2=$(tmp); api PUT "/pulls/$n/merge" "$payload" > "$resp2"
  python -c "
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
if d.get('merged'):
    print('PR #%s mergeado: %s' % (sys.argv[2], d['sha'][:7]))
else:
    print('FALLO:', d.get('message'), file=sys.stderr); sys.exit(1)
" "$resp2" "$n"
  ;;

*)
  sed -n '2,14p' "$0"
  exit 1
  ;;
esac
