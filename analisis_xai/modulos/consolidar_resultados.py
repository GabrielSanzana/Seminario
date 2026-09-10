"""
Consolida todos los resultados del pipeline en un único archivo ZIP.
Incluye JSONs, PNGs y un resumen ejecutivo en texto plano.

Uso:
  python consolidar_resultados.py

Genera:
  resultados_pipeline.zip  (en la misma carpeta del script)
"""
import json
import zipfile
import os
import sys
from pathlib import Path
from datetime import datetime

# La consola de Windows usa cp1252 y revienta con los emoji de los prints.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).parent.resolve() if '__file__' in globals() else Path.cwd()
RESULTS_DIR = SCRIPT_DIR / "resultados"
OUTPUT_ZIP = SCRIPT_DIR / "resultados_pipeline.zip"


def main():
    if not RESULTS_DIR.exists():
        print(f"❌ No se encontró: {RESULTS_DIR}")
        return

    print(f"📦 Consolidando resultados de: {RESULTS_DIR}")
    print(f"📦 ZIP de salida: {OUTPUT_ZIP}")
    print()

    # Recopilar todos los archivos relevantes
    archivos = []
    for ext in ["*.json", "*.png"]:
        archivos.extend(RESULTS_DIR.rglob(ext))

    # También incluir el stack principal si existe
    stack_path = RESULTS_DIR / "indices_12.npy"
    if stack_path.exists():
        archivos.append(stack_path)

    print(f"📊 Total de archivos a incluir: {len(archivos)}")
    print()

    # Crear ZIP
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Resumen ejecutivo
        resumen = generar_resumen(archivos)
        zf.writestr("00_RESUMEN.txt", resumen)
        print(f"  ✅ 00_RESUMEN.txt")

        # Archivos
        for f in archivos:
            if f.is_file():
                arcname = str(f.relative_to(SCRIPT_DIR))
                # Limitar tamaño del stack .npy (puede ser grande)
                if f.suffix == ".npy" and f.stat().st_size > 50 * 1024 * 1024:
                    print(f"  ⚠ {f.name} omitido (>50MB)")
                    continue
                zf.write(f, arcname)
                size_kb = f.stat().st_size / 1024
                print(f"  ✅ {arcname} ({size_kb:.0f} KB)")

    zip_size = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"✅ ZIP generado: {OUTPUT_ZIP}")
    print(f"   Tamaño: {zip_size:.1f} MB")
    print(f"   Total archivos: {len(archivos) + 1}")
    print(f"{'='*60}")
    print(f"\n📧 Envía este archivo para análisis:")
    print(f"   {OUTPUT_ZIP}")


def generar_resumen(archivos):
    """Genera un resumen ejecutivo en texto plano."""
    lines = []
    lines.append("=" * 60)
    lines.append("RESUMEN EJECUTIVO — Pipeline CIREN")
    lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    # Contar por tipo
    jsons = [f for f in archivos if f.suffix == ".json"]
    pngs = [f for f in archivos if f.suffix == ".png"]
    npys = [f for f in archivos if f.suffix == ".npy"]

    lines.append(f"ARCHIVOS INCLUIDOS:")
    lines.append(f"  JSONs: {len(jsons)}")
    lines.append(f"  PNGs:  {len(pngs)}")
    lines.append(f"  NPYs:  {len(npys)}")
    lines.append("")

    # Listar JSONs con su contenido clave
    lines.append("JSONs DE RESULTADOS:")
    lines.append("-" * 60)
    for jf in sorted(jsons):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Extraer métricas clave según el tipo
            rel = jf.relative_to(RESULTS_DIR)
            lines.append(f"\n📄 {rel}")

            # num(): formatea solo si el valor es numérico. Antes se hacía
            # f"{data.get(k,'?'):.4f}" directamente, así que si la clave había
            # cambiado de nombre el '?' llegaba al formato numérico y reventaba
            # con "Unknown format code 'f' for object of type 'str'". Eso fue lo
            # que dejó el resumen del zip lleno de '?' y con un error en P5.
            def num(v, dec=4):
                return f"{v:.{dec}f}" if isinstance(v, (int, float)) else "n/d"

            if "paradigm1_results" in jf.name:
                edges = data.get("edges", [])
                n_bh = sum(1 for e in edges if e.get("bh_sig"))
                lines.append(f"   P* = {data.get('p_optimo', 'n/d')} · "
                             f"operador {data.get('operador', 'n/d')} · "
                             f"η = {num(data.get('eta'), 6)}")
                lines.append(f"   Aristas: {len(edges)} · significativas BH: {n_bh}")
            elif "paradigm2_results" in jf.name:
                abl = data.get("ablation", [])
                lines.append(f"   Aristas analizadas: {data.get('n_candidates', len(abl))}")
                if abl:
                    top = max(abl, key=lambda e: e.get("auc_degradation", 0))
                    lines.append(f"   Top AUC entropía: "
                                 f"{top.get('label_i','?')}→{top.get('label_j','?')} "
                                 f"= {num(top.get('auc_degradation'))}")
                dm = data.get("delta_mse") or {}
                if dm.get("aristas"):
                    t = max(dm["aristas"], key=lambda r: r.get("delta_mse_alpha0", 0))
                    lines.append(f"   ΔMSE real (§8.2): media "
                                 f"{num(dm.get('delta_mse_media'), 6)} · máx "
                                 f"{num(dm.get('delta_mse_max'), 6)}")
                    lines.append(f"   Top ΔMSE: {t.get('label_i','?')}→"
                                 f"{t.get('label_j','?')} = "
                                 f"{num(t.get('delta_mse_alpha0'), 6)}")
                    lines.append(f"   Umbrales Tabla 7: aprueban "
                                 f"{dm.get('n_aprueba_delta','?')}, fallan "
                                 f"{dm.get('n_falla_delta','?')} de "
                                 f"{len(dm['aristas'])}")
                else:
                    lines.append(f"   ΔMSE real: NO disponible "
                                 f"(solo el proxy de entropía)")
            elif "paradigm3_results" in jf.name:
                rs = data.get("node_role_summary", {})
                stable = sum(1 for v in rs.values() if v.get("prob", 0) > 0.5)
                s = data.get("summary", {})
                lines.append(f"   Nodos con rol estable (P>0.5): {stable}/{len(rs)}")
                lines.append(f"   GED {num(s.get('ged_mean'), 3)} · "
                             f"ρrole {num(s.get('role_stability_spearman_mean'), 3)} · "
                             f"Jaccard {num(s.get('jaccard_mean'), 3)}")
            elif "triangulation_results" in jf.name:
                counts = data.get("counts", {})
                lines.append(f"   Perfiles:")
                for p, n in counts.items():
                    lines.append(f"     {p}: {n}")
                ctx = data.get("contexto_global", {})
                lines.append(f"   Te global: "
                             f"{'APRUEBA' if ctx.get('te_global_pass') else 'FALLA'}"
                             f"  [{ctx.get('te_origen', 'n/d')}]")
                if ctx.get("te_p_empirico") is not None:
                    lines.append(f"     p empírico "
                                 f"{num(ctx.get('te_p_empirico'), 3)} · "
                                 f"razón {num(ctx.get('te_razon_vs_nulo'), 1)}x "
                                 f"· Z {num(ctx.get('te_z_calibrado'), 1)} "
                                 f"(inflado, no citar)")
                lines.append(f"   Fe: {ctx.get('fe_origen', 'n/d')}")
                sin_ev = data.get("n_fe_sin_evaluar")
                if sin_ev:
                    ev = data.get("n_fe_evaluadas", "?")
                    lines.append(f"     ΔMSE medido en {ev} de "
                                 f"{data.get('total_edges', '?')} aristas; "
                                 f"{sin_ev} nunca evaluadas (fuera del "
                                 f"grafo Top-P) cuentan como ruido")
            elif "paradigm4_results" in jf.name:
                lines.append(f"   D_JS media: {num(data.get('djs_mean'), 6)}")
                lines.append(f"   Wasserstein media: {num(data.get('wasserstein_mean'))}")
                lines.append(f"   NMI máx: {num(data.get('nmi_max'))}")
            elif "paradigm5_results" in jf.name:
                bc = data.get("baseline_comparison", {})
                lines.append(f"   Consenso: {data.get('consensus_size', 'n/d')} aristas")
                lines.append(f"   Jaccard Pearson: {num(bc.get('jaccard_pearson'))}")
                lines.append(f"   Jaccard Granger: {num(bc.get('jaccard_granger'))}")
                for k, v in (data.get("kmatched") or {}).items():
                    lines.append(f"   Densidad igualada P={k}: "
                                 f"A/A {num(v.get('j_attn_attn'))} · "
                                 f"A/P {num(v.get('ratio_attn_vs_pearson'), 2)}x · "
                                 f"A/G {num(v.get('ratio_attn_vs_granger'), 2)}x")
            elif "calibracion_nulo" in jf.name:
                ETQ = {"rho_mean": "Spearman", "tau_mean": "Kendall",
                       "djs_mean": "D_JS"}
                lines.append(f"   {data.get('_n_nulos', '?')} nulos · "
                             f"{data.get('_n_pares', '?')} pares por nulo")
                for tipo in ("sin_estructura", "solo_columnas"):
                    d = data.get(tipo)
                    if not isinstance(d, dict):
                        continue
                    lines.append(f"   nulo {tipo}:")
                    for k, v in d.items():
                        p = v.get("p_empirico")
                        # La razón sólo se imprime si la media del nulo no
                        # roza el cero; si no, se dispara y no dice nada.
                        r = (f" · {v['razon']:.1f}x"
                             if v.get("razon") is not None
                             and abs(v.get("nulo_media", 0)) > 0.02 else "")
                        lines.append(
                            f"     {ETQ.get(k, k):<9} obs "
                            f"{num(v.get('observado'), 4)} vs nulo "
                            f"{num(v.get('nulo_media'), 4)}{r} · "
                            + (f"p={p:.3f}" if isinstance(p, float)
                               else f"Z={num(v.get('z'), 1)} (sin p)"))
            elif jf.name == "CERTIFICACION.json":
                # Es el resultado que responde la pregunta de la tesis: va
                # primero y completo, no resumido.
                for _k, _r in data.items():
                    lines.append(f"   {_r.get('etiqueta', _k)}")
                    lines.append(f"     rho condicional {_r['rho_condicional']:+.3f} "
                                 f"(p={_r['p_condicional']:.3f}) · "
                                 f"rho marginal {_r['rho_marginal']:+.3f} "
                                 f"(p={_r['p_marginal']:.3f})")
                    lines.append(f"     controles positivos "
                                 f"{_r['n_controles_recuperados']}/{_r['n_controles']} · "
                                 f"semillas pro-condicional "
                                 f"{_r['frac_semillas_gana_condicional']*100:.0f}%")
                    lines.append(f"     VEREDICTO: {_r['veredicto']}")
                    _pr = ", ".join(f"{t['par']} {t['puesto']}/{_r['n_pares']}"
                                    for t in _r["top_pares_reales"])
                    lines.append(f"     pares reales: {_pr}")
            elif "certificacion" in jf.name.lower():
                lines.append("   (ver CERTIFICACION.json en la raiz de resultados/)")
            elif "fase1_resumen" in jf.name:
                lines.append(f"   ρ {num(data.get('rho_mean'), 3)} · "
                             f"τ {num(data.get('tau_mean'), 3)} · "
                             f"D_JS {num(data.get('djs_mean'), 4)}")
                lines.append(f"   Veredicto GLOBAL: "
                             f"{data.get('veredicto', {}).get('GLOBAL', 'n/d')}")
                cal = data.get("calibracion_nulo", {}).get("solo_columnas", {})
                if cal.get("rho_mean"):
                    c = cal["rho_mean"]
                    p = c.get("p_empirico")
                    raz = c.get("razon") or (c["observado"]
                                             / max(abs(c["nulo_media"]), 1e-9))
                    lines.append(f"   Calibrado vs nulo solo-columnas: "
                                 f"obs {num(c.get('observado'), 3)} vs "
                                 f"{num(c.get('nulo_media'), 3)} → "
                                 f"{num(raz, 1)}x"
                                 + (f" · p empírico {p:.3f}"
                                    if isinstance(p, float) else ""))
                    lines.append(f"   (cita el p, no el múltiplo: la razón "
                                 f"no es estable entre entrenamientos)")
                else:
                    lines.append(f"   Calibración vs nulo: NO generada "
                                 f"(borra fase1_resumen.json y relanza)")
            elif "rashomon" in jf.name:
                lines.append(f"   Modelos en el conjunto: {data.get('M', 'n/d')}")
                lines.append(f"   Aristas robustas: {data.get('n_robustas', 'n/d')}"
                             f"/{len(data.get('aristas', []))}")
            elif "val_losses" in jf.name and isinstance(data, dict) and data:
                vs = [v for v in data.values() if isinstance(v, (int, float))]
                if vs:
                    import statistics as st
                    lines.append(f"   {len(vs)} semillas · val MSE mediana "
                                 f"{num(st.median(vs), 5)} · "
                                 f"[{num(min(vs), 5)}, {num(max(vs), 5)}]")
            elif "fuente_atencion" in jf.name:
                lines.append(f"   Fuente de la matriz A: {data.get('fuente', 'n/d')}"
                             + (f" (capa {data.get('capa')})"
                                if data.get("capa") is not None else "")
                             + (f" (cabeza {data.get('cabeza')})"
                                if data.get("cabeza") is not None else ""))
                if data.get("detalle"):
                    lines.append(f"   {data['detalle']}")
            elif "significant_edges" in jf.name:
                if "n_sig_bh" in data:
                    lines.append(f"   {data.get('n_sig_bh', 'n/d')} sig. BH y "
                                 f"{data.get('n_sig_bonferroni', 'n/d')} sig. Bonferroni "
                                 f"de {data.get('n_tests', 'n/d')} tests "
                                 f"({data.get('n_seeds', 'n/d')} semillas)")
                    lines.append(f"   Dispersión {num(data.get('sparsity'), 3)} · "
                                 f"cobertura {num(data.get('cobertura'), 3)} · "
                                 f"Z modularidad {num(data.get('z_modularity'), 2)}")
                    v = data.get("veredicto", {})
                    if isinstance(v, dict) and v:
                        lines.append("   Veredicto: " + " · ".join(
                            f"{k}={vv}" for k, vv in v.items()))
                else:
                    lines.append(f"   Total aristas: "
                                 f"{data.get('total_edges', len(data.get('edges', [])) or 'n/d')}")
            elif "method1_results" in jf.name:
                lines.append(f"   Réplicas: {data.get('n_replicas', '?')}")
                lines.append(f"   Pares/réplica: {data.get('n_pairs', '?')}")
                lines.append(f"   Aristas probadas: {data.get('n_significant_edges_tested', '?')}")
            elif "method2_results" in jf.name:
                lines.append(f"   Especies: {data.get('n_species', '?')}")
                lines.append(f"   Modelos/especie: {data.get('n_models_per_species', '?')}")
                tests = data.get("per_edge_tests", {})
                n_sig = sum(1 for t in tests.values()
                             if t.get("p_kruskal") is not None and t["p_kruskal"] < 0.01)
                lines.append(f"   Aristas con diff sig.: {n_sig}")
        except Exception as e:
            lines.append(f"   ⚠ Error leyendo: {e}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("FIN DEL RESUMEN")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
