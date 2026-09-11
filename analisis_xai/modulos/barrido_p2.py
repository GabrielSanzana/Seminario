"""Barrido de la agregacion entre semillas, sin reentrenar.

P2 exige invariancia a las decisiones que no cambian la cantidad estimada.
La agregacion entre semillas es una de ellas: promediar, tomar mediana o
recortar colas estiman lo mismo. Nunca se habia probado.

Se recomputan las matrices por semilla desde los 15 checkpoints publicados,
con la configuracion publicada (familia_balanceada, mascara base de familia).
Como comprobacion, la media sobre semillas debe reproducir el Datt publicado.
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/srv/pfigueroa")
import hello as H
import ablacion_atencion as AB
from certificar_todo import INDEX_NAMES, N_IDX

H.CustomTransformerEncoderLayer.forward = AB._forward_con_corte
H.ConvTransformer._recortar_atencion = AB._recortar_sin_renormalizar

pub = json.load(open("resultados/ABLACION_ATENCION_base.json"))
Dpub = np.array(pub["Datt"])

frames = AB.frames_validacion(608)
dev = H.device
img = ((H.TOKEN_PARCHE, H.TOKEN_PARCHE) if H.TOKEN_PARCHE else (52, 52))
usar_fam = (H.ABLACION_CONDICIONADA_FAMILIA
            and H.MASCARA_MODO == "familia_balanceada")
msk = (H._mascara_familias(dev) if usar_fam
       else torch.eye(N_IDX, dtype=torch.bool, device=dev))
print("  mascara base:", "familia" if usar_fam else "identidad")

with open(os.path.join(H.DIR_MATRICES, "val_losses.json")) as f:
    vl = json.load(f)
semillas = [s for s in list(H.SEEDS) if str(s) in vl][:15]
mats = []
for s in semillas:
    ck = os.path.join(H.CHECKPOINT_DIR, "model_seed_%d.pth" % s)
    if not os.path.exists(ck):
        continue
    m = H.ConvTransformer(num_indices=H.NUM_INDICES, seq_length=H.SEQ_LENGTH,
                          img_size=img, num_heads=4, num_layers=2)
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    m = m.to(dev).eval()
    mats.append(AB.matriz_ablacion_atencion(m, frames, msk, 32))
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
M = np.array(mats)
print("  %d semillas recomputadas" % len(M))
off = ~np.eye(N_IDX, dtype=bool)
err = float(np.abs(M.mean(0)[off] - Dpub[off]).max() / np.abs(Dpub[off]).max())
print("  comprobacion: la media reproduce el Datt publicado, error relativo "
      "maximo %.3f" % err)
if err > 0.15:
    print("  AVISO: no reproduce bien; el n_max de parches o la mascara "
          "difieren de los publicados. El barrido sigue siendo valido entre "
          "agregaciones, pero no es comparable con las 19 publicadas.")

def suelo_fam(D):
    v = D[off]; neg = v[v < 0]
    if not neg.size:
        return {}
    return {"media|neg|": np.abs(neg).mean(),
            "mediana|neg|": np.median(np.abs(neg)),
            "sd de los negativos": neg.std(),
            "1.4826*MAD": 1.4826 * np.median(np.abs(neg - np.median(neg)))}

agregs = {"media": M.mean(0), "mediana": np.median(M, 0),
          "recortada 20%": np.sort(M, 0)[3:-3].mean(0)}
conj = {}
for an, D in agregs.items():
    np.fill_diagonal(D, 0.0)
    for en, r in suelo_fam(D).items():
        for u in (2.5, 3.0, 4.0):
            conj[(an, en, u)] = {(i, j) for i, j in map(tuple, np.argwhere(D > u * r)) if i != j}
tam = [len(s) for s in conj.values()]
inter = set.intersection(*conj.values())
print()
print("  BARRIDO COMPLETO: %d agregaciones x 4 estimadores x 3 umbrales = %d celdas"
      % (len(agregs), len(conj)))
print("  tamano por celda: min %d, max %d, mediana %.0f" % (min(tam), max(tam), np.median(tam)))
print("  NUCLEO estable en las %d celdas: %d aristas" % (len(conj), len(inter)))
Dm = agregs["media"]
print()
for i, j in sorted(inter, key=lambda t: -Dm[t[0], t[1]]):
    print("    %-14s <- %-14s" % (INDEX_NAMES[i], INDEX_NAMES[j]))
json.dump({"nucleo": sorted([[int(i), int(j)] for i, j in inter]),
           "n_celdas": len(conj), "tam_min": int(min(tam)),
           "tam_max": int(max(tam)), "error_vs_publicado": err,
           "por_semilla": M.tolist()},
          open("resultados/BARRIDO_P2.json", "w"), indent=1)
print("\n  guardado resultados/BARRIDO_P2.json")
