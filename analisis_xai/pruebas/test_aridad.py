import numpy as np, sys
sys.path.insert(0, ".")
from analisis_atencion import (descomponer_aridad, fracciones_aridad,
                               sustituto_unario, _icc, OFF)
from certificar_todo import N_IDX
rng = np.random.default_rng(0)

# 1. Matriz puramente unaria: la parte de par tiene que salir cero.
r = rng.normal(size=N_IDX); c = rng.normal(size=N_IDX)
A = 3.0 + r[:, None] + c[None, :]
f = fracciones_aridad(A)
print("unaria      frac_par %.6f  (esperado ~0)" % f["frac_par"])

# 2. Matriz puramente de par: fila y columna tienen que salir cero.
B = rng.normal(size=(N_IDX, N_IDX))
mu, rr, cc, R = descomponer_aridad(B)
Apar = np.zeros((N_IDX, N_IDX)); Apar[OFF] = R[OFF]
f = fracciones_aridad(Apar)
print("de par      frac_par %.6f  frac_fila %.6f  (esperado 1 y 0)"
      % (f["frac_par"], f["frac_fila"]))

# 3. Recomposicion exacta.
mu, rr, cc, R = descomponer_aridad(B)
rec = mu + rr[:, None] + cc[None, :] + R
print("recompone   max error %.2e" % np.abs(rec[OFF] - B[OFF]).max())

# 4. Ortogonalidad: las sumas de cuadrados suman el total.
tot = float(((B - np.nanmean(np.where(OFF, B, np.nan)))[OFF] ** 2).sum())
ss = ((N_IDX-1)*np.sum(rr**2) + (N_IDX-1)*np.sum(cc**2)
      + np.sum(R[OFF]**2))
print("ortogonal   SS total %.6f  suma partes %.6f" % (tot, ss))

# 5. El sustituto de una matriz unaria devuelve la misma matriz.
Au = 3.0 + r[:, None] + c[None, :]
S = sustituto_unario(Au)
print("sustituto   corr con original %.6f (esperado 1)"
      % np.corrcoef(S[OFF], Au[OFF])[0, 1])

# 6. ICC: tabla con senal fuerte contra tabla de puro ruido.
T1 = rng.normal(size=(50, 20)) * 0.1 + rng.normal(size=20)
T2 = rng.normal(size=(50, 20))
print("icc         senal %.3f   ruido %.3f" % (_icc(T1), _icc(T2)))
