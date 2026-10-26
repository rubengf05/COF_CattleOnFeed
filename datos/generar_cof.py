#!/usr/bin/env python3
"""generar_cof.py — saca datos/cof.json para la página, desde los CSV de NASS que ya
baja a diario el proyecto de carne (~/projects/CARNE/LE_GF/data/fundamental).

La página ya no consulta la API del USDA en vivo: lee este JSON. Así no hay clave
en una página pública y todo lo que se ve ha pasado antes por las comprobaciones
de abajo. Si alguna falla, el script para y no escribe nada.

    python3 datos/generar_cof.py [--fuente DIR]

Comprobaciones (dos caminos independientes para cada cifra):
  1. Suma de colocaciones por peso = colocaciones totales, todos los meses con desglose.
  2. Balance: inventario(m+1) = inventario(m) + colocaciones - ventas - otras bajas.
     NASS revisa cifras: se tolera solo el mes conocido (2013-01, -56.000 cab.).
  3. Los % sobre el año anterior del último informe = los que publica la nota de
     Hightower (datos/estimaciones.csv, columna real_pct). El informe que sale
     después puede revisar el mes anterior: solo se exige en el último.

Proyección: «si se mantiene el % sobre el año anterior del último inventario».
Se probó también la de la versión anterior (ratios estacionales de 5 años con
otras bajas reales): desde 2002 falla más que esta a 1, 2 y 3 meses
(error medio 1,21 / 2,43 / 3,59 % contra 1,17 / 1,89 / 2,29 %). La banda es el
error real de este método en el pasado (percentiles 10 y 90, desde 2002).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
FUENTE = Path.home() / "projects/CARNE/LE_GF/data/fundamental"
MESES = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
CAPACIDAD = "CAPACITY: (1,000 OR MORE HEAD)"
PESOS = [("lt600", "<600 lb", 550), ("600_699", "600-699", 650), ("700_799", "700-799", 750),
         ("800_899", "800-899", 850), ("900_999", "900-999", 950), ("ge1000", "≥1.000", 1050)]
# Del pipeline de carne (build_cof_pipeline.py): días = (1400 - peso) / 3,2 lb/día.
PESO_SALIDA, GANANCIA = 1400, 3.2
BALANCE_CONOCIDOS = {"2013-01"}


def serie(fuente, nombre, puntual=False):
    d = pd.read_csv(fuente / f"nass_cof_{nombre}.csv", thousands=",")
    d = d[(d.state_name == "US TOTAL") & (d.domaincat_desc == CAPACIDAD)
          & (d.freq_desc == ("POINT IN TIME" if puntual else "MONTHLY"))]
    mes = d.reference_period_desc.str.replace("FIRST OF ", "").map(MESES)
    d = d.assign(f=pd.PeriodIndex(pd.to_datetime(dict(year=d.year, month=mes, day=1)), freq="M"))
    return d.dropna(subset=["f"]).groupby("f").Value.first().astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", type=Path, default=FUENTE)
    fuente = ap.parse_args().fuente

    t = pd.DataFrame({"inv": serie(fuente, "inventory", True)})
    for k, n in [("col", "placements"), ("ven", "marketings"), ("otr", "other_disap")]:
        t[k] = serie(fuente, n)
    for k, _, _ in PESOS:
        t["p_" + k] = serie(fuente, "place_" + k)
    t["novillos"] = serie(fuente, "steers", True)
    t["novillas"] = serie(fuente, "heifers", True)
    t = t.sort_index()
    t = t[t.index >= t.col.first_valid_index()]
    fallos = []

    # 1. pesos
    cp = t[["p_" + k for k, _, _ in PESOS]].sum(axis=1, min_count=6) - t.col
    cp = cp.dropna()
    if (cp != 0).any():
        fallos.append(f"pesos ≠ total en {(cp != 0).sum()} de {len(cp)} meses")
    # 2. balance
    cb = (t.inv.shift(-1) - (t.inv + t.col - t.ven - t.otr)).dropna()
    malos = {str(p) for p in cb[cb != 0].index} - BALANCE_CONOCIDOS
    if malos:
        fallos.append(f"balance no cuadra en {sorted(malos)}")
    # 3. último informe contra la nota de Hightower
    est = pd.read_csv(AQUI / "estimaciones.csv")
    ult = t.inv.last_valid_index()
    yoy = {"inventario": t.inv[ult] / t.inv[ult - 12] * 100,
           "colocaciones": t.col[ult - 1] / t.col[ult - 13] * 100,
           "ventas": t.ven[ult - 1] / t.ven[ult - 13] * 100}
    ue = est[est.informe == est.informe.max()]
    for _, r in ue.iterrows():
        if round(yoy[r.dato], 1) != r.real_pct:
            fallos.append(f"{r.dato}: NASS {yoy[r.dato]:.1f} % ≠ Hightower {r.real_pct} %")
    if pd.Period(ue.informe.iloc[0], "M") != ult:
        fallos.append(f"estimaciones.csv acaba en {ue.informe.iloc[0]} y el inventario en {ult}: falta el informe nuevo")
    if fallos:
        print("NO SE ESCRIBE cof.json:\n  " + "\n  ".join(fallos))
        sys.exit(1)

    # Proyección: % sobre año anterior constante, con banda del error histórico.
    H = 3
    err = {h: [] for h in range(1, H + 1)}
    for a in t.index:
        if a < pd.Period("2002-01", "M"):
            continue
        for h in range(1, H + 1):
            m = a + h
            if m in t.index and pd.notna(t.inv.get(m)) and pd.notna(t.inv.get(a - 12)):
                p = t.inv[m - 12] * t.inv[a] / t.inv[a - 12]
                err[h].append(t.inv[m] / p - 1)
    proy = []
    for h in range(1, H + 1):
        m = ult + h
        base = t.inv[m - 12] * t.inv[ult] / t.inv[ult - 12]
        lo, hi = np.percentile(err[h], [10, 90])
        proy.append({"mes": str(m), "cab": round(base), "lo": round(base * (1 + lo)),
                     "hi": round(base * (1 + hi)), "err_medio_pct": round(np.mean(np.abs(err[h])) * 100, 2),
                     "n": len(err[h])})

    def lst(s):
        return [None if pd.isna(v) else int(v) for v in s]

    out = {
        "generado": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ultimo_informe": est.informe.max(),
        "ultimo_inventario": str(ult),
        "meses": [str(p) for p in t.index],
        "inv": lst(t.inv), "col": lst(t.col), "ven": lst(t.ven), "otr": lst(t.otr),
        "pesos": [{"k": k, "nombre": n, "salida_meses": round((PESO_SALIDA - p) / GANANCIA / 30.4, 1),
                   "serie": lst(t["p_" + k])} for k, n, p in PESOS],
        "novillos": lst(t.novillos), "novillas": lst(t.novillas),
        "proyeccion": proy,
        "estimaciones": est.to_dict(orient="records"),
        "comprobaciones": {
            "pesos_meses": int(len(cp)),
            "balance_meses": int(len(cb)),
            "balance_excepciones": sorted(BALANCE_CONOCIDOS),
            "yoy_ultimo": {k: round(v, 1) for k, v in yoy.items()},
        },
    }
    (AQUI / "cof.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"cof.json: {len(t)} meses hasta {ult} · pesos {len(cp)} meses OK · balance {len(cb)} meses OK "
          f"· último informe cuadra con Hightower {yoy}")
    print("proyección:", proy)


if __name__ == "__main__":
    main()
