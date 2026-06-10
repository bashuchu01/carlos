# -*- coding: utf-8 -*-
"""
========================================================================
 WFP GLOBAL FOOD PRICES — DASHBOARD INTERACTIVO (2025 vs 2026)
========================================================================
Fuente de datos: World Food Programme (WFP) - disponible en Kaggle.

Decisiones analíticas clave:
  1) MONEDA: se usa la columna `usdprice` (ya convertida a USD), NUNCA
     `price`, porque la data contiene 60+ monedas locales no comparables.
  2) UNIDAD: el campo `unit` tiene 100+ formatos ("50 KG", "500 G",
     "1.5 L"...). Se normaliza a USD por KG y USD por L para que la
     comparación entre productos/países sea válida.
  3) GRANULARIDAD: snapshots mensuales (día 15) -> serie temporal limpia.

Autor: Analista de Datos
========================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="WFP Food Prices | Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estética: paleta sobria, tarjetas KPI, tipografía limpia
st.markdown(
    """
    <style>
        .main { background-color: #0e1117; }
        h1, h2, h3 { letter-spacing: -0.5px; }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #1c2333 0%, #161b27 100%);
            border: 1px solid #2a3142;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.35);
        }
        div[data-testid="stMetricLabel"] { color: #9aa4b2; font-weight: 600; }
        div[data-testid="stMetricValue"] { font-size: 28px; }
        .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paleta para gráficos
PALETTE = px.colors.qualitative.Set2
TEMPLATE = "plotly_dark"

# Archivos esperados (mismo directorio que app.py)
DATA_FILES = {
    2025: "wfp_food_prices_global_2025.csv",
    2026: "wfp_food_prices_global_2026.csv",
}


# ----------------------------------------------------------------------
# 2. CARGA Y NORMALIZACIÓN DE DATOS (cacheado)
# ----------------------------------------------------------------------
def _parse_unit(u: str):
    """Convierte el texto de 'unit' a (cantidad, base) en KG o L.

    Devuelve (None, None) si la unidad no es de peso/volumen
    (p.ej. 'Bar', 'Box', 'Unit', 'pcs') -> se excluye del comparativo.
    """
    u = str(u).strip()
    m = re.match(r"^([\d.]+)\s*(KG|G|L|ML)$", u, re.IGNORECASE)
    if m:
        qty, unit = float(m.group(1)), m.group(2).upper()
        if unit == "KG":
            return qty, "KG"
        if unit == "G":
            return qty / 1000.0, "KG"
        if unit == "L":
            return qty, "L"
        if unit == "ML":
            return qty / 1000.0, "L"
    simple = {"KG": (1.0, "KG"), "L": (1.0, "L"),
              "G": (0.001, "KG"), "ML": (0.001, "L")}
    return simple.get(u.upper(), (None, None))


@st.cache_data(show_spinner="Cargando y normalizando datos...")
def load_data() -> pd.DataFrame:
    frames = []
    for year, fname in DATA_FILES.items():
        if not os.path.exists(fname):
            st.error(f"No se encontró el archivo: {fname}. "
                     "Colócalo en la misma carpeta que app.py.")
            st.stop()
        d = pd.read_csv(fname)
        d["year"] = year
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # Tipos
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month

    # Normalización de unidades -> USD por KG / USD por L
    parsed = df["unit"].apply(lambda x: pd.Series(_parse_unit(x),
                                                  index=["qty", "base"]))
    df = pd.concat([df, parsed], axis=1)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["usd_per_base"] = np.where(
        df["base"].notna() & df["usdprice"].notna() & (df["qty"] > 0),
        df["usdprice"] / df["qty"],
        np.nan,
    )

    # Limpieza ligera de columnas de texto
    for c in ["category", "commodity", "currency", "pricetype"]:
        df[c] = df[c].astype(str).str.strip()

    return df


df = load_data()

# Catálogo de nombres de país (ISO3 -> nombre legible para mapa/tooltips)
ISO3_NAME = (
    df.dropna(subset=["countryiso3"])
    .groupby("countryiso3")
    .size()
    .index.to_series()
)


# ----------------------------------------------------------------------
# 3. SIDEBAR — CONTROLES INTERACTIVOS
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Controles")
st.sidebar.caption("Filtra la data y observa los cambios en tiempo real.")

# 3.1 Año (incluye comparativo)
year_opt = st.sidebar.radio(
    "Año",
    options=["2025", "2026", "Comparar 2025 vs 2026"],
    index=2,
)

# 3.2 Base de medida (peso o volumen)
base_opt = st.sidebar.radio(
    "Base de comparación",
    options=["KG (sólidos)", "L (líquidos)"],
    horizontal=True,
)
base = "KG" if base_opt.startswith("KG") else "L"
unit_label = f"USD/{base}"

# 3.3 Tipo de precio
ptype = st.sidebar.selectbox(
    "Tipo de precio",
    options=["Retail", "Wholesale", "Ambos"],
    index=0,
)

# 3.4 País
countries = sorted(df["countryiso3"].dropna().unique().tolist())
country = st.sidebar.selectbox(
    "País (ISO3)",
    options=["🌐 Todos"] + countries,
    index=0,
)

# 3.5 Categoría
categories = sorted(df["category"].dropna().unique().tolist())
cat_sel = st.sidebar.multiselect(
    "Categorías de alimentos",
    options=categories,
    default=categories,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Robustez estadística")

# 3.6 Estadístico de agregación (la mediana resiste outliers de captura)
stat_opt = st.sidebar.radio(
    "Estadístico",
    options=["Mediana (robusto)", "Media"],
    index=0,
    help="Los precios tienen outliers extremos (errores de captura). "
         "La mediana es más fiable que la media.",
)
AGG = "median" if stat_opt.startswith("Mediana") else "mean"

# 3.7 Recorte de outliers al percentil 99
clip_outliers = st.sidebar.checkbox(
    "Recortar outliers (p99)", value=True,
    help="Elimina el 1% de precios más extremos por producto.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 La comparación usa **usdprice normalizado por unidad** para "
    "neutralizar 60+ monedas y 100+ formatos de envase."
)


# ----------------------------------------------------------------------
# 4. APLICAR FILTROS
# ----------------------------------------------------------------------
def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    d = data[data["base"] == base].copy()  # sólo unidades normalizables
    d = d.dropna(subset=["usd_per_base"])

    if year_opt == "2025":
        d = d[d["year"] == 2025]
    elif year_opt == "2026":
        d = d[d["year"] == 2026]
    # "Comparar" -> ambos años

    if ptype != "Ambos":
        d = d[d["pricetype"] == ptype]

    if country != "🌐 Todos":
        d = d[d["countryiso3"] == country]

    if cat_sel:
        d = d[d["category"].isin(cat_sel)]

    # Recorte de outliers al p99 (por producto) para precios robustos
    if clip_outliers and not d.empty:
        cap = d.groupby("commodity")["usd_per_base"].transform(
            lambda s: s.quantile(0.99)
        )
        d = d[d["usd_per_base"] <= cap]

    return d


fdf = apply_filters(df)


# ----------------------------------------------------------------------
# 5. ENCABEZADO
# ----------------------------------------------------------------------
title_scope = "Global" if country == "🌐 Todos" else country
st.title("🌍 WFP Global Food Prices — Dashboard")
st.markdown(
    f"**Alcance:** {title_scope}  ·  **Año:** {year_opt}  ·  "
    f"**Base:** {unit_label}  ·  **Tipo:** {ptype}"
)

if fdf.empty:
    st.warning("No hay datos para la combinación de filtros seleccionada. "
               "Prueba con otra base de medida, país o categoría.")
    st.stop()


# ----------------------------------------------------------------------
# 6. KPIs (3 ETIQUETAS DE INTERÉS)
# ----------------------------------------------------------------------
def variation_pct(d: pd.DataFrame) -> float:
    """Inflación alimentaria interanual: % cambio precio medio 2025->2026."""
    g = d.groupby("year")["usd_per_base"].agg(AGG)
    if 2025 in g.index and 2026 in g.index and g[2025] > 0:
        return (g[2026] - g[2025]) / g[2025] * 100
    return np.nan


avg_price = getattr(fdf["usd_per_base"], AGG)()
n_markets = fdf["market"].nunique()
n_commodities = fdf["commodity"].nunique()
infl = variation_pct(fdf)

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Precio {stat_opt.split()[0].lower()} ({unit_label})", f"${avg_price:,.2f}")
k2.metric("Mercados monitoreados", f"{n_markets:,}")
k3.metric("Productos analizados", f"{n_commodities:,}")
if not np.isnan(infl):
    k4.metric("Inflación alimentaria 25→26", f"{infl:+.1f}%",
              delta=f"{infl:+.1f}%", delta_color="inverse")
else:
    top_com = (fdf.groupby("commodity")["usd_per_base"].mean()
               .sort_values(ascending=False).index[0])
    k4.metric("Producto más caro", top_com)

st.markdown("---")


# ----------------------------------------------------------------------
# 7. GRÁFICOS INTERACTIVOS (Barras · Línea · Pastel · Mapa)
# ----------------------------------------------------------------------
tab_map, tab_bar, tab_line, tab_pie = st.tabs(
    ["🗺️ Mapa", "📊 Barras", "📈 Tendencia", "🥧 Composición"]
)

# 7.1 MAPA COROPLÉTICO — precio medio por país
with tab_map:
    st.subheader(f"Precio medio de alimentos por país ({unit_label})")
    map_df = (
        fdf.groupby("countryiso3", as_index=False)["usd_per_base"]
        .agg(AGG)
        .rename(columns={"usd_per_base": "precio_medio"})
    )
    fig_map = px.choropleth(
        map_df,
        locations="countryiso3",
        color="precio_medio",
        color_continuous_scale="YlOrRd",
        labels={"precio_medio": unit_label},
        template=TEMPLATE,
    )
    fig_map.update_geos(showcoastlines=True, coastlinecolor="#2a3142",
                        bgcolor="rgba(0,0,0,0)", showframe=False)
    fig_map.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_colorbar=dict(title=unit_label))
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("Pasa el cursor sobre cada país para ver su precio medio. "
               "Selecciona un país en el panel lateral para profundizar.")

# 7.2 BARRAS — Top productos más caros
with tab_bar:
    topn = st.slider("Top N productos", 5, 25, 12, key="topn_bar")
    st.subheader(f"Top {topn} productos por precio ({unit_label})")
    bar_df = (
        fdf.groupby("commodity", as_index=False)["usd_per_base"]
        .agg(AGG)
        .sort_values("usd_per_base", ascending=False)
        .head(topn)
    )
    fig_bar = px.bar(
        bar_df.sort_values("usd_per_base"),
        x="usd_per_base", y="commodity", orientation="h",
        color="usd_per_base", color_continuous_scale="Tealgrn",
        labels={"usd_per_base": unit_label, "commodity": "Producto"},
        template=TEMPLATE, text_auto=".2f",
    )
    fig_bar.update_layout(height=520, coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

# 7.3 LÍNEA — Evolución mensual del precio
with tab_line:
    st.subheader(f"Evolución mensual del precio medio ({unit_label})")
    # Permite enfocar un producto concreto
    com_opts = ["(Promedio de todos)"] + sorted(fdf["commodity"].unique())
    com_sel = st.selectbox("Producto", com_opts, index=0, key="line_com")
    line_src = fdf if com_sel == "(Promedio de todos)" else \
        fdf[fdf["commodity"] == com_sel]

    line_df = (
        line_src.groupby(["year", "month"], as_index=False)["usd_per_base"]
        .agg(AGG)
    )
    line_df["año"] = line_df["year"].astype(str)
    fig_line = px.line(
        line_df, x="month", y="usd_per_base", color="año",
        markers=True, color_discrete_sequence=PALETTE,
        labels={"month": "Mes", "usd_per_base": unit_label, "año": "Año"},
        template=TEMPLATE,
    )
    fig_line.update_xaxes(dtick=1)
    fig_line.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0),
                           legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_line, use_container_width=True)
    st.caption("Compara la estacionalidad de precios entre 2025 y 2026.")

# 7.4 PASTEL — Composición por categoría
with tab_pie:
    st.subheader("Composición del precio medio por categoría de alimento")
    pie_df = (
        fdf.groupby("category", as_index=False)["usd_per_base"]
        .agg(AGG)
        .rename(columns={"usd_per_base": "precio_medio"})
    )
    fig_pie = px.pie(
        pie_df, names="category", values="precio_medio", hole=0.45,
        color_discrete_sequence=PALETTE, template=TEMPLATE,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0),
                          showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption("Peso relativo del precio medio por categoría dentro del "
               "alcance seleccionado.")


# ----------------------------------------------------------------------
# 8. TABLA DETALLE + DESCARGA
# ----------------------------------------------------------------------
with st.expander("🔎 Ver datos detallados / descargar"):
    cols = ["year", "date", "countryiso3", "market", "category",
            "commodity", "unit", "currency", "price", "usdprice",
            "usd_per_base", "pricetype"]
    st.dataframe(fdf[cols].sort_values("date"), use_container_width=True,
                 height=320)
    st.download_button(
        "⬇️ Descargar selección (CSV)",
        data=fdf[cols].to_csv(index=False).encode("utf-8"),
        file_name="wfp_seleccion.csv",
        mime="text/csv",
    )

st.markdown(
    "<br><center><sub>Fuente: World Food Programme (WFP) · Kaggle · "
    "Precios normalizados a USD por unidad base.</sub></center>",
    unsafe_allow_html=True,
)
