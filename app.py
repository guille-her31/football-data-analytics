import os
import re
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Football Team Dashboard", layout="wide")

# -----------------------------
# Data sources (football-data.co.uk)
# -----------------------------
# Notes:
# - football-data.co.uk hosts league-season CSVs under the pattern:
#   https://www.football-data.co.uk/mmz4281/<SEASON>/<DIV>.csv
# - SEASON is usually like "2324" (2023-24), "2223", etc.
# - DIV examples:
#   England: E0 (Premier League), E1 (Championship)
#   Spain:   SP1 (LaLiga)
#   Italy:   I1 (Serie A)
#   Germany: D1 (Bundesliga)
#   France:  F1 (Ligue 1)
LEAGUES = {
    "England: Premier League (E0)": "E0",
    "England: Championship (E1)": "E1",
    "Spain: LaLiga (SP1)": "SP1",
    "Italy: Serie A (I1)": "I1",
    "Germany: Bundesliga (D1)": "D1",
    "France: Ligue 1 (F1)": "F1",
}

DEFAULT_SEASON = "2324"  # you can change this in the UI

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

def build_url(season: str, div: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

def safe_filename(season: str, div: str) -> str:
    return os.path.join(DATA_DIR, f"{div}_{season}.csv")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names used in football-data.co.uk files.
    # Required core columns:
    # Date, HomeTeam, AwayTeam, FTHG, FTAG
    # Optional: Time, Div, etc.
    df = df.copy()

    # Some CSVs may have weird BOM or spacing
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # Ensure required columns exist
    required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Parse Date if available
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    else:
        df["Date"] = pd.NaT

    # Convert goals to numeric
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

    # Filter to played matches only (goals present)
    df = df.dropna(subset=["FTHG", "FTAG"]).copy()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Create result columns
    df["HomePoints"] = np.where(df["FTHG"] > df["FTAG"], 3, np.where(df["FTHG"] == df["FTAG"], 1, 0))
    df["AwayPoints"] = np.where(df["FTAG"] > df["FTHG"], 3, np.where(df["FTAG"] == df["FTHG"], 1, 0))

    df["FTR_calc"] = np.where(df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] == df["FTAG"], "D", "A"))

    return df

@st.cache_data(show_spinner=False)
def load_matches(season: str, div: str) -> pd.DataFrame:
    """
    Downloads the CSV (if not cached) and returns a cleaned matches dataframe.
    """
    url = build_url(season, div)
    cache_path = safe_filename(season, div)

    # Prefer cached file if it exists and is non-empty
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        df = pd.read_csv(cache_path)
        return normalize_columns(df)

    # Otherwise download
    r = requests.get(url, timeout=30)
    if r.status_code != 200 or len(r.text) < 100:
        raise ValueError(
            f"Could not download dataset. HTTP {r.status_code}. "
            f"Check season code '{season}' and league '{div}'. URL: {url}"
        )

    with open(cache_path, "wb") as f:
        f.write(r.content)

    df = pd.read_csv(cache_path)
    return normalize_columns(df)

def compute_table(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Computes league table from match-level data.
    """
    home = matches[["HomeTeam", "FTHG", "FTAG", "HomePoints"]].copy()
    home.columns = ["Team", "GF", "GA", "Pts"]
    home["W"] = (matches["FTHG"] > matches["FTAG"]).astype(int)
    home["D"] = (matches["FTHG"] == matches["FTAG"]).astype(int)
    home["L"] = (matches["FTHG"] < matches["FTAG"]).astype(int)
    home["MP"] = 1

    away = matches[["AwayTeam", "FTAG", "FTHG", "AwayPoints"]].copy()
    away.columns = ["Team", "GF", "GA", "Pts"]
    away["W"] = (matches["FTAG"] > matches["FTHG"]).astype(int)
    away["D"] = (matches["FTAG"] == matches["FTHG"]).astype(int)
    away["L"] = (matches["FTAG"] < matches["FTHG"]).astype(int)
    away["MP"] = 1

    agg = pd.concat([home, away], ignore_index=True).groupby("Team", as_index=False).agg(
        MP=("MP", "sum"),
        W=("W", "sum"),
        D=("D", "sum"),
        L=("L", "sum"),
        GF=("GF", "sum"),
        GA=("GA", "sum"),
        Pts=("Pts", "sum"),
    )
    agg["GD"] = agg["GF"] - agg["GA"]

    # Sort by points, goal difference, goals for (common convention)
    agg = agg.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    agg.insert(0, "Pos", np.arange(1, len(agg) + 1))
    return agg

def compute_home_away_tables(matches: pd.DataFrame):
    """
    Returns two tables: home-only and away-only.
    """
    home = matches.copy()
    home_tbl = home.groupby("HomeTeam", as_index=False).agg(
        MP=("HomeTeam", "size"),
        W=("HomePoints", lambda s: int((s == 3).sum())),
        D=("HomePoints", lambda s: int((s == 1).sum())),
        L=("HomePoints", lambda s: int((s == 0).sum())),
        GF=("FTHG", "sum"),
        GA=("FTAG", "sum"),
        Pts=("HomePoints", "sum"),
    ).rename(columns={"HomeTeam": "Team"})
    home_tbl["GD"] = home_tbl["GF"] - home_tbl["GA"]
    home_tbl = home_tbl.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    home_tbl.insert(0, "Pos", np.arange(1, len(home_tbl) + 1))

    away = matches.copy()
    away_tbl = away.groupby("AwayTeam", as_index=False).agg(
        MP=("AwayTeam", "size"),
        W=("AwayPoints", lambda s: int((s == 3).sum())),
        D=("AwayPoints", lambda s: int((s == 1).sum())),
        L=("AwayPoints", lambda s: int((s == 0).sum())),
        GF=("FTAG", "sum"),
        GA=("FTHG", "sum"),
        Pts=("AwayPoints", "sum"),
    ).rename(columns={"AwayTeam": "Team"})
    away_tbl["GD"] = away_tbl["GF"] - away_tbl["GA"]
    away_tbl = away_tbl.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    away_tbl.insert(0, "Pos", np.arange(1, len(away_tbl) + 1))

    return home_tbl, away_tbl

def build_team_match_log(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Creates a per-match log for a team with points, GF, GA, and a running cumulative points series.
    """
    m = matches.copy()

    is_home = m["HomeTeam"] == team
    is_away = m["AwayTeam"] == team

    home_rows = m[is_home].copy()
    home_rows["Venue"] = "Home"
    home_rows["Opponent"] = home_rows["AwayTeam"]
    home_rows["GF_team"] = home_rows["FTHG"]
    home_rows["GA_team"] = home_rows["FTAG"]
    home_rows["Pts_team"] = home_rows["HomePoints"]

    away_rows = m[is_away].copy()
    away_rows["Venue"] = "Away"
    away_rows["Opponent"] = away_rows["HomeTeam"]
    away_rows["GF_team"] = away_rows["FTAG"]
    away_rows["GA_team"] = away_rows["FTHG"]
    away_rows["Pts_team"] = away_rows["AwayPoints"]

    log = pd.concat([home_rows, away_rows], ignore_index=True)
    log = log[["Date", "Venue", "Opponent", "GF_team", "GA_team", "Pts_team"]].copy()

    # If Date missing, keep original order. If present, sort by date.
    if log["Date"].notna().any():
        log = log.sort_values("Date")
    else:
        log = log.reset_index(drop=True)

    log["MatchNo"] = np.arange(1, len(log) + 1)
    log["CumPts"] = log["Pts_team"].cumsum()

    # Form label
    def outcome(row):
        if row["GF_team"] > row["GA_team"]:
            return "W"
        if row["GF_team"] == row["GA_team"]:
            return "D"
        return "L"
    log["Result"] = log.apply(outcome, axis=1)

    return log.reset_index(drop=True)

def recent_form_string(log: pd.DataFrame, n: int) -> str:
    if log.empty:
        return ""
    tail = log.tail(n)["Result"].tolist()
    return "".join(tail)

# -----------------------------
# UI
# -----------------------------
st.title("Football Team Dashboard (Equipos)")

with st.sidebar:
    st.header("Fuente y filtros")
    league_label = st.selectbox("Liga", list(LEAGUES.keys()), index=0)
    div = LEAGUES[league_label]
    season = st.text_input("Temporada (código)", value=DEFAULT_SEASON, help="Ejemplo: 2324 significa 2023-24 en football-data.co.uk")
    show_raw = st.checkbox("Mostrar tabla de partidos (raw)", value=False)

    st.caption("Los datos se descargan desde Football-Data.co.uk y se cachean localmente para acelerar recargas.")

# Load data
try:
    matches = load_matches(season.strip(), div)
except Exception as e:
    st.error(
        f"No se pudo cargar la data. Verifique el código de temporada y la liga.\n\nDetalle: {e}"
    )
    st.stop()

# Basic computed tables
table = compute_table(matches)
home_tbl, away_tbl = compute_home_away_tables(matches)

teams = table["Team"].tolist()
colA, colB = st.columns([2, 1], vertical_alignment="top")

with colA:
    st.subheader("Tabla de posiciones")
    st.dataframe(table, use_container_width=True, hide_index=True)

with colB:
    st.subheader("Resumen rápido")
    st.metric("Partidos en dataset", value=int(len(matches)))
    st.metric("Equipos", value=int(table["Team"].nunique()))
    st.metric("Goles totales", value=int(matches["FTHG"].sum() + matches["FTAG"].sum()))
    st.metric("Promedio goles/partido", value=round((matches["FTHG"].sum() + matches["FTAG"].sum()) / max(len(matches), 1), 2))

st.divider()

st.subheader("Local vs. visita")
c1, c2 = st.columns(2, vertical_alignment="top")
with c1:
    st.caption("Tabla solo de local")
    st.dataframe(home_tbl, use_container_width=True, hide_index=True)
with c2:
    st.caption("Tabla solo de visita")
    st.dataframe(away_tbl, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Análisis por equipo")
team = st.selectbox("Seleccione un equipo", teams, index=0)

log = build_team_match_log(matches, team)
if log.empty:
    st.info("No hay partidos para el equipo seleccionado en esta temporada/liga.")
    st.stop()

k = st.slider("Ventana de forma (últimos N partidos)", min_value=3, max_value=10, value=5, step=1)
form = recent_form_string(log, k)

# Team KPIs
pts = int(log["Pts_team"].sum())
gf = int(log["GF_team"].sum())
ga = int(log["GA_team"].sum())
gd = gf - ga
mp = int(len(log))
ppg = round(pts / mp, 2) if mp > 0 else 0.0

#textual summary

pos = int(table.loc[table["Team"] == team, "Pos"].iloc[0])
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Posición", pos)
c2.metric("Puntos", pts)
c3.metric("Goles a favor", gf)
c4.metric("Goles en contra", ga)
c5.metric("Forma", form)

# Plots
p1, p2 = st.columns(2, vertical_alignment="top")


with p1:
    st.caption("Evolución de puntos acumulados")
    chart_df = log[["MatchNo", "CumPts"]].copy().set_index("MatchNo")
    st.line_chart(chart_df)

with p2:
    st.caption("Goles a favor y en contra por partido")
    gg = log[["MatchNo", "GF_team", "GA_team"]].copy().set_index("MatchNo")
    st.bar_chart(gg)

st.caption("Bitácora de partidos del equipo (para exportar o auditar)")
st.dataframe(log, use_container_width=True, hide_index=True)

if show_raw:
    st.divider()
    st.subheader("Partidos (raw)")
    st.dataframe(matches.sort_values("Date") if matches["Date"].notna().any() else matches, use_container_width=True, hide_index=True)
