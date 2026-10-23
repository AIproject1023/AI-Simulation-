import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from itertools import combinations

st.set_page_config(page_title="AI EPL Season Simulator", page_icon="⚽", layout="wide")
st.title("⚽ AI Premier League Season Simulator")
st.write("Train a machine-learning model on five previous EPL seasons and simulate 100 possible seasons.")

SEASONS = {
    "2021/22": "2122",
    "2022/23": "2223",
    "2023/24": "2324",
    "2024/25": "2425",
    "2025/26": "2526",
}

@st.cache_data(show_spinner="Downloading historical EPL results...")
def load_data():
    frames = []
    for season, code in SEASONS.items():
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{season} CSV is missing columns: {', '.join(missing)}")
        df = df[required].copy()
        df["Season"] = season
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=required)
        df = df[df["FTR"].isin(["H", "D", "A"])]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

try:
    data = load_data()
except Exception as e:
    st.error("Couldn't download the historical data. Check the app's internet connection and try again.")
    st.caption(f"Technical detail: {e}")
    st.stop()

data["FTHG"] = data["FTHG"].astype(int)
data["FTAG"] = data["FTAG"].astype(int)
data = data.sort_values(["Season"]).reset_index(drop=True)

# Build form features before each match, so the model never sees that match's result.
team_stats = defaultdict(lambda: {"gf": 0, "ga": 0, "matches": 0})
X_rows, y_rows = [], []

for _, match in data.iterrows():
    home, away = match["HomeTeam"], match["AwayTeam"]
    hs, aws = team_stats[home], team_stats[away]

    def averages(s):
        if s["matches"] == 0:
            return 1.4, 1.4
        return s["gf"] / s["matches"], s["ga"] / s["matches"]

    hgf, hga = averages(hs)
    agf, aga = averages(aws)
    X_rows.append([hgf, hga, agf, aga])
    y_rows.append(match["FTR"])

    hg, ag = int(match["FTHG"]), int(match["FTAG"])
    hs["gf"] += hg
    hs["ga"] += ag
    hs["matches"] += 1
    aws["gf"] += ag
    aws["ga"] += hg
    aws["matches"] += 1

X = pd.DataFrame(X_rows, columns=["HomeGF", "HomeGA", "AwayGF", "AwayGA"])
y = pd.Series(y_rows)

model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42,
                               class_weight="balanced")
model.fit(X, y)

st.success(f"AI trained on {len(data):,} real historical matches from {len(SEASONS)} seasons.")
st.caption("Source: Football-Data.co.uk. Historical results are used to learn outcome patterns.")

# The latest season in the chosen data provides the league's team list.
latest_season = "2025/26"
latest = data[data["Season"] == latest_season].copy()
teams = sorted(set(latest["HomeTeam"]) | set(latest["AwayTeam"]))

if len(teams) < 2:
    st.error("Not enough teams were found in the latest season.")
    st.stop()

left, right = st.columns(2)
with left:
    simulations = st.number_input("Seasons to simulate", min_value=10, max_value=1000,
                                  value=100, step=10)
with right:
    st.metric("Teams", len(teams))
    st.metric("Matches per season", len(teams) * (len(teams) - 1))

# Start simulated seasons with each team's actual record from the latest season.
def initial_form():
    form = defaultdict(lambda: {"gf": 0, "ga": 0, "matches": 0})
    for _, m in latest.iterrows():
        home, away = m["HomeTeam"], m["AwayTeam"]
        hg, ag = int(m["FTHG"]), int(m["FTAG"])
        form[home]["gf"] += hg
        form[home]["ga"] += ag
        form[home]["matches"] += 1
        form[away]["gf"] += ag
        form[away]["ga"] += hg
        form[away]["matches"] += 1
    return form

def team_averages(s):
    if s["matches"] == 0:
        return 1.4, 1.4
    return s["gf"] / s["matches"], s["ga"] / s["matches"]

# Real historical scorelines are sampled after the AI predicts win/draw/loss.
scorelines = {
    outcome: data.loc[data["FTR"] == outcome, ["FTHG", "FTAG"]].to_numpy()
    for outcome in ["H", "D", "A"]
}

def simulate_season(seed):
    rng = np.random.default_rng(seed)
    form = initial_form()
    table = {
        t: {"Played": 0, "Won": 0, "Drawn": 0, "Lost": 0,
            "GF": 0, "GA": 0, "Points": 0}
        for t in teams
    }

    # Every pair plays twice: once at each team's home ground.
    fixtures = [(h, a) for h, a in combinations(teams, 2)]
    fixtures += [(a, h) for h, a in combinations(teams, 2)]
    rng.shuffle(fixtures)
    played = []

    for home, away in fixtures:
        hgf, hga = team_averages(form[home])
        agf, aga = team_averages(form[away])
        row = pd.DataFrame([[hgf, hga, agf, aga]], columns=X.columns)

        probabilities = model.predict_proba(row)[0]
        outcome = rng.choice(model.classes_, p=probabilities)
        scores = scorelines[outcome]
        hg, ag = scores[rng.integers(0, len(scores))]
        hg, ag = int(hg), int(ag)

        played.append({"Home Team": home, "Home Goals": hg,
                       "Away Goals": ag, "Away Team": away})

        table[home]["Played"] += 1
        table[away]["Played"] += 1
        table[home]["GF"] += hg
        table[home]["GA"] += ag
        table[away]["GF"] += ag
        table[away]["GA"] += hg

        if hg > ag:
            table[home]["Won"] += 1
            table[away]["Lost"] += 1
            table[home]["Points"] += 3
        elif hg < ag:
            table[away]["Won"] += 1
            table[home]["Lost"] += 1
            table[away]["Points"] += 3
        else:
            table[home]["Drawn"] += 1
            table[away]["Drawn"] += 1
            table[home]["Points"] += 1
            table[away]["Points"] += 1

        # Update form only after predicting and generating this match.
        form[home]["gf"] += hg
        form[home]["ga"] += ag
        form[home]["matches"] += 1
        form[away]["gf"] += ag
        form[away]["ga"] += hg
        form[away]["matches"] += 1

    result = pd.DataFrame.from_dict(table, orient="index")
    result.index.name = "Team"
    result["GD"] = result["GF"] - result["GA"]
    result = result.sort_values(["Points", "GD", "GF"], ascending=False).reset_index()
    result.insert(0, "Position", range(1, len(result) + 1))
    return result, pd.DataFrame(played)

if st.button("🚀 Simulate league season", type="primary"):
    title_counts = dict.fromkeys(teams, 0)
    example_table, example_fixtures = None, None
    progress = st.progress(0, text="Simulating seasons...")

    for i in range(int(simulations)):
        table, fixtures = simulate_season(seed=1000 + i)
        title_counts[table.iloc[0]["Team"]] += 1
        if i == 0:
            example_table = table
            example_fixtures = fixtures
        progress.progress((i + 1) / int(simulations), text=f"Simulating season {i + 1}/{int(simulations)}")

    chances = pd.DataFrame({
        "Team": list(title_counts.keys()),
        "Titles Won": list(title_counts.values()),
    })
    chances["Title Chance (%)"] = (chances["Titles Won"] / int(simulations) * 100).round(1)
    chances = chances.sort_values(["Title Chance (%)", "Titles Won"], ascending=False).reset_index(drop=True)

    st.subheader("🏆 Predicted champion")
    winner = chances.iloc[0]
    st.success(f"{winner['Team']} — {winner['Title Chance (%)']}% title chance in these simulations")
    st.caption("This is a simulation estimate, not a guarantee about the real season.")

    st.subheader("Title-winning probabilities")
    st.bar_chart(chances.set_index("Team")["Title Chance (%)"])
    st.dataframe(chances, hide_index=True, use_container_width=True)

    st.subheader("Example simulated league table")
    st.dataframe(example_table, hide_index=True, use_container_width=True)

    st.subheader("Example simulated fixtures and results")
    st.dataframe(example_fixtures, hide_index=True, use_container_width=True)

    csv = example_fixtures.to_csv(index=False).encode("utf-8")
    st.download_button("Download simulated fixtures as CSV", csv,
                       "simulated_epl_fixtures.csv", "text/csv")

st.divider()
st.caption("Educational Class 12 project | Python, Streamlit, pandas, NumPy and scikit-learn")
