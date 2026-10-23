
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict

st.set_page_config(page_title="AI EPL Simulator", page_icon="⚽")
st.title("⚽ AI Premier League Season Simulator")
st.write("Train AI on 5 previous seasons and simulate 100 possible seasons.")

SEASONS = {
    "2021/22": "2122",
    "2022/23": "2223",
    "2023/24": "2324",
    "2024/25": "2425",
    "2025/26": "2526"
}

@st.cache_data
def load_data():
    frames = []
    for season, code in SEASONS.items():
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        needed = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        df = df.dropna(subset=needed)
        df["Season"] = season
        frames.append(df[needed + ["Season"]])

    return pd.concat(frames, ignore_index=True)

try:
    data = load_data()
except Exception as e:
    st.error("Could not download the football data. Check your internet connection and try again.")
    st.caption(str(e))
    st.stop()

data["FTHG"] = pd.to_numeric(data["FTHG"], errors="coerce")
data["FTAG"] = pd.to_numeric(data["FTAG"], errors="coerce")
data = data.dropna(subset=["FTHG", "FTAG", "FTR"])
data = data[data["FTR"].isin(["H", "D", "A"])]

# Build historical form features using only earlier matches
stats = defaultdict(lambda: [0, 0, 0])  # goals for, against, matches
features = []
labels = []

for _, row in data.iterrows():
    home = row["HomeTeam"]
    away = row["AwayTeam"]

    h = stats[home]
    a = stats[away]

    features.append([
        h[0] / h[2] if h[2] else 1.4,
        h[1] / h[2] if h[2] else 1.4,
        a[0] / a[2] if a[2] else 1.4,
        a[1] / a[2] if a[2] else 1.4
    ])
    labels.append(row["FTR"])

    hg, ag = int(row["FTHG"]), int(row["FTAG"])
    stats[home] = [h[0] + hg, h[1] + ag, h[2] + 1]
    stats[away] = [a[0] + ag, a[1] + hg, a[2] + 1]

X = pd.DataFrame(features, columns=["HomeGF", "HomeGA", "AwayGF", "AwayGA"])
y = labels

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)

st.success(f"AI trained using {len(data)} real historical matches.")

# Teams from the latest completed season in the dataset
latest = data[data["Season"] == "2025/26"]
teams = sorted(set(latest["HomeTeam"]) | set(latest["AwayTeam"]))

if len(teams) < 2:
    st.error("Not enough teams found in the latest season.")
    st.stop()

st.subheader("League settings")
simulations = st.number_input(
    "Number of seasons to simulate",
    min_value=10, max_value=1000, value=100, step=10
)

st.write(f"Teams: {len(teams)}")
st.write(f"Matches per simulated season: {len(teams) * (len(teams) - 1)}")

# Historical scorelines used to create realistic simulated scores
score_data = data[["FTR", "FTHG", "FTAG"]].copy()
score_data["FTHG"] = score_data["FTHG"].astype(int)
score_data["FTAG"] = score_data["FTAG"].astype(int)

# Start each simulation with latest-season team form
def starting_stats():
    result = defaultdict(lambda: [0, 0, 0])
    for _, row in latest.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        hg, ag = int(row["FTHG"]), int(row["FTAG"])
        result[home][0] += hg
        result[home][1] += ag
        result[home][2] += 1
        result[away][0] += ag
        result[away][1] += hg
        result[away][2] += 1
    return result

def simulate_season():
    form = starting_stats()
    table = {
        t: {"Played": 0, "Won": 0, "Drawn": 0, "Lost": 0,
            "GF": 0, "GA": 0, "Points": 0}
        for t in teams
    }

    fixtures = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            fixtures.append((teams[i], teams[j]))
            fixtures.append((teams[j], teams[i]))

    np.random.shuffle(fixtures)

    for home, away in fixtures:
        h, a = form[home], form[away]
        row = [[
            h[0] / h[2] if h[2] else 1.4,
            h[1] / h[2] if h[2] else 1.4,
            a[0] / a[2] if a[2] else 1.4,
            a[1] / a[2] if a[2] else 1.4
        ]]

        probabilities = model.predict_proba(row)[0]
        outcome = np.random.choice(model.classes_, p=probabilities)

        # Pick a real historical score with the predicted outcome
        possible = score_data[score_data["FTR"] == outcome]
        score = possible.sample(1).iloc[0]
        hg, ag = int(score["FTHG"]), int(score["FTAG"])

        # Update league table
        for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
            table[team]["Played"] += 1
            table[team]["GF"] += gf
            table[team]["GA"] += ga

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

        # Update team form with simulated result
        form[home] = [h[0] + hg, h[1] + ag, h[2] + 1]
        form[away] = [a[0] + ag, a[1] + hg, a[2] + 1]

    result = pd.DataFrame.from_dict(table, orient="index")
    result.index.name = "Team"
    result["GD"] = result["GF"] - result["GA"]
    return result.sort_values(
        ["Points", "GD", "GF"], ascending=False
    )

if st.button("🚀 Simulate League Season"):
    title_counts = dict.fromkeys(teams, 0)
    example_table = None

    progress = st.progress(0, text="Simulating seasons...")

    for i in range(int(simulations)):
        result = simulate_season()
        title_counts[result.index[0]] += 1
        if i == 0:
            example_table = result.reset_index()
        progress.progress((i + 1) / simulations)

    st.subheader("🏆 Most likely champion")
    chances = pd.DataFrame({
        "Team": list(title_counts.keys()),
        "Title Chance (%)": [
            round(v / simulations * 100, 1)
            for v in title_counts.values()
        ]
    }).sort_values("Title Chance (%)", ascending=False)

    winner = chances.iloc[0]
    st.success(
        f"Predicted champion: {winner['Team']} "
        f"({winner['Title Chance (%)']}% of simulations)"
    )

    st.subheader("Title-winning probabilities")
    st.bar_chart(chances.set_index("Team")["Title Chance (%)"])
    st.dataframe(chances, hide_index=True, use_container_width=True)

    st.subheader("Example simulated league table")
    example_table.insert(0, "Position", range(1, len(example_table) + 1))
    st.dataframe(example_table, hide_index=True, use_container_width=True)

    st.caption(
        "This is an educational simulation, not a guaranteed prediction. "
        "It uses historical match results and simulated outcomes."
    )

st.caption("Historical data source: football-data.co.uk")
