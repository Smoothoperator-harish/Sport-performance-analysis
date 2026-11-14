
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#MODULE 1 : DATA INPUT MODULE

print("=" * 55)
print("     SPORTS PERFORMANCE ANALYSER")
print("=" * 55)
print()
print("MODULE 1 : Data Input")
print()

# read the csv file into a dataframe
df = pd.read_csv("players_15.csv")

print("Dataset loaded successfully.")
print("Total players in file :", len(df))
print("Total columns         :", len(df.columns))
print()

# show a preview of the first 5 rows
print("Preview of raw data (first 5 rows) :")
print(df[["short_name", "nationality", "club", "overall",
          "pace", "shooting", "passing",
          "dribbling", "defending", "physic"]].head().to_string(index=False))
print()


# MODULE 2 : DATA PROCESSING MODULE

print("-" * 55)
print("MODULE 2 : Data Processing")
print("-" * 55)
print()

# some columns have text like "90+2" instead of plain numbers
# pd.to_numeric forces them to numbers, bad values become NaN
df["overall"]   = pd.to_numeric(df["overall"],   errors="coerce")
df["pace"]      = pd.to_numeric(df["pace"],       errors="coerce")
df["shooting"]  = pd.to_numeric(df["shooting"],   errors="coerce")
df["passing"]   = pd.to_numeric(df["passing"],    errors="coerce")
df["dribbling"] = pd.to_numeric(df["dribbling"],  errors="coerce")
df["defending"] = pd.to_numeric(df["defending"],  errors="coerce")
df["physic"]    = pd.to_numeric(df["physic"],     errors="coerce")
df["potential"] = pd.to_numeric(df["potential"],  errors="coerce")
df["age"]       = pd.to_numeric(df["age"],        errors="coerce")

# goalkeepers don't have pace or shooting, they show as NaN
# keep only outfield players where pace has a value
players = df[df["pace"].notna()].copy()

print("Data cleaned successfully.")
print("Outfield players kept :", len(players))
print("Goalkeepers removed   :", len(df) - len(players))
print()

# check for missing values in the key columns
key_cols = ["overall", "pace", "shooting", "passing",
            "dribbling", "defending", "physic"]
print("Missing values check :")
print(players[key_cols].isnull().sum())
print()

# map each position code to a group name
# this lets us compare forwards vs midfielders vs defenders
position_map = {
    "ST" : "Forward",    "CF" : "Forward",
    "LW" : "Forward",    "RW" : "Forward",
    "CM" : "Midfielder", "CAM": "Midfielder",
    "CDM": "Midfielder", "LM" : "Midfielder", "RM": "Midfielder",
    "CB" : "Defender",   "LB" : "Defender",
    "RB" : "Defender",   "LWB": "Defender",   "RWB": "Defender",
}

# take first position listed for each player, map it to a group
players["position_group"] = (
    players["player_positions"]
    .str.split(",")
    .str[0]
    .str.strip()
    .map(position_map)
)

# drop players where position didn't match any group
players = players[players["position_group"].notna()]

print("Position groups created :")
print(players["position_group"].value_counts().to_string())
print()



# MODULE 3 : ANALYSIS MODULE

print("-" * 55)
print("MODULE 3 : Analysis")
print("-" * 55)
print()

# --- 3a : statistical summary of all key stats ---
print("--- Overall Statistical Summary (describe) ---")
print(players[key_cols].describe().round(1).to_string())
print()

# --- 3b : top 10 athletes by overall rating ---
print("--- Top 10 Athletes by Overall Rating ---")
top10 = players.nlargest(10, "overall")[
    ["short_name", "club", "nationality", "overall",
     "pace", "shooting", "dribbling"]
]
print(top10.to_string(index=False))
print()

# --- 3c : compare performance across position groups using groupby ---
print("--- Average Stats by Position Group (groupby + mean) ---")
stat_cols = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
position_stats = players.groupby("position_group")[stat_cols].mean().round(1)
print(position_stats.to_string())
print()

# --- 3d : top 5 nations by average overall rating ---
print("--- Top 5 Nations by Average Overall Rating ---")
nation_stats = (
    players.groupby("nationality")["overall"]
    .agg(["mean", "count"])
    .query("count >= 50")           # only nations with 50 or more players
    .sort_values("mean", ascending=False)
    .head(5)
    .round(1)
)
nation_stats.columns = ["Avg Overall", "Player Count"]
print(nation_stats.to_string())
print()

# --- 3e : rising talents - young players with big potential ---
print("--- Top 5 Rising Talents (Age 21 or under) ---")
players["potential_gap"] = players["potential"] - players["overall"]
young = players[players["age"] <= 21]
talents = young.nlargest(5, "potential_gap")[
    ["short_name", "age", "club", "overall", "potential", "potential_gap"]
]
print(talents.to_string(index=False))
print()

# --- 3f : numpy stats on speed (pace) and score (overall) ---
print("--- NumPy Stats ---")
pace_vals    = players["pace"].values
overall_vals = players["overall"].values

print("Speed (Pace)  — Mean:", round(np.mean(pace_vals),    1),
      " | Std:", round(np.std(pace_vals),    1),
      " | Max:", int(np.max(pace_vals)))

print("Score (Overall) — Mean:", round(np.mean(overall_vals), 1),
      " | Std:", round(np.std(overall_vals), 1),
      " | Max:", int(np.max(overall_vals)))
print()



#MODULE 4 : INSIGHT & OUTPUT MODULE

print("-" * 55)
print("MODULE 4 : Insight & Output")
print("-" * 55)
print()

# --- 4a : strengths and weaknesses per position group ---
print("--- Strengths and Areas for Improvement ---")
print()

for group, row in position_stats.iterrows():
    best_stat  = row.idxmax()   # stat with highest average
    worst_stat = row.idxmin()   # stat with lowest average
    print(f"  {group:<12} Strength : {best_stat:<12} | Needs Improvement : {worst_stat}")

print()

# --- 4b : coaching decisions ---
print("--- Coaching Decisions ---")
print()

coaching = {
    "Forward"   : "Work on defensive positioning and tracking back to support the team.",
    "Midfielder": "Improve physical strength to win more duels in the middle of the pitch.",
    "Defender"  : "Focus on shooting and ball distribution from the back.",
}

for group, advice in coaching.items():
    weak = position_stats.loc[group].idxmin()
    print(f"  {group:<12} Weakness : {weak:<12}")
    print(f"               Advice   : {advice}")
    print()

# ---- charts ----

print("-" * 55)
print("Drawing Charts  (close each window to see the next one)")
print("-" * 55)

BG = "#f0f7f0"


# chart 1 - top 10 players horizontal bar chart

top10_plot = players.nlargest(10, "overall").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# [::-1] reverses so highest rated player appears at the top
ax.barh(top10_plot["short_name"][::-1],
        top10_plot["overall"][::-1],
        color="steelblue", edgecolor="white")

ax.set_xlabel("Overall Rating")
ax.set_title("Top 10 Athletes by Overall Rating")
ax.set_xlim(84, 95)

plt.tight_layout()
plt.show()
print("  Chart 1 displayed - Top 10 Athletes")


# chart 2 - average attributes by position group (grouped bar)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

groups    = position_stats.index.tolist()
x         = np.arange(len(stat_cols))   # [0, 1, 2, 3, 4, 5]
bar_width = 0.25
colors    = ["tomato", "orange", "steelblue"]

for i, group in enumerate(groups):
    values = position_stats.loc[group, stat_cols].values
    ax.bar(x + i * bar_width, values, bar_width,
           label=group, color=colors[i], edgecolor="white")

ax.set_xticks(x + bar_width)
ax.set_xticklabels(["Pace", "Shooting", "Passing",
                    "Dribbling", "Defending", "Physic"])
ax.set_ylabel("Average Rating")
ax.set_title("Average Attributes — Forward vs Midfielder vs Defender")
ax.legend()

plt.tight_layout()
plt.show()
print("  Chart 2 displayed - Position Group Comparison")


# chart 3 - overall rating distribution histogram

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

avg_overall = round(players["overall"].mean(), 1)

ax.hist(players["overall"], bins=40,
        color="steelblue", edgecolor="white")
# red dashed line at the average
ax.axvline(avg_overall, color="red", linestyle="--",
           label=f"Mean = {avg_overall}")

ax.set_xlabel("Overall Rating (Score)")
ax.set_ylabel("Number of Players")
ax.set_title("How Athlete Scores Are Distributed")
ax.legend()

plt.tight_layout()
plt.show()
print("  Chart 3 displayed - Score Distribution")


# chart 4 - speed vs score scatter plot

sample = players.sample(500, random_state=42)

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.scatter(sample["pace"], sample["overall"],
           alpha=0.4, color="steelblue", s=20)

ax.set_xlabel("Speed (Pace)")
ax.set_ylabel("Score (Overall)")
ax.set_title("Speed vs Score  (500 Random Athletes)")

plt.tight_layout()
plt.show()
print("  Chart 4 displayed - Speed vs Score")


# ============================================================
#   FINAL SUMMARY REPORT
# ============================================================

best_player  = players.loc[players["overall"].idxmax(), "short_name"]
best_score   = int(players["overall"].max())
best_talent  = talents.iloc[0]["short_name"]
best_gap     = int(talents.iloc[0]["potential_gap"])
top_nation   = nation_stats.index[0]

print()
print("=" * 55)
print("          FINAL SUMMARY REPORT")
print("=" * 55)
print(f"  Total athletes analysed  : {len(players):,}")
print(f"  Best athlete             : {best_player} (score {best_score})")
print(f"  Average score            : {avg_overall}")
print(f"  Average speed            : {round(np.mean(pace_vals), 1)}")
print(f"  Best nation              : {top_nation}")
print(f"  Top rising talent        : {best_talent} (gap +{best_gap})")
print(f"  Charts displayed         : 4")
print("=" * 55)
print()
print("Done! All 4 modules executed successfully.")
