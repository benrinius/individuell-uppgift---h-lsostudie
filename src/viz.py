import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def hist_bp(df, bp_col="systolic_bp", bins=20):
    """"Histogram blodtryck"""
    df2 = df.copy()
    ser = pd.to_numeric(df2[bp_col], errors="coerce")
    plt.figure(figsize=(7,4))
    plt.hist(ser.dropna(), bins=bins)
    plt.xlabel("Systoliskt blodtryck")
    plt.ylabel("Antal deltagare")
    plt.title("Histogram: systoliskt blodtryck")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

def box_weight_by_sex(df, weight_col="weight", sex_col="sex"):
    """"Boxplot vikt per kön"""
    df2 = df.copy()
    df2[weight_col] = pd.to_numeric(df2[weight_col], errors="coerce")
    plt.figure(figsize=(6,5))
    df2.boxplot(column=weight_col, by=sex_col)
    plt.suptitle("")
    plt.title("Vikt per kön")
    plt.xlabel("Kön")
    plt.ylabel("Vikt (kg)")
    plt.show()

def bar_smokers(df, smoke_col="smoker"):
    """"Stapel andel rökare"""
    df2 = df.copy()
    counts = df2[smoke_col].value_counts()
 
    plt.figure(figsize=(5,4))
    plt.bar(counts.index, counts.values)
    plt.ylabel("Antal deltagare")
    plt.title("Andel rökare")
    plt.grid(axis="y", alpha=0.3)
    plt.show()

def plot_age_vs_bp(df):
    """
    Scatter för relationen mellan ålder och systoliskt blodtryck med en linjär regressionslinje
    """
    x = df["age"].astype(float).values.reshape(-1,1)
    y = df["systolic_bp"].astype(float).values

    model = LinearRegression()
    model.fit(x,y)
    y_pred = model.predict(x)

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, alpha=0.4, label="individer")
    plt.plot(x, y_pred, color="red", linewidth=2, label="Regressionslinje")
    plt.xlabel("Ålder")
    plt.ylabel("Systoliskt blodtryck")
    plt.title("Relation mellan ålder och blodtryck")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()