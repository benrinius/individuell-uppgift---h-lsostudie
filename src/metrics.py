import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression    

def describe_stats(df: pd.DataFrame, cols):
    """Returnera DataFrame med mean, median, min, max för valda kolumner."""
    out = []
    for c in cols:
        ser = pd.to_numeric(df[c], errors = "coerce")
        out.append({
            "variable": c,
            "mean": ser.mean(),
            "median": ser.median(),
            "min": ser.min(),
            "max": ser.max(),
            "count": ser.count()
        })
    return pd.DataFrame(out).set_index("variable")

def disease_prevalence(df: pd.DataFrame, disease_col="disease"):
    """Andel deltagare med sjukdomen (0/1 eller True/False)"""
    ser = pd.to_numeric(df[disease_col], errors="coerce").copy().dropna()
    ser = ser.map({True:1, False:0, "Yes":1, "No":0}).fillna(ser)  # om True/False ######################
    p = (ser == 1).mean()
    n = len(ser)
    return p, n

def simulate_disease_prop(n_sim=1000, n_people=1000, p=None, seed=42):
    """
    Simulera n_sim gånger andel personer med sjukdom i n_people,
    med sannolikhet p (om p är None måste användaren ange).
    Returnerar array med andelar.
    """
    rng = np.random.RandomState(seed)
    sims = rng.binomial(n_people, p, size=n_sim) / n_people
    return sims

def bootstrap_ci_mean(x, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap-konfidensintervall för medelvärde av x."""
    rng = np.random.RandomState(seed)
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    n = len(x)
    boots = rng.choice(x, size=(n_boot, n), replace=True)
    boot_means = boots.mean(axis=1)

    lo = np.percentile(boot_means, 100 * (alpha/2))
    hi = np.percentile(boot_means, 100 * (1 - alpha/2))
    return lo, hi, boot_means

def t_test_smokers_vs_nonsmokers(df, bp_col="systolic_bp", smoke_col="smoker"):
    """t-test rökare/ icke-rökare"""
    import numpy as np
    from scipy import stats

    # Kopiera och säkerställ strängar
    df2 = df.copy()
    df2[smoke_col] = df2[smoke_col].astype(str).str.strip().str.lower()
    
    # Konvertera blodtryck till numeriskt
    df2[bp_col] = pd.to_numeric(df2[bp_col], errors='coerce')

    # Filtrera bort rader där någon kolumn är NaN
    df2 = df2.dropna(subset=[bp_col, smoke_col])

    g1 = df2.loc[df2[smoke_col] == 'yes', bp_col]
    g0 = df2.loc[df2[smoke_col] == 'no', bp_col]

    # Kontrollera att grupperna inte är tomma
    if len(g1) < 2 or len(g0) < 2:
        return np.nan, np.nan, len(g1), len(g0)

    # Welch's t-test
    tstat, p_two = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')

    # One-sided test
    p_one = p_two / 2 if tstat > 0 else 1 - p_two / 2

    return tstat, p_one, len(g1), len(g0)


def linear_regression_bp(df, features=("age", "weight"), target="systolic_bp"):
    """
    Linjär regression för att förutsäga blodtryck från ålder och vikt
    """

    df2 = df.copy()

    x = df2[list(features)].astype(float).values
    y = df2[target].astype(float).values

    model = LinearRegression()
    model.fit(x,y)

    r2 = model.score(x,y)

    return model, model.coef_, model.intercept_, r2