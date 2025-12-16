import numpy as np
import pandas as pd
from . io_utils import load_data
from . import metrics, viz

class HealthAnalyzer:
    """En klass som anropar alla funkti"""
    def __init__(self, path="data/health_study_dataset.csv", seed=42):
        np.random.seed(seed)
        self.df = load_data(path)
        self.seed = seed

    def describe(self):
        """Returnera DataFrame med mean, median, min, max för valda kolumner."""
        cols = ["age", "weight", "height", "systolic_bp", "cholesterol"]
        return metrics.describe_stats(self.df, cols)

    def prevalence(self):
        """Andel deltagare med sjukdomen (0/1 eller True/False)"""
        return metrics.disease_prevalence(self.df, disease_col="disease")

    def simulate(self, n_sim=1000, n_people=1000):
        """Simulera n_sim gånger andel personer med sjukdom i n_people 
        med sannolikhet p (om p är None måste användaren ange).
        Returnerar array med andelar."""
        p, _ = self.prevalence()
        sims = metrics.simulate_disease_prop(n_sim=n_sim, n_people=n_people, p=p, seed=self.seed)
        return sims

    def bp_confidence_interval(self, n_boot=2000, alpha=0.05):
        """bootstrap-konfidensintervall för medelvärdet av systoliskt blodtryck."""
        bp = self.df["systolic_bp"].astype(float)
        return metrics.bootstrap_ci_mean(bp, n_boot=n_boot, alpha=alpha, seed=self.seed)

    def test_smoking_bp(self):
        """t-test rökare/ icke-rökare"""
        return metrics.t_test_smokers_vs_nonsmokers(self.df, bp_col="systolic_bp", smoke_col="smoker")
    
    def linear_regression(self):
        """returnerar linear_regression_bp från metrics som förutsäger blodtryck från ålder och vikt"""
        return metrics.linear_regression_bp(self.df)

    # För smidighetens skull
    def plot_all(self):
        """"Histogram blodtryck, boxplot vikt per kön, stapel andel rökare, scatter för relation mellan ålder och blodtryck"""
        viz.hist_bp(self.df)
        viz.box_weight_by_sex(self.df)
        viz.bar_smokers(self.df)

    def plot_age_vs_bp(self):
        viz.plot_age_vs_bp(self.df)



    def summary(self):
        """"Snabb översikt över viktiga nykeltal """
        ha = HealthAnalyzer()
        bp = ha.df["systolic_bp"].astype(float)

        print("Blodtryck – sammanfattning:")
        print(f"Medelvärde: {bp.mean():.2f}")
        print(f"Median: {bp.median():.2f}")
        print(f"Standardavvikelse: {bp.std():.2f}")
        print(f"Lägsta: {bp.min():.2f}")
        print(f"Högsta: {bp.max():.2f}")

        g = ha.df.groupby("sex")["weight"].agg(["mean", "median", "std", "min", "max", "count"])
        g

        sm = ha.df["smoker"].astype(str).str.lower().str.strip()
        counts = sm.value_counts()
        props = sm.value_counts(normalize=True) * 100

        print("Antal rökare / icke-rökare:")
        print(counts)
        print("\nAndelar (%):")
        print(props.round(2))

        p_real, n_real = ha.prevalence()
        print(f"Verklig andel med sjukdomen: {p_real:.3f} (n={n_real})")

        sims = ha.simulate(n_sim=1000, n_people=n_real)

        print("Simulering av andel med sjukdom (1000 simuleringar):")
        print(f"Medelvärde: {sims.mean():.3f}")
        print(f"Std: {sims.std():.3f}")
        print(f"5:e percentilen: {np.percentile(sims, 5):.3f}")
        print(f"95:e percentilen: {np.percentile(sims, 95):.3f}")

        lo, hi, boots = ha.bp_confidence_interval(n_boot=2000, alpha=0.05)
        print(f"Bootstrap 95% CI: [{lo:.2f}, {hi:.2f}]")

        print(f"Bootstrap-medelvärde: {boots.mean():.2f}")
        print(f"Bootstrap-std: {boots.std():.2f}")

        tstat, p_one, n1, n0 = ha.test_smoking_bp()
        print(tstat, p_one)

        df2 = ha.df.copy()
        df2["smoker"] = df2["smoker"].astype(str).str.lower().str.strip()
        df2["systolic_bp"] = pd.to_numeric(df2["systolic_bp"], errors="coerce")

        g1 = df2.loc[df2["smoker"] == "yes", "systolic_bp"]
        g0 = df2.loc[df2["smoker"] == "no", "systolic_bp"]

        print("Rökare:")
        print(f"  Medelvärde: {g1.mean():.2f}, n={len(g1)}")

        print("Icke-rökare:")
        print(f"  Medelvärde: {g0.mean():.2f}, n={len(g0)}")

        print(f"\nT-stat: {tstat:.3f}, p-värde (ensidigt): {p_one:.4f}")