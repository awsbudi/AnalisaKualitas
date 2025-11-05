import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import io

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="PMG - RM BTPN Syariah", layout="wide")
st.title("Risk Management BTPN Syariah")
st.title("Analysis Tools 📊")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Analysis"])

uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])
df = None
model = None
target = None
features = []

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

# -----------------------------
# MAIN PAGE
# -----------------------------
if page == "Upload & Analysis" and df is not None:
    st.write("### Preview Data")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Variable", df.columns)
    features = st.multiselect("Select Features", [col for col in df.columns if col != target])

    if st.button("Run Analysis"):
        df_clean = df[[target] + features].dropna()
        X = df_clean[features]
        X = sm.add_constant(X)
        y = df_clean[target]

        model = sm.OLS(y, X).fit()

        tab1, tab2, tab3 = st.tabs([
            "Regression Summary & Prediction",
            "Correlation Matrix",
            "Feature Strength & Binning",
        ])

        # ==============================================
        # TAB 1: REGRESSION SUMMARY + INTERPRETATION + PREDICTION
        # ==============================================
        with tab1:
            st.write("### 📊 OLS Regression Results")

            summary_df = pd.DataFrame({
                "Feature": model.params.index,
                "Coefficient": model.params.values,
                "Std Error": model.bse.values,
                "t-Statistic": model.tvalues.values,
                "P-Value": model.pvalues.values,
                "CI Lower": model.conf_int()[0].values,
                "CI Upper": model.conf_int()[1].values
            }).round(6)

            st.dataframe(summary_df, use_container_width=True)
            st.markdown(
                f"**R²:** {model.rsquared:.4f} | **Adjusted R²:** {model.rsquared_adj:.4f} | **F-statistic:** {model.fvalue:.4f}"
            )

            # Download
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Regression Summary", index=False)
            st.download_button("📥 Download Results (Excel)", excel_buffer.getvalue(), "analysis_results.xlsx")

            st.markdown("---")

            # INTERPRETASI
            st.subheader("🧠 Automatic Interpretation")
            coef = model.params.drop("const")
            top_pos = coef.sort_values(ascending=False).head(3)
            top_neg = coef.sort_values().head(3)

            interpretation = ""
            interpretation += "**Fitur dengan pengaruh POSITIF terbesar:**\n"
            for f, c in top_pos.items():
                interpretation += f"- {f}: koefisien +{c:.6f}\n"
            interpretation += "\n**Fitur dengan pengaruh NEGATIF terbesar:**\n"
            for f, c in top_neg.items():
                interpretation += f"- {f}: koefisien {c:.6f}\n"
            interpretation += "\n**💡 Implikasi:** Fitur dengan koefisien positif tinggi cenderung meningkatkan nilai target. Fitur negatif signifikan perlu dikendalikan atau dioptimasi."

            st.markdown(interpretation)
            st.markdown("---")

            # PREDIKSI
            st.subheader("🎯 Predict Target Value (Interactive Simulation)")
            st.caption("Masukkan nilai fitur untuk mensimulasikan hasil prediksi model OLS.")

            input_values = {}
            col_inputs = st.columns(3)
            for i, f in enumerate(features):
                with col_inputs[i % 3]:
                    input_values[f] = st.number_input(
                        f"Nilai {f}",
                        value=float(df[f].mean()),
                        step=0.1,
                        format="%.3f"
                    )

            if st.button("🔮 Hitung Prediksi"):
                X_pred = pd.DataFrame([input_values])
                X_pred = sm.add_constant(X_pred, has_constant='add')
                prediction = model.predict(X_pred)[0]

                # tampilkan hasil
                st.success(f"**Prediksi {target}: {prediction:.6f}**")

                # progress bar
                avg_target = df[target].mean()
                min_target = df[target].min()
                max_target = df[target].max()

                norm_value = (prediction - min_target) / (max_target - min_target)
                norm_value = max(0, min(1, norm_value))  # clamp 0-1

                # indikator warna
                if norm_value < 0.33:
                    color = "🔴 Rendah"
                elif norm_value < 0.66:
                    color = "🟡 Sedang"
                else:
                    color = "🟢 Tinggi"

                st.progress(norm_value)
                st.markdown(f"**Indikator:** {color}  \n*(dibandingkan range historis target di dataset)*")

        # ==============================================
        # TAB 2: CORRELATION MATRIX
        # ==============================================
        with tab2:
            st.write("### 🔗 Correlation Matrix")
            corr = df_clean.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # ==============================================
        # TAB 3: FEATURE IMPORTANCE + BINNING COMPARISON
        # ==============================================
        with tab3:
            st.subheader("🌳 Feature Strength & Decision Tree Binning (Interactive)")

            X_features = df_clean[features]
            y_target = df_clean[target]

            # ---------- METODE 1: OLS ----------
            coef = model.params.drop("const")
            abs_coef = coef.abs().sort_values(ascending=False)
            strongest_feature_ols = abs_coef.index[0]
            r2_ols = model.rsquared

            # ---------- METODE 2: Decision Tree ----------
            tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
            tree_model.fit(X_features, y_target)
            importances_tree = pd.Series(tree_model.feature_importances_, index=features)
            strongest_feature_tree = importances_tree.sort_values(ascending=False).index[0]
            r2_tree = r2_score(y_target, tree_model.predict(X_features))

            # ---------- METODE 3: Random Forest ----------
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf_model.fit(X_features, y_target)
            importances_rf = pd.Series(rf_model.feature_importances_, index=features)
            strongest_feature_rf = importances_rf.sort_values(ascending=False).index[0]
            r2_rf = r2_score(y_target, rf_model.predict(X_features))

            # ---------- RINGKASAN ----------
            compare_df = pd.DataFrame({
                "Metode": ["OLS Coefficient", "Decision Tree", "Random Forest"],
                "Fitur Terkuat": [strongest_feature_ols, strongest_feature_tree, strongest_feature_rf],
                "Kekuatan Model (R²)": [r2_ols, r2_tree, r2_rf]
            }).round(4)

            st.markdown("### 📊 Perbandingan Kekuatan Model & Fitur Terkuat")
            st.dataframe(compare_df, use_container_width=True)
            st.markdown("---")

            st.markdown("### 🔍 Visualisasi Binning (Interaktif)")
            st.caption("Hover tiap titik untuk melihat nilai fitur dan target. Garis merah = batas split dari pohon keputusan (max_depth=3).")

            # ---------- FUNGSI PEMBUAT PLOT INTERAKTIF ----------
            def make_interactive_binning_plot(feature_name, method_label):
                X_single = df_clean[[feature_name]]
                y_single = df_clean[target]
                tree_single = DecisionTreeRegressor(max_depth=3, random_state=42)
                tree_single.fit(X_single, y_single)
                thresholds = sorted(tree_single.tree_.threshold[tree_single.tree_.threshold > 0])

                fig = px.scatter(
                    df_clean,
                    x=feature_name,
                    y=target,
                    title=f"{method_label} ({feature_name})",
                    opacity=0.7,
                    labels={feature_name: feature_name, target: target},
                )
                for thr in thresholds:
                    fig.add_vline(x=thr, line_width=2, line_dash="dash", line_color="red")
                fig.update_layout(
                    height=400,
                    title_x=0.5,
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                return fig

            # ---------- 3 PLOT ----------
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**Metode 1️⃣: OLS Coefficient**  \nR² = {r2_ols:.3f}")
                fig1 = make_interactive_binning_plot(strongest_feature_ols, "OLS Coefficient")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.markdown(f"**Metode 2️⃣: Decision Tree**  \nR² = {r2_tree:.3f}")
                fig2 = make_interactive_binning_plot(strongest_feature_tree, "Decision Tree")
                st.plotly_chart(fig2, use_container_width=True)

            with col3:
                st.markdown(f"**Metode 3️⃣: Random Forest**  \nR² = {r2_rf:.3f}")
                fig3 = make_interactive_binning_plot(strongest_feature_rf, "Random Forest")
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown("---")
            st.caption("📘 Garis merah menunjukkan titik split hasil decision tree terhadap fitur terkuat setiap metode.")






