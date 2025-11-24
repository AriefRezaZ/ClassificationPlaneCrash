import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ----------------- LOGIN SYSTEM -----------------
USER = "admin"
PASS = "aman_banget"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USER and password == PASS:
            st.session_state.logged_in = True
            st.success("Login successful! 🎉")
            st.rerun()
        else:
            st.error("Invalid username or password ❌")
# ------------------------------------------------

else:    
    # Add a logout button in the sidebar
    st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.update({"logged_in": False}) or st.rerun())

    @st.cache_data
    def load_data():
        df = pd.read_csv("Plane Crashes.csv")
        return df

    st.title("✈️ Plane Crash Data ML Classification (with Country Filter)")

    # Load CSV
    data = load_data()    

    # Filter sidebar / search for country
    st.sidebar.header("Filter Options")
    all_countries = data["Country"].dropna().unique()
    all_countries = sorted(all_countries)
    selected_country = st.sidebar.selectbox(
        "Select Country to filter (All = no filter)", ["All"] + all_countries
    )

    if selected_country != "All":
        data_filtered = data[data["Country"] == selected_country].copy()
    else:
        data_filtered = data.copy()


    st.subheader("Raw Data Preview")
    st.dataframe(data_filtered, use_container_width=True)

    # --- Feature Engineering ---
    data_feat = data_filtered.dropna(
        subset=["Total fatalities", "Crew on board", "Pax on board"]
    )

    aboard = data_feat["Crew on board"].fillna(0) + data_feat["Pax on board"].fillna(0)
    data_feat["HighFatality"] = (
        (data_feat["Total fatalities"] / aboard > 0.5).astype(int)
    )

    valid_flight_types = data_feat["Flight type"].dropna().unique()
    valid_crash_causes = data_feat["Crash cause"].dropna().unique()

    # Features
    numeric_features = ["Crew on board", "Pax on board"]
    categorical_features = ["Flight type", "Crash cause"]

    X = data_feat[numeric_features + categorical_features]
    y = data_feat["HighFatality"]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # st.write(f"✅ Model Accuracy: {acc:.2f}")

    # --- Try Prediction ---
    st.subheader("🔮 Try Classification")

    crew_input = st.number_input("Crew on board:", min_value=0, step=1)
    pax_input = st.number_input("Passengers on board:", min_value=0, step=1)

    flight_type_input = st.selectbox(
        "Flight type:", options=data_feat["Flight type"].dropna().unique()
    )
    cause_input = st.selectbox(
        "Crash cause:", options=data_feat["Crash cause"].dropna().unique()
    )

    if st.button("Calculate Fatality Risk"):
        if (flight_type_input not in valid_flight_types) or (cause_input not in valid_crash_causes):
            st.warning("⚠️ Data tidak ditemukan untuk kombinasi input tersebut.")
        else:
            user_df = pd.DataFrame(
                [[crew_input, pax_input, flight_type_input, cause_input]],
                columns=numeric_features + categorical_features,
            )
            user_pred = clf.predict(user_df)[0]
            if user_pred == 1:
                st.error("High Fatality Risk 🚨")
            else:
                st.success("Low Fatality Risk ✅")

    # --- Visualization: Top 10 Countries with Most Crashes ---
    st.subheader("🌍 Top 10 Countries with Most Crashes")

    top_countries = data["Country"].value_counts().head(10)

    fig2, ax2 = plt.subplots()
    top_countries.plot(kind="bar", color="skyblue", ax=ax2)
    ax2.set_title("Top 10 Countries with Most Plane Crashes")
    ax2.set_xlabel("Country")
    ax2.set_ylabel("Number of Crashes")

    st.pyplot(fig2)

    # --- Visualization: Distribution of Fatalities ---
    fig, ax = plt.subplots()
    data_feat["HighFatalityLabel"] = data_feat["HighFatality"].replace({0:"Low Fatality", 1:"High Fatality"})

    data_feat["HighFatalityLabel"].value_counts().plot.bar(
        color=["green", "red"], ax=ax
    )

    ax.set_title("Crash Fatality Distribution")
    ax.set_xlabel("Fatality Category")
    ax.set_ylabel("Number of Crashes")

    st.pyplot(fig)
