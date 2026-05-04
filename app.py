#IMPORT#######################################################################################
import json
import os
import pickle
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq

    LANGCHAIN_AVAILABLE = True
    LANGCHAIN_IMPORT_ERROR = ""
except Exception as exc:
    AgentExecutor = None
    create_tool_calling_agent = None
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    ChatGroq = None
    tool = None
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_IMPORT_ERROR = str(exc)

LANGCHAIN_RUNTIME_ERROR = ""

#PATH TO PARAMETERS FILE##############################################################################
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Churn_Modelling.csv"
MODEL_PATH = BASE_DIR / "model.h5"
UPDATE_CSV_PATH = BASE_DIR / "logs" / "update_history.csv"
PARAMETERS_CSV_PATH = BASE_DIR / "logs" / "parameters.csv"
UPDATE_HISTORY_COLUMNS = [
    "timestamp",
    "customer_id",
    "pre_rec",
    "cur_rec",
    "pre_churn",
    "current_churn",
    "offer_given",
    "offer_string",
    "accepted_offer",
    "feedback_reason",
    "offer_attempt",
]

#FEATURES WHICH ARE EXPECTED TO CHANGE BY GIVING OFFER#####################################################################################################
ACTIONABLE_FEATURES = [
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
]
LEARNING_FEATURES = ACTIONABLE_FEATURES

#SCOPE OF OFFERS#################################################################################################################
OFFER_LIBRARY = {
    "cashback_bonus": "cashback bonus on card and debit spends",
    "fee_waiver": "temporary waiver on service and transfer fees",
    "rate_bonus": "higher savings return for a limited retention window",
    "credit_limit_review": "priority credit review with card upgrade support",
    "product_bundle": "bundled second product with loyalty benefits",
    "advisor_callback": "dedicated advisor callback with tailored account guidance"
}
OFFER_FEATURES = list(OFFER_LIBRARY.keys())

#HEURISTICS FOR EACH OFFER(BASED ON PRIOR KNOWLEDGE FROM Churn_Modeling.csv)###############################################################################################################
OFFER_EFFECTS = {
    "cashback_bonus": {"Balance": 0.45, "NumOfProducts": 0.15, "HasCrCard": 0.10, "IsActiveMember": 0.30},
    "fee_waiver": {"Balance": 0.35, "NumOfProducts": 0.15, "HasCrCard": 0.20, "IsActiveMember": 0.30},
    "rate_bonus": {"Balance": 0.50, "NumOfProducts": 0.10, "HasCrCard": 0.10, "IsActiveMember": 0.30},
    "credit_limit_review": {"Balance": 0.10, "NumOfProducts": 0.10, "HasCrCard": 0.55, "IsActiveMember": 0.25},
    "product_bundle": {"Balance": 0.10, "NumOfProducts": 0.50, "HasCrCard": 0.15, "IsActiveMember": 0.25},
    "advisor_callback": {"Balance": 0.20, "NumOfProducts": 0.15, "HasCrCard": 0.15, "IsActiveMember": 0.50},
}# these are the expected influence of each offer to the features these will be updated affter getting feedback from environment

RNG = np.random.default_rng()
SAFE_CHURN_MARGIN = 0.01

#BATCH SIZE FOR UPDATING PARAMETERS####################################################################################
BATCH_SIZE = 5

st.set_page_config(page_title="Bank Customer Retention Agent", layout="wide")

#USED TO ACCESS ENV VARIABLES(GROQ API KEY)##################################################################################################################
def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
load_env_file(BASE_DIR / ".env")

#LOADING ENCODER AND SCALING OBJECTS FOR FEATURE ENGINEERING##############################################################################################################
@st.cache_resource
def load_assets() -> tuple[Any, Any, Any, Any]:
    model = tf.keras.models.load_model(MODEL_PATH)
    with (BASE_DIR / "label_encoder_gender.pkl").open("rb") as file:
        label_encoder_gender = pickle.load(file)
    with (BASE_DIR / "onehot_encoder_geo.pkl").open("rb") as file:
        onehot_encoder_geo = pickle.load(file)
    with (BASE_DIR / "Scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

@st.cache_data
def load_customer_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["CustomerId"] = df["CustomerId"].astype(str)
    return df

model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()
customer_df = load_customer_dataset()


#PICKING LEARNING FEATURES FROM ALL FEATURES ###################################################################
@st.cache_data
def build_learning_feature_dataset() -> tuple[pd.DataFrame, pd.Series]:
    base_features = customer_df[LEARNING_FEATURES].astype(float).copy()
    target = customer_df["Exited"].astype(int).copy()
    return base_features, target

#FEATURE SCALING################
def scale_learning_features(feature_frame: pd.DataFrame) -> pd.DataFrame:
    scaler_features = list(getattr(scaler, "feature_names_in_", []))
    scaler_mean = pd.Series(getattr(scaler, "mean_", []), index=scaler_features, dtype=float)
    scaler_scale = pd.Series(getattr(scaler, "scale_", []), index=scaler_features, dtype=float)
    learning_mean = scaler_mean.reindex(LEARNING_FEATURES)
    learning_scale = scaler_scale.reindex(LEARNING_FEATURES).replace(0.0, 1.0)
    return (feature_frame - learning_mean) / learning_scale


def customer_to_learning_vector(customer_data: dict[str, Any]) -> pd.Series:
    return pd.Series(
        {feature: float(customer_data[feature]) for feature in LEARNING_FEATURES},
        dtype=float,
    )


def serialize_series(series: pd.Series) -> dict[str, float]:
    return {key: float(value) for key, value in series.items()}


#CHURN PREDICTOR###################
def predict_churn_(customer_data: dict[str, Any], learning_vector: pd.Series) -> float:
    updated_customer = dict(customer_data)
    for feature, value in learning_vector.items():
        updated_customer[feature] = float(value)
    return predict_churn(updated_customer)# 0 - retaiined, 1 - exited

#INITIALIZE PARAMETERS######################################################
def initialize_learning_state() -> dict[str, Any]:
    feature_frame, target = build_learning_feature_dataset()
    scaled_feature_frame = scale_learning_features(feature_frame)
    retained_mean = feature_frame.loc[target == 0].mean()
    retained_mean_scaled = scaled_feature_frame.loc[target == 0].mean()
    churned_mean_scaled = scaled_feature_frame.loc[target == 1].mean()
    feature_weights = (retained_mean_scaled - churned_mean_scaled).fillna(0.0) 
    # gives expected difference in features for ustomers who didn.t exited to tose who exited
    # based on this only feature weights(importance) is caculated
    offer_weights = {
        feature: {
            offer_name: float(OFFER_EFFECTS.get(offer_name, {}).get(feature, 0.0))
            for offer_name in OFFER_LIBRARY
        }
        for feature in LEARNING_FEATURES
    }
    return {
        "retained_mean": serialize_series(retained_mean),# used to give target value to achieve, for any current record with low retaining probability
        "feature_weights": serialize_series(feature_weights),
        "offer_weights": offer_weights,
    }

#STORING UPDATED PARAMETERS########
def append_parameters_snapshot(
    state: dict[str, Any],
    reason: str,
    customer_id: str = "",
) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason,
        "customer_id": customer_id,
        "feature_weights": json.dumps(state["feature_weights"]),
        "offer_weights": json.dumps(state["offer_weights"]),
        "retained_mean": json.dumps(state["retained_mean"]),
    }
    PARAMETERS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot_frame = pd.DataFrame([payload])
    if PARAMETERS_CSV_PATH.exists():
        snapshot_frame.to_csv(PARAMETERS_CSV_PATH, mode="a", header=False, index=False)
    else:
        snapshot_frame.to_csv(PARAMETERS_CSV_PATH, index=False)


def load_learning_state() -> dict[str, Any]:
    if not PARAMETERS_CSV_PATH.exists():# When no parameters initialized in starting
        state = initialize_learning_state()
        append_parameters_snapshot(state, reason="initialize")
        return state

    state_frame = pd.read_csv(PARAMETERS_CSV_PATH)
    if state_frame.empty:# When no parameters in parameters.csv
        state = initialize_learning_state()# initializing parameters
        append_parameters_snapshot(state, reason="initialize")
        return state

    # taking latest parameters for giving offers################################################
    latest = state_frame.iloc[-1]
    try:
        state = {
            "feature_weights": json.loads(latest["feature_weights"]),
            "offer_weights": json.loads(latest["offer_weights"]),
            "retained_mean": json.loads(latest["retained_mean"]),
        }
    #If incase no features in .csv
    except Exception:
        state = initialize_learning_state()
        append_parameters_snapshot(state, reason="reinitialize")
        return state

    if set(state.get("feature_weights", {})) != set(LEARNING_FEATURES):
        state = initialize_learning_state()
        append_parameters_snapshot(state, reason="reinitialize")
        return state

    if set(state.get("offer_weights", {})) != set(LEARNING_FEATURES):
        state = initialize_learning_state()
        append_parameters_snapshot(state, reason="reinitialize")
        return state

    return state

#STORING SESSION TILL NO UPDATE FROM ENVIRONMENT FOR THE ACTION(OFFER GIVEN)###########################
def get_observation_store() -> dict[str, dict[str, Any]]:
    return st.session_state.setdefault("observations", {})

def store_in_observation(customer_id: str, observation_payload: dict[str, Any]) -> None:
    get_observation_store()[customer_id] = observation_payload

def remove_from_observation(customer_id: str) -> dict[str, Any] | None:
    return get_observation_store().pop(customer_id, None)


def peek_observation(customer_id: str) -> dict[str, Any] | None:
    return get_observation_store().get(customer_id)


#USED TO SCALING PARAMETERS TO PROBABILITY FOR SELECTION
def softmax_dict(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    labels = list(score_map)
    scores = np.array([score_map[label] for label in labels], dtype=float)
    scores = scores - float(scores.max())
    weights = np.exp(scores)
    probabilities = weights / weights.sum()
    return {label: float(probability) for label, probability in zip(labels, probabilities)}

#SELECTING MOST INFLUENCING FEATURES AND OFFERS(EPSILION GREEDY APPROACH)
def safe_selection(
    items: list[str],
    scores: dict[str, float],
    k: int,
    rng: np.random.Generator,
    epsilon: float = 0.1   # exploration rate
) -> tuple[list[str], dict[str, float]]:

    if not items:
        return [], {}

    k = int(min(k, len(items)))
    if k <= 0:
        return [], {}

    # ---------- EXPLORATION ----------
    if rng.random() < epsilon:
        chosen = rng.choice(items, size=k, replace=False)
        prob = 1.0 / len(items)
        return list(chosen), {it: prob for it in chosen}

    # ---------- EXPLOITATION (your existing logic) ----------
    raw = np.array([float(scores.get(it, 0.0)) for it in items], dtype=float)
    raw = raw - raw.max()
    probs = np.exp(raw)

    probs = probs + 1e-8
    probs = probs / probs.sum()

    filtered_items = np.array(items)
    filtered_probs = probs

    if len(filtered_items) == 0:
        filtered_items = np.array(items)
        filtered_probs = np.ones(len(items)) / len(items)
    else:
        filtered_probs = filtered_probs / filtered_probs.sum()

    k = min(k, len(filtered_items))

    chosen = rng.choice(
        filtered_items,
        size=k,
        replace=False,
        p=filtered_probs
    )

    # faster mapping (better than np.where)
    prob_lookup = dict(zip(filtered_items, filtered_probs))

    prob_map = {
        it: float(prob_lookup[it])
        for it in chosen
    }

    return list(chosen), prob_map

#THIS COMPUTES REWARD BASED ON "OFFER ACCEPTED OR NOT" AND "DID OFFER INCREASED CUSTOMER CHURN PROBABILITY"#################
def compute_reward(update: dict[str, Any]) -> float:
    alpha = 0.8 # ACCEPTED INFLUENCE
    beta = 0.2 # CHURN INFLUENCE

    accepted = int(update.get("accepted_offer", 0))
    pre_churn = float(update["pre_churn"])
    current_churn = float(update["current_churn"])

    acceptance_signal = 1.0 if accepted == 1 else -1.0
    churn_signal = 1.0 if current_churn <= pre_churn else -1.0

    reward = alpha * acceptance_signal + beta * churn_signal

    # ---------- structured feedback influence ----------
    reason = update.get("feedback_reason", "")

    if reason == "Liked benefits":
        reward += 0.2
    elif reason == "Too expensive":
        reward -= 0.2
    elif reason == "Not relevant":
        reward -= 0.2
    elif reason == "Prefers other offer":
        reward -= 0.3

    return float(np.clip(reward, -1.0, 1.0))

#SCALING CUSTOMER RECORD(FOR PREDICTING CHURN)##################################
def build_feature_frame(customer_data: dict[str, Any]) -> pd.DataFrame:
    encoded_gender = label_encoder_gender.transform([customer_data["Gender"]])[0]
    base_frame = pd.DataFrame(
        {
            "CreditScore": [customer_data["CreditScore"]],
            "Gender": [encoded_gender],
            "Age": [customer_data["Age"]],
            "Tenure": [customer_data["Tenure"]],
            "Balance": [customer_data["Balance"]],
            "NumOfProducts": [customer_data["NumOfProducts"]],
            "HasCrCard": [customer_data["HasCrCard"]],
            "IsActiveMember": [customer_data["IsActiveMember"]],
            "EstimatedSalary": [customer_data["EstimatedSalary"]],
        }
    )
    geography_encoded = onehot_encoder_geo.transform([[customer_data["Geography"]]]).toarray()
    geography_frame = pd.DataFrame(
        geography_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
    )
    final_frame = pd.concat([base_frame.reset_index(drop=True), geography_frame], axis=1)
    return final_frame


def predict_churn(customer_data: dict[str, Any]) -> float:
    feature_frame = build_feature_frame(customer_data)
    scaled_data = scaler.transform(feature_frame)
    prediction = model.predict(scaled_data, verbose=0)
    return float(prediction[0][0])


def build_model_input_frame(records: pd.DataFrame) -> pd.DataFrame:
    encoded_gender = label_encoder_gender.transform(records["Gender"])
    base_frame = pd.DataFrame(
        {
            "CreditScore": records["CreditScore"].astype(float).to_numpy(),
            "Gender": encoded_gender,
            "Age": records["Age"].astype(float).to_numpy(),
            "Tenure": records["Tenure"].astype(float).to_numpy(),
            "Balance": records["Balance"].astype(float).to_numpy(),
            "NumOfProducts": records["NumOfProducts"].astype(float).to_numpy(),
            "HasCrCard": records["HasCrCard"].astype(float).to_numpy(),
            "IsActiveMember": records["IsActiveMember"].astype(float).to_numpy(),
            "EstimatedSalary": records["EstimatedSalary"].astype(float).to_numpy(),
        }
    )
    geography_encoded = onehot_encoder_geo.transform(records[["Geography"]]).toarray()
    geography_frame = pd.DataFrame(
        geography_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
    )
    return pd.concat([base_frame.reset_index(drop=True), geography_frame.reset_index(drop=True)], axis=1)


def predict_churn_batch(records: pd.DataFrame) -> np.ndarray:
    feature_frame = build_model_input_frame(records)
    scaled_data = scaler.transform(feature_frame)
    prediction = model.predict(scaled_data, verbose=0).reshape(-1)
    return prediction

#SEARCH THROUGH UPDATED RECORDS################################
def _search_update_df(df: pd.DataFrame, customer_id: str) -> dict[str, Any] | None:
    df_rev = df.iloc[::-1]
    record = df_rev[df_rev["customer_id"].astype(str) == str(customer_id)]
    if record.empty:
        return None
    row = record.iloc[0]
    cur_rec = json.loads(row["cur_rec"])
    return {str(k): v for k, v in cur_rec.items()}
#SEARCH THROUGH CHURN MODELING.CSV######################
def _search_customer_df(df: pd.DataFrame, customer_id: str) -> dict[str, Any] | None:
    df_rev = df.iloc[::-1]
    record = df_rev[df_rev["CustomerId"].astype(str) == str(customer_id)]
    if record.empty:
        return None
    row = record.iloc[0].to_dict()
    row.pop("RowNumber", None)
    return {str(k): v for k, v in row.items()}

#FETCH RECORD FOR CUSTOMER ID(first from updt record then churn model.csv)
def fetch_customer_record(customer_id: str) -> dict[str, Any] | None:
    base_record = _search_customer_df(customer_df, customer_id)
    if base_record is None:
        return None
    if UPDATE_CSV_PATH.exists():
        update_df = pd.read_csv(UPDATE_CSV_PATH)
        updated_fields = _search_update_df(update_df, customer_id)
        if updated_fields is not None:
            base_record.update(updated_fields)
    return base_record


#INSIGHTS FOR A CUSTOMER(Hardcoded based on prior knowledge just for intution)##################################
def build_insights(customer_data: dict[str, Any], probability: float) -> list[str]:
    insights: list[str] = []
    if float(customer_data["Balance"]) < 50000:
        insights.append("low retained balance compared with typical banking relationship value")
    if int(customer_data["IsActiveMember"]) == 0:
        insights.append("customer is inactive, which is a strong churn signal")
    if int(customer_data["NumOfProducts"]) <= 1:
        insights.append("limited product holding suggests weaker bank stickiness")
    if int(customer_data["Age"]) >= 55:
        insights.append("senior segment may need tailored servicing and wealth support")
    if int(customer_data["Tenure"]) <= 2:
        insights.append("short tenure indicates an early-stage relationship with lower loyalty")
    if probability >= 0.8:
        insights.append("model predicts very high churn risk requiring immediate intervention")
    elif probability >= 0.6:
        insights.append("model predicts moderate churn risk and targeted outreach is justified")
    else:
        insights.append("current profile looks comparatively stable")
    return insights[:4]


def get_learning_state_signature() -> str:
    state = load_learning_state()
    return json.dumps(state, sort_keys=True)


@st.cache_data(show_spinner=False)
def get_safe_high_churn_customer_candidates(state_signature: str) -> pd.DataFrame:
    del state_signature
    state = load_learning_state()
    retained_mean = pd.Series(state["retained_mean"], dtype=float)

    scored_df = customer_df.copy()
    scored_df["predicted_churn"] = predict_churn_batch(scored_df)
    high_df = scored_df[scored_df["predicted_churn"] >= 0.8].copy().reset_index(drop=True)
    if high_df.empty:
        return pd.DataFrame(columns=["CustomerId", "predicted_churn", "best_simulated_churn", "best_offer_key"])

    candidate_rows: list[pd.Series] = []
    candidate_meta: list[tuple[str, float, str]] = []

    for _, row in high_df.iterrows():
        current_rec = customer_to_learning_vector(row.to_dict())
        available_offer_names = [
            offer_name
            for offer_name in OFFER_LIBRARY
            if not (float(row["CreditScore"]) >= 650 and offer_name == "advisor_callback")
        ]
        candidate_sets: list[tuple[str, ...]] = [(offer_name,) for offer_name in available_offer_names]
        candidate_sets.extend(combinations(available_offer_names, 2))

        for offer_set in candidate_sets:
            temp_offer = {
                offer_name: float(1.0 / (rank + 1))
                for rank, offer_name in enumerate(offer_set)
            }
            simulated_rec = simulate_post_offer_record(
                current_rec,
                retained_mean,
                temp_offer,
                True,
                state,
            )
            candidate_row = row.copy()
            candidate_row["Balance"] = float(simulated_rec["Balance"])
            candidate_row["NumOfProducts"] = int(round(float(simulated_rec["NumOfProducts"])))
            candidate_row["HasCrCard"] = int(round(float(simulated_rec["HasCrCard"])))
            candidate_row["IsActiveMember"] = int(round(float(simulated_rec["IsActiveMember"])))
            candidate_rows.append(candidate_row)
            candidate_meta.append((str(row["CustomerId"]), float(row["predicted_churn"]), "+".join(offer_set)))

    candidate_df = pd.DataFrame(candidate_rows)
    simulated_predictions = predict_churn_batch(candidate_df)

    best_by_customer: dict[str, tuple[float, float, str]] = {}
    for (customer_id, baseline_churn, offer_key), simulated_churn in zip(candidate_meta, simulated_predictions):
        if float(simulated_churn) >= float(baseline_churn - SAFE_CHURN_MARGIN):
            continue
        previous = best_by_customer.get(customer_id)
        if previous is None or float(simulated_churn) < previous[1]:
            best_by_customer[customer_id] = (baseline_churn, float(simulated_churn), offer_key)

    safe_rows = [
        {
            "CustomerId": customer_id,
            "predicted_churn": baseline_churn,
            "best_simulated_churn": best_simulated_churn,
            "best_offer_key": offer_key,
        }
        for customer_id, (baseline_churn, best_simulated_churn, offer_key) in best_by_customer.items()
    ]
    result = pd.DataFrame(safe_rows)
    if result.empty:
        return result
    return result.sort_values(["predicted_churn", "CustomerId"]).reset_index(drop=True)

#USED WHEN API DOESN'T WORK IN BACKEND #
def create_offer_text(
    customer_data: dict[str, Any],
    churn_rate: float,
    offer_importance: dict[str, float],
    previous_one_rejected: int = 0,
) -> str:
    credit_score = float(customer_data["CreditScore"])
    top_offers = sorted(offer_importance.items(), key=lambda item: item[1], reverse=True)
    selected_labels = [OFFER_LIBRARY[offer_name] for offer_name, _ in top_offers]
    if credit_score >= 750:
        benefit_band = "premium"
    elif credit_score >= 650:
        benefit_band = "balanced"
    else:
        benefit_band = "supportive"
    urgency = "immediate" if churn_rate >= 0.8 else "targeted" if churn_rate >= 0.6 else "preventive"
    prefix = "Alternative offer" if previous_one_rejected else "Primary offer"
    return (
        f"{prefix}: propose a {urgency} {benefit_band} retention package centered on "
        f"{', '.join(selected_labels)} for customer {customer_data['CustomerId']}."
        f" Credit score {int(credit_score)} and churn risk {churn_rate:.2f}."
    )

#USED TO EXTRACT OFFER FROM RESPONSE OF LLM###################
def extract_llm_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text", "")
            else:
                text_value = str(item)
            text_value = str(text_value).strip()
            if text_value:
                parts.append(text_value)
        return " ".join(parts).strip()
    return str(content).strip()

#ACTIVATING GROQ API######################################
@st.cache_resource
def get_groq_llm() -> Any:
    if not LANGCHAIN_AVAILABLE or not os.getenv("GROQ_API_KEY"):
        return None
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

#THIS GENERATES OFFER####################################
def generate_offer_text(
    customer_data: dict[str, Any],
    churn_rate: float,
    offer_importance: dict[str, float],
    offer_scores: dict[str, float],
    selected_features: list[str],
    previous_one_rejected: int = 0,
) -> str:
    global LANGCHAIN_RUNTIME_ERROR
    
    #USED WHEN GROQ API KEY NOT AVAILABLE
    fallback_text = create_offer_text(
        customer_data=customer_data,
        churn_rate=churn_rate,
        offer_importance=offer_importance,
        previous_one_rejected=previous_one_rejected,
    )
    llm = get_groq_llm()
    if llm is None:
        return fallback_text

    #FORMATTING PARAMETER VALUES TO MAKE PROMPT
    ranked_offers = sorted(offer_importance.items(), key=lambda item: item[1], reverse=True)
    offer_lines = [
        (
            f"- {offer_name}: label={OFFER_LIBRARY[offer_name]}, "
            f"probability={probability:.4f}, raw_score={float(offer_scores.get(offer_name, 0.0)):.4f}"
        )
        for offer_name, probability in ranked_offers
    ]

    #THIS PROMPT IS PASSED TO LLM
    prompt = "\n".join(
        [
            "You are a bank retention strategist.",
            "Write a concise retention offer in at most 3 sentences.",
            "Use the offer importance and customer data to produce realistic offer",
            "You can include schemes,coupon,perks,relaxation but should avoid risky offers based on customer data",
            "Do not mention probabilities, raw scores, models, JSON, or Groq.",
            "Do not invent offer types outside the candidate list.",
            f"Customer ID: {customer_data['CustomerId']}",
            f"Churn risk: {churn_rate:.4f}",
            f"Credit score: {float(customer_data['CreditScore']):.0f}",
            f"Selected features to improve: {', '.join(selected_features)}",
            "Candidate offers:",
            *offer_lines,
            "Return only the final recommendation text.",
        ]
    )

    #RESPONSE WILL BE BASED ON PARAMETER VALUES AND HISTORY OF OFFERS PROVIDED BY LLM
    #NOTE: LLM is just forming offer sentence
    # offer scope is predefined, and influnce of each offer is provided by learning agent impolemented by us
    try:
        response = llm.invoke(prompt)
        offer_text = extract_llm_text(response)
        if offer_text:
            LANGCHAIN_RUNTIME_ERROR = ""
            return offer_text
        LANGCHAIN_RUNTIME_ERROR = "Groq returned an empty offer response."
    except Exception as exc:
        LANGCHAIN_RUNTIME_ERROR = str(exc)
    return fallback_text

#for adding update in customer record(due to offer) to update.csv###########################################################
def add_to_updated_csv(
    pre_rec: pd.Series,
    cur_rec: pd.Series,
    pre_churn: float,
    current_churn: float,
    offer_given: dict[str, float],
    offer_string: str,
    customer_id: str,
    accepted_offer: bool,
    feedback_reason: str = "",
    offer_attempt: int = 1,
) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "customer_id": customer_id,
        "pre_rec": json.dumps(serialize_series(pre_rec)),
        "cur_rec": json.dumps(serialize_series(cur_rec)),
        "pre_churn": float(pre_churn),
        "current_churn": float(current_churn),
        "offer_given": json.dumps({k: float(v) for k, v in offer_given.items()}),
        "offer_string": offer_string,
        "accepted_offer": int(accepted_offer),
        "feedback_reason": feedback_reason,
        "offer_attempt": int(offer_attempt),
    }
    update_frame = pd.DataFrame([payload])
    UPDATE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if UPDATE_CSV_PATH.exists():
        existing_frame = pd.read_csv(UPDATE_CSV_PATH)
        for column in UPDATE_HISTORY_COLUMNS:
            if column not in existing_frame.columns:
                existing_frame[column] = ""
        extra_columns = [column for column in existing_frame.columns if column not in UPDATE_HISTORY_COLUMNS]
        existing_frame = existing_frame[UPDATE_HISTORY_COLUMNS + extra_columns]
        existing_frame.to_csv(UPDATE_CSV_PATH, index=False)
        update_frame = update_frame.reindex(columns=existing_frame.columns, fill_value="")
        update_frame.to_csv(UPDATE_CSV_PATH, mode="a", header=False, index=False)
    else:
        update_frame = update_frame.reindex(columns=UPDATE_HISTORY_COLUMNS, fill_value="")
        update_frame.to_csv(UPDATE_CSV_PATH, index=False)

#SIMULATE CUSTOMER BANK DETAIL CHANGES DUE TO OFFER#############################################
def simulate_post_offer_record(
    pre_rec: pd.Series,
    retained_mean: pd.Series,
    offer_given: dict[str, float],
    accepted_offer: bool,
    state: dict[str, Any],
) -> pd.Series:
    if not accepted_offer:
        return pre_rec.copy()
    current = pre_rec.copy()
    for offer_name, importance in offer_given.items():
        for feature in ACTIONABLE_FEATURES:
            learned_effect = float(state["offer_weights"][feature].get(offer_name, 0.0))
            gap = float(retained_mean[feature] - current[feature])
            delta = gap * float(importance) * learned_effect
            if feature == "Balance":
                current[feature] = max(current[feature] + delta, 0.0)
            elif feature == "NumOfProducts":
                current[feature] = min(max(current[feature] + delta, 1.0), 4.0)
            elif feature in {"HasCrCard", "IsActiveMember"}:
                current[feature] = min(max(current[feature] + delta, 0.0), 1.0)
    return current

#UPDATE OFFER INFLUENCE BASED ON THE CUSTOMER RESPONSE TO OFFER########################################
def update_offer_weights(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    retained_mean = pd.Series(state["retained_mean"], dtype=float)
    pre_rec = pd.Series(update["pre_rec"], dtype=float)
    cur_rec = pd.Series(update["cur_rec"], dtype=float)
    offer_given = {
        offer_name: float(weight)
        for offer_name, weight in update["offer_given"].items()
        if float(weight) != 0.0
    }
    if not offer_given:
        return state

    d_rec = cur_rec - pre_rec
    reward = compute_reward(update)
    # How big is the change relative to how far we are from the ideal (retained_mean)?############
    # Just for scaling
    normalization_base = (retained_mean - pre_rec).abs().replace(0.0, 1.0)
    relative_change = (d_rec / normalization_base).clip(-1.0, 1.0)
    
    for offer_name, offer_strength in offer_given.items():
        for feature in ACTIONABLE_FEATURES:
            alpha = abs(float(relative_change.get(feature, 0.0)))

            if alpha == 0:
                continue

            current_weight = float(state["offer_weights"][feature][offer_name])
            learning_rate = 0.1

            updated_weight = current_weight + (
                learning_rate * reward * alpha * offer_strength
            )

            state["offer_weights"][feature][offer_name] = float(
                np.clip(updated_weight, -1.0, 1.0)
            )
    return state

def update_feature_weights(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    pre_rec = pd.Series(update["pre_rec"], dtype=float)
    cur_rec = pd.Series(update["cur_rec"], dtype=float)

    d_rec = cur_rec - pre_rec
    reward = compute_reward(update)
    direction = reward

    for feature in ACTIONABLE_FEATURES:
        change = abs(float(d_rec.get(feature, 0.0)))

        if change == 0:
            continue

        current_weight = float(state["feature_weights"][feature])

        updated_weight = current_weight + direction * change * 0.1

        state["feature_weights"][feature] = float(
            np.clip(updated_weight, -1.0, 1.0)
        )

    return state

#THIS PERSONALIZES OFFER FOR A CUSTOMER########################
def get_customer_offer_history(customer_id: str) -> list[dict[str, Any]]:
    if not UPDATE_CSV_PATH.exists():
        return []

    df = pd.read_csv(UPDATE_CSV_PATH)
    history = df[df["customer_id"].astype(str) == str(customer_id)]
    if history.empty:
        return []
    return history.tail(5).to_dict("records")

#SELECTING FEATURES BASED ON THEIR DIFFERENCE FROM TARGET(EXPECTED VALUES FOR RETAINED CUSTOMER) ##########################
def select_offer_features(
    required_gap: pd.Series,
    feature_weights: pd.Series,
    excluded_features: set[str] | None = None,
) -> list[str]:
    excluded = excluded_features or set()
    candidate_scores: dict[str, float] = {}
    for feature in ACTIONABLE_FEATURES:
        if feature in excluded:
            continue
        gap = float(required_gap.get(feature, 0.0))
        score = gap * float(feature_weights.get(feature, 0.0)) # larger the gap, focus more on that feature
        candidate_scores[feature] = score
    if not candidate_scores:
        fallback = (required_gap[ACTIONABLE_FEATURES] * feature_weights.reindex(ACTIONABLE_FEATURES).fillna(0.0)).sort_values(ascending=False)
        return [str(feature) for feature in fallback.index[:2]]
    sample_size = min(3, len(candidate_scores)) # 3 features are selected
    features = list(candidate_scores)
    # probabilities = softmax_dict(candidate_scores) # Converting focus into probability
    
    selected, _ = safe_selection(
    items=features,
    scores=candidate_scores,
    k=sample_size,
    rng=RNG)

    return [str(feature) for feature in selected]

#FUNCTION
# considers whether previous offer was rejected or not
# OFFERS ARE SELECTED IN THIS CODE ONLY
def give_offer(
    customer_data: dict[str, Any],
    churn_rate: float,
    state: dict[str, Any],
    excluded_offers: set[str] | None = None,
    previous_one_rejected: int = 0,
) -> dict[str, Any]:
    current_rec = customer_to_learning_vector(customer_data)
    retained_mean = pd.Series(state["retained_mean"], dtype=float)
    feature_weights = pd.Series(state["feature_weights"], dtype=float)
    required_gap = retained_mean - current_rec
    selected_features = select_offer_features(required_gap, feature_weights)
    if not selected_features:
        selected_features = ACTIONABLE_FEATURES[:2]

    selected_feature_scores = { # gap * feature weight
        feature: float(required_gap.get(feature, 0.0)) * float(feature_weights.get(feature, 0.0))
        for feature in selected_features
    }
    # Below is caculated how many previous offers were rejected and how many were not
    # These helps in telling which offers actually user finds useful and which are not relevant to user
    history = get_customer_offer_history(customer_data["CustomerId"])
    history_bonus = {offer_name: 0.0 for offer_name in OFFER_LIBRARY}
    history_penalty = {offer_name: 0.0 for offer_name in OFFER_LIBRARY}

    for record in history:
        past_offers = json.loads(record["offer_given"])
        for offer_name in past_offers:
            if int(record["accepted_offer"]) == 1:
                history_bonus[offer_name] += 0.2
            else:
                history_penalty[offer_name] += 0.3

    blocked_offers = excluded_offers or set()
    available_offer_names = [
        offer_name
        for offer_name in OFFER_LIBRARY
        if offer_name not in blocked_offers
        and not (float(customer_data["CreditScore"]) >= 650 and offer_name == "advisor_callback")
    ]
    if not available_offer_names:
        available_offer_names = [
            offer_name
            for offer_name in OFFER_LIBRARY
            if not (float(customer_data["CreditScore"]) >= 650 and offer_name == "advisor_callback")
        ]

    offer_scores = {offer_name: 0.0 for offer_name in OFFER_LIBRARY}
    offer_candidates: list[dict[str, Any]] = []
    candidate_sets: list[tuple[str, ...]] = [(offer_name,) for offer_name in available_offer_names]
    candidate_sets.extend(combinations(available_offer_names, 2))

    for offer_set in candidate_sets:
        temp_offer = {
            offer_name: float(1.0 / (rank + 1))
            for rank, offer_name in enumerate(offer_set)
        }
        simulated_rec = simulate_post_offer_record(
            current_rec,
            retained_mean,
            temp_offer,
            True,
            state,
        )
        simulated_churn = predict_churn_(customer_data, simulated_rec)
        churn_delta = float(simulated_churn - churn_rate)
        history_adjustment = sum(
            history_bonus.get(offer_name, 0.0) - history_penalty.get(offer_name, 0.0)
            for offer_name in offer_set
        )
        movement_penalty = 0.01 * float((simulated_rec - current_rec).abs().sum())
        raw_score = (-float(simulated_churn)) + history_adjustment - movement_penalty
        improves_churn = float(simulated_churn) < float(churn_rate - SAFE_CHURN_MARGIN)

        for offer_name in offer_set:
            offer_scores[offer_name] = max(float(offer_scores.get(offer_name, -1e9)), raw_score)

        offer_candidates.append(
            {
                "offer_names": list(offer_set),
                "candidate_label": " + ".join(OFFER_LIBRARY[offer_name] for offer_name in offer_set),
                "offer_weights": temp_offer,
                "simulated_churn": float(simulated_churn),
                "churn_delta": churn_delta,
                "score": float(raw_score),
                "history_adjustment": float(history_adjustment),
                "movement_penalty": float(movement_penalty),
                "improves_churn": bool(improves_churn),
            }
        )

    ranked_candidates = [
        candidate for candidate in offer_candidates if bool(candidate["improves_churn"])
    ]
    ranked_candidates = sorted(
        ranked_candidates,
        key=lambda candidate: (
            float(candidate["simulated_churn"]),
            -float(candidate["history_adjustment"]),
            len(candidate["offer_names"]),
            str(candidate["candidate_label"]),
        ),
    )

    all_ranked_candidates = sorted(
        offer_candidates,
        key=lambda candidate: (
            not bool(candidate["improves_churn"]),
            float(candidate["simulated_churn"]),
            -float(candidate["history_adjustment"]),
            len(candidate["offer_names"]),
            str(candidate["candidate_label"]),
        ),
    )
    ranking_score_map = {
        str(index): float(candidate["score"])
        for index, candidate in enumerate(all_ranked_candidates)
    }
    available_offer_probabilities = softmax_dict(ranking_score_map)
    offer_rankings = [
        {
            "offer_name": " + ".join(candidate["offer_names"]),
            "label": str(candidate["candidate_label"]),
            "score": float(candidate["score"]),
            "probability": float(available_offer_probabilities.get(str(index), 0.0)),
            "simulated_churn": float(candidate["simulated_churn"]),
            "churn_delta": float(candidate["churn_delta"]),
            "improves_churn": bool(candidate["improves_churn"]),
            "offer_names": list(candidate["offer_names"]),
        }
        for index, candidate in enumerate(all_ranked_candidates)
    ]

    if not ranked_candidates:
        return {
            "selected_features": selected_features,
            "selected_feature_scores": selected_feature_scores,
            "offers_importance": {},
            "offer_scores": {},
            "offer_rankings": offer_rankings,
            "offer_string": (
                "No safe automated offer is available for this customer right now. "
                "All evaluated offers increased predicted churn, so escalate to advisor review."
            ),
            "offer_labels": [],
            "current_rec": serialize_series(current_rec),
            "no_safe_offer": True,
        }

    chosen_candidate = ranked_candidates[0]
    offers_importance = {
        str(offer_name): float(weight)
        for offer_name, weight in chosen_candidate["offer_weights"].items()
    }
    chosen_offer_scores = {
        str(offer_name): float(offer_scores.get(offer_name, 0.0))
        for offer_name in chosen_candidate["offer_names"]
    }

    #This offer string will be used if LLM doesn't work properly to generate text
    offer_string = generate_offer_text(
        customer_data=customer_data,
        churn_rate=churn_rate,
        offer_importance=offers_importance,
        offer_scores=chosen_offer_scores,
        selected_features=selected_features,
        previous_one_rejected=previous_one_rejected,
    )
    return {
        "selected_features": selected_features,
        "selected_feature_scores": selected_feature_scores,
        "offers_importance": offers_importance,
        "offer_scores": chosen_offer_scores,
        "offer_rankings": offer_rankings,
        "offer_string": offer_string,
        "offer_labels": [OFFER_LIBRARY[offer_name] for offer_name in offers_importance],
        "current_rec": serialize_series(current_rec),
        "no_safe_offer": False,
    }


langchain_executor = None


def analyze_customer(customer_id: str, customer_query: str) -> dict[str, Any]:
    customer_data = fetch_customer_record(customer_id)
    if customer_data is None:
        raise ValueError(f"CustomerId {customer_id} was not found in the dataset.")

    probability = predict_churn(customer_data)
    insights = build_insights(customer_data, probability)
    # state contains parameters values
    state = load_learning_state()
    # offer_payload gets offers selected with their probabilities and scores
    offer_payload = give_offer(customer_data, probability, state)

    if not offer_payload.get("no_safe_offer", False):
        # Here offer is sent and observe change in customer behaviour(acceptence and feedback)
        # When accepted/Not accepted and feedback is submitted it removes from observation
        store_in_observation(
            customer_id,
            {
                "customer_id": customer_id,
                "pre_rec": offer_payload["current_rec"],
                "pre_churn": float(probability),
                "offer_given": offer_payload["offers_importance"],
                "offer_string": offer_payload["offer_string"],
                "selected_features": offer_payload["selected_features"],
                "offer_attempt": 1,
                "customer_query": customer_query,
                "created_at": datetime.utcnow().isoformat(),
            },
        )

    # this string is used when LLM is not working
    recommended_actions = [offer_payload["offer_string"]]
    if offer_payload.get("no_safe_offer", False):
        recommended_actions.append("Review the customer manually instead of sending an automated retention offer.")
    else:
        recommended_actions.extend(
            f"Prioritize {feature} improvement for this customer." for feature in offer_payload["selected_features"]
        )
    return {
        "customer_id": customer_id,
        "customer_query": customer_query,
        "churn_risk": round(probability, 4),
        "insights": insights,
        "recommended_actions": recommended_actions,
        "critic": {"score": None, "status": "pending"},
        "offer_context": {
            "selected_features": offer_payload["selected_features"],
            "selected_feature_scores": offer_payload["selected_feature_scores"],
            "offers_importance": offer_payload["offers_importance"],
            "offer_scores": offer_payload["offer_scores"],
            "offer_rankings": offer_payload["offer_rankings"],
            "offer_labels": offer_payload["offer_labels"],
        },
        "feedback_status": {
            "accepted_offer": None,
            "feedback_reason": "",
            "replacement_generated": False,
            "replacement_limit_reached": False,
            "interaction_complete": bool(offer_payload.get("no_safe_offer", False)),
            "offer_attempt": 1,
        },
    }

#When gets feedback then this function is called and customer is removed from observation and whatever customer behaviour change was observed is stored to update.csv
# Assuming offer is rejected then it stores customer behaviour in update.csv again stores it in observation 
def process_feedback(
    latest_result: dict[str, Any],
    accepted_offer: bool,
    feedback_reason: str
) -> dict[str, Any]:
    customer_id = latest_result["customer_id"]
    customer_data = fetch_customer_record(customer_id)
    if customer_data is None:
        raise ValueError(f"CustomerId {customer_id} was not found in the dataset.")

    state = load_learning_state()
    observation = remove_from_observation(customer_id)
    if observation is None:
        raise ValueError("No active observation found for this customer. Analyze the customer again before submitting feedback.")
    offer_attempt = int(observation.get("offer_attempt", 1))

    pre_rec = pd.Series(observation["pre_rec"], dtype=float)
    retained_mean = pd.Series(state["retained_mean"], dtype=float)
    pre_churn = float(observation["pre_churn"])
    offer_given = {key: float(value) for key, value in observation["offer_given"].items()}
    cur_rec = simulate_post_offer_record(pre_rec, retained_mean, offer_given, accepted_offer, state)
    current_churn = predict_churn_(customer_data, cur_rec)# gives churn for customer data after offer

    add_to_updated_csv(
        pre_rec=pre_rec,
        cur_rec=cur_rec,
        pre_churn=pre_churn,
        current_churn=current_churn,
        offer_given=offer_given,
        offer_string=observation["offer_string"],
        customer_id=customer_id,
        accepted_offer=accepted_offer,
        feedback_reason=feedback_reason,
        offer_attempt=offer_attempt,
    )

    batch_store = st.session_state.setdefault("batch_updates", [])
    batch_store.append(
        {
            "pre_rec": serialize_series(pre_rec),
            "cur_rec": serialize_series(cur_rec),
            "pre_churn": pre_churn,
            "current_churn": current_churn,
            "offer_given": offer_given,
            "accepted_offer": int(accepted_offer),
            "feedback_reason": feedback_reason
        }
    )

    if len(batch_store) >= BATCH_SIZE:
        for update in batch_store:
            state = update_offer_weights(state, update)
            state = update_feature_weights(state, update)

        append_parameters_snapshot(state, reason="batch_update", customer_id=customer_id)
        batch_store.clear()

    latest_result["critic"] = {"score": None,"status": "removed (using reward-based learning)"}
    latest_result["post_offer_churn"] = round(current_churn, 4)
    latest_result["feedback_status"] = {
        "accepted_offer": bool(accepted_offer),
        "feedback_reason": feedback_reason,
        "replacement_generated": False,
        "replacement_limit_reached": False,
        "interaction_complete": bool(accepted_offer),
        "offer_attempt": offer_attempt,
    }

    # new offer and don't consider previously given offers.
    if not accepted_offer and offer_attempt == 1:
        replacement_offer = give_offer(
            customer_data,
            current_churn,
            state,
            excluded_offers=set(offer_given),# here the very next offer is not containing same offer(this avoids repetative offer to customer) but it may include offers in next-to-next offers
            previous_one_rejected=1,
        )
        if replacement_offer.get("no_safe_offer", False):
            latest_result["recommended_actions"] = [replacement_offer["offer_string"]]
            latest_result["recommended_actions"].append(
                "Stop automated offers for this interaction and escalate to advisor review."
            )
            latest_result["offer_context"] = {
                "selected_features": replacement_offer["selected_features"],
                "selected_feature_scores": replacement_offer["selected_feature_scores"],
                "offers_importance": replacement_offer["offers_importance"],
                "offer_scores": replacement_offer["offer_scores"],
                "offer_rankings": replacement_offer["offer_rankings"],
                "offer_labels": replacement_offer["offer_labels"],
                "previous_one_rejected": 1,
                "replacement_limit_reached": True,
                "no_safe_offer": True,
            }
            latest_result["feedback_status"] = {
                "accepted_offer": False,
                "feedback_reason": feedback_reason,
                "replacement_generated": False,
                "replacement_limit_reached": True,
                "interaction_complete": True,
                "offer_attempt": offer_attempt,
                "rejected_offer": observation["offer_string"],
                "rejected_offer_keys": list(offer_given),
            }
        else:
            store_in_observation(
                customer_id,
                {
                    "customer_id": customer_id,
                    "pre_rec": replacement_offer["current_rec"],
                    "pre_churn": float(current_churn),
                    "offer_given": replacement_offer["offers_importance"],
                    "offer_string": replacement_offer["offer_string"],
                    "selected_features": replacement_offer["selected_features"],
                    "offer_attempt": 2,
                    "customer_query": latest_result.get("customer_query", ""),
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
            latest_result["recommended_actions"] = [replacement_offer["offer_string"]]
            latest_result["recommended_actions"].extend(
                f"Follow up through {label}." for label in replacement_offer["offer_labels"]
            )
            latest_result["offer_context"] = {
                "selected_features": replacement_offer["selected_features"],
                "selected_feature_scores": replacement_offer["selected_feature_scores"],
                "offers_importance": replacement_offer["offers_importance"],
                "offer_scores": replacement_offer["offer_scores"],
                "offer_rankings": replacement_offer["offer_rankings"],
                "offer_labels": replacement_offer["offer_labels"],
                "previous_one_rejected": 1,
                "no_safe_offer": False,
            }
            latest_result["feedback_status"] = {
                "accepted_offer": False,
                "feedback_reason": feedback_reason,
                "replacement_generated": True,
                "replacement_limit_reached": False,
                "interaction_complete": False,
                "offer_attempt": 2,
                "rejected_offer": observation["offer_string"],
                "rejected_offer_keys": list(offer_given),
            }
    elif not accepted_offer:
        latest_result["recommended_actions"] = [
            "Replacement offer was rejected. Stop automated offers for this interaction and escalate to advisor review."
        ]
        latest_result["recommended_actions"].append(
            "Review customer feedback before proposing another retention offer."
        )
        latest_result["offer_context"] = {
            **latest_result.get("offer_context", {}),
            "previous_one_rejected": 1,
            "replacement_limit_reached": True,
        }
        latest_result["feedback_status"] = {
            "accepted_offer": False,
            "feedback_reason": feedback_reason,
            "replacement_generated": False,
            "replacement_limit_reached": True,
            "interaction_complete": True,
            "offer_attempt": offer_attempt,
            "rejected_offer": observation["offer_string"],
            "rejected_offer_keys": list(offer_given),
        }

    return latest_result


def render_offer_strategy(offer_context: dict[str, Any]) -> None:
    selected_offer_weights = {
        offer_name: float(weight)
        for offer_name, weight in offer_context.get("offers_importance", {}).items()
    }
    selected_feature_scores = {
        feature: float(score)
        for feature, score in offer_context.get("selected_feature_scores", {}).items()
    }
    selected_features = offer_context.get("selected_features", list(selected_feature_scores))
    offer_scores = offer_context.get("offer_scores", {})
    offer_rankings = offer_context.get("offer_rankings", [])

    st.markdown("**Selected offers**")
    if selected_offer_weights:
        selected_offer_df = pd.DataFrame(
            [
                {
                    "Offer Key": offer_name,
                    "Offer": OFFER_LIBRARY.get(offer_name, offer_name),
                    "Selection Weight": round(weight, 4),
                    "Score": round(float(offer_scores.get(offer_name, 0.0)), 4),
                }
                for offer_name, weight in selected_offer_weights.items()
            ]
        )
        st.dataframe(selected_offer_df, width="stretch", hide_index=True)
    else:
        st.caption("No offer was selected for this strategy.")

    st.markdown("**Selected feature drivers**")
    if selected_feature_scores:
        feature_df = pd.DataFrame(
            [
                {
                    "Feature": feature,
                    "Importance": round(float(selected_feature_scores.get(feature, 0.0)), 4),
                }
                for feature in selected_features
            ]
        )
        st.dataframe(feature_df, width="stretch", hide_index=True)
        st.bar_chart(feature_df.set_index("Feature"))
    else:
        st.caption("No feature driver was selected for this strategy.")

    if offer_rankings:
        st.markdown("**Full offer ranking**")
        offer_df = pd.DataFrame(
            [
                {
                    "Selected": "Yes" if offer.get("offer_name") in selected_offer_weights else "",
                    "Offer Key": offer.get("offer_name", ""),
                    "Offer": offer.get("label", ""),
                    "Score": round(float(offer.get("score", 0.0)), 4),
                    "Probability": round(float(offer.get("probability", 0.0)), 4),
                }
                for offer in offer_rankings
            ]
        )
        st.dataframe(offer_df, width="stretch", hide_index=True)


def render_result_summary(result: dict[str, Any]) -> None:
    st.subheader("Churn Risk")
    churn_risk = float(result["churn_risk"])
    if "post_offer_churn" in result:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Before Offer", f"{churn_risk:.3f}")
        with col2:
            post_offer_churn = float(result["post_offer_churn"])
            st.metric(
                "After Offer",
                f"{post_offer_churn:.3f}",
                delta=round(post_offer_churn - churn_risk, 4),
            )
    else:
        st.metric("Churn Probability", f"{churn_risk:.3f}")
        st.progress(max(0.0, min(1.0, churn_risk)))

    st.subheader("Insights")
    for insight in result.get("insights", []):
        st.write("-", insight)

    actions = result.get("recommended_actions", [])
    if actions:
        feedback_status = result.get("feedback_status", {})
        offer_context = result.get("offer_context", {})
        replacement_limit_reached = bool(
            feedback_status.get("replacement_limit_reached")
            or offer_context.get("replacement_limit_reached")
        )
        replacement_generated = bool(feedback_status.get("replacement_generated"))
        is_replacement_offer = bool(offer_context.get("previous_one_rejected")) and not replacement_limit_reached
        if replacement_limit_reached:
            st.warning("Replacement offer was rejected. No more automated offers will be generated for this interaction.")
        if replacement_generated:
            st.info("Previous offer was rejected. A replacement offer has been generated.")
        if replacement_limit_reached:
            offer_heading = "Advisor Review"
        elif replacement_generated or is_replacement_offer:
            offer_heading = "Replacement Offer"
        else:
            offer_heading = "Recommended Offer"
        st.subheader(offer_heading)
        if replacement_limit_reached:
            st.warning(actions[0])
        else:
            st.success(actions[0])

    if len(actions) > 1:
        st.subheader("Focus Areas")
        for action in actions[1:]:
            st.write("-", action)

    offer_context = result.get("offer_context")
    if offer_context:
        st.subheader("Offer Strategy")
        render_offer_strategy(offer_context)


def render_parameter_visualizer() -> None:
    with st.expander("Parameter Visualizer", expanded=False):
        if not PARAMETERS_CSV_PATH.exists():
            st.caption("No parameter snapshots are available yet.")
            return

        params_df = pd.read_csv(PARAMETERS_CSV_PATH)
        if params_df.empty:
            st.caption("No parameter snapshots are available yet.")
            return

        params_df["feature_weights"] = params_df["feature_weights"].apply(json.loads)
        params_df["offer_weights"] = params_df["offer_weights"].apply(json.loads)
        params_df["batch_step"] = range(1, len(params_df) + 1)

        fw_df = pd.json_normalize(params_df["feature_weights"])
        fw_df["batch_step"] = params_df["batch_step"]
        st.subheader("Feature Weights Evolution")
        st.line_chart(fw_df.set_index("batch_step"))

        st.subheader("Current Offer Strategy")
        latest_offer = pd.DataFrame(params_df.iloc[-1]["offer_weights"])
        st.dataframe(latest_offer.style.format("{:.2f}"))

        if len(params_df) <= 1:
            return

        prev_offer = pd.DataFrame(params_df.iloc[-2]["offer_weights"])
        curr_offer = pd.DataFrame(params_df.iloc[-1]["offer_weights"])
        delta_offer = (curr_offer - prev_offer).round(3)

        st.subheader("Recent Learning Update (Delta weights)")
        st.dataframe(delta_offer)

        prev_feature = params_df.iloc[-2]["feature_weights"]
        curr_feature = params_df.iloc[-1]["feature_weights"]
        delta_feature = {
            k: round(float(curr_feature[k]) - float(prev_feature[k]), 4)
            for k in curr_feature
        }
        st.subheader("Feature Importance Change")
        st.json(delta_feature)

        latest_fw = params_df.iloc[-1]["feature_weights"]
        top_feature = max(latest_fw, key=lambda x: abs(latest_fw[x]))
        max_change = delta_offer.abs().stack().idxmax()

        st.subheader("Key Insights")
        st.write(f"- Most influential feature: **{top_feature}**")
        st.write(f"- Largest update: **{max_change[1]} -> {max_change[0]}**")


def render_learning_agent_ui() -> None:
    st.title("Bank Customer Retention Learning Agent")
    st.caption("Performance element + learning element with reward-based feedback on top of the churn model.")
    st.subheader("Learning Agent")

    with st.expander("Manual Predictor", expanded=False):
        st.write("Run the same churn and offer strategy pipeline with custom customer inputs.")

        with st.form("manual_predictor_form"):
            col1, col2 = st.columns(2)
            with col1:
                geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0], key="manual_geography")
                gender = st.selectbox("Gender", label_encoder_gender.classes_, key="manual_gender")
                age = st.slider("Age", 18, 92, key="manual_age")
                tenure = st.slider("Tenure", 0, 10, key="manual_tenure")
                balance = st.number_input("Balance", min_value=0.0, key="manual_balance")
            with col2:
                credit_score = st.number_input(
                    "Credit Score",
                    min_value=300.0,
                    max_value=900.0,
                    value=650.0,
                    key="manual_credit_score",
                )
                estimated_salary = st.number_input("Estimated Salary", min_value=0.0, key="manual_estimated_salary")
                num_of_products = st.slider("Number of Products", 1, 4, key="manual_num_of_products")
                has_cr_card = st.selectbox("Has Credit Card", [0, 1], key="manual_has_cr_card")
                is_active_member = st.selectbox("Is Active Member", [0, 1], key="manual_is_active_member")
            manual_submitted = st.form_submit_button("Predict Churn", width="stretch")

        if manual_submitted:
            try:
                manual_customer = {
                    "CustomerId": "manual_input",
                    "CreditScore": credit_score,
                    "Gender": gender,
                    "Age": age,
                    "Tenure": tenure,
                    "Balance": balance,
                    "NumOfProducts": num_of_products,
                    "HasCrCard": has_cr_card,
                    "IsActiveMember": is_active_member,
                    "EstimatedSalary": estimated_salary,
                    "Geography": geography,
                }
                probability = predict_churn(manual_customer)
                manual_offer_payload = give_offer(manual_customer, probability, load_learning_state())
                manual_actions = [manual_offer_payload["offer_string"]]
                manual_actions.extend(
                    f"Prioritize {feature} improvement for this customer."
                    for feature in manual_offer_payload["selected_features"]
                )
                st.session_state["latest_manual_result"] = {
                    "customer_id": "manual_input",
                    "churn_risk": round(probability, 4),
                    "insights": build_insights(manual_customer, probability),
                    "recommended_actions": manual_actions,
                    "offer_context": {
                        "selected_features": manual_offer_payload["selected_features"],
                        "selected_feature_scores": manual_offer_payload["selected_feature_scores"],
                        "offers_importance": manual_offer_payload["offers_importance"],
                        "offer_scores": manual_offer_payload["offer_scores"],
                        "offer_rankings": manual_offer_payload["offer_rankings"],
                        "offer_labels": manual_offer_payload["offer_labels"],
                    },
                }
            except Exception as exc:
                st.error(str(exc))

        manual_result = st.session_state.get("latest_manual_result")
        if manual_result:
            render_result_summary(manual_result)

    with st.expander("Analyze Existing Customer", expanded=True):
        st.write("Use a `CustomerId` from `Churn_Modelling.csv`, for example `15634602`.")

        with st.form("existing_customer_form"):
            state_signature = get_learning_state_signature()
            safe_high_churn_df = get_safe_high_churn_customer_candidates(state_signature)
            customer_pool = st.radio(
                "Customer Pool",
                ["All customers", "High churn with safe offer"],
                horizontal=True,
                help="Use the filtered pool to test customers where the current model finds at least one safe automated offer.",
            )
            if customer_pool == "High churn with safe offer" and not safe_high_churn_df.empty:
                customer_ids = safe_high_churn_df["CustomerId"].astype(str).tolist()
                st.caption(f"{len(customer_ids)} high-churn customers currently pass the safe-offer gate.")
            else:
                customer_ids = sorted(customer_df["CustomerId"].astype(str).unique())
                if customer_pool == "High churn with safe offer":
                    st.caption("No high-churn customers currently pass the safe-offer gate. Showing the full customer list instead.")
            customer_id = st.selectbox("Customer ID", customer_ids, index=0)
            customer_query = st.text_area(
                "Agent Query",
                value="Analyze the customer, estimate churn risk, and recommend retention actions.",
                height=100,
            )
            st.caption("Feature selection and offer scoring are computed locally; Groq is used to turn the scored offers into natural-language recommendations when available.")
            if LANGCHAIN_AVAILABLE and os.getenv("GROQ_API_KEY"):
                st.caption("Groq offer generation is enabled.")
                if LANGCHAIN_RUNTIME_ERROR:
                    st.info(f"Groq status: {LANGCHAIN_RUNTIME_ERROR}")
            elif LANGCHAIN_AVAILABLE:
                st.caption("LangChain packages are available, but `GROQ_API_KEY` is missing. Offer text falls back to the local template.")
            else:
                st.caption(f"LangChain packages are unavailable: {LANGCHAIN_IMPORT_ERROR}. Offer text falls back to the local template.")
            analyze_submitted = st.form_submit_button("Analyze Customer", width="stretch")

        if analyze_submitted:
            try:
                result = analyze_customer(customer_id.strip(), customer_query.strip())
                st.session_state["latest_agent_result"] = result
            except Exception as exc:
                st.error(str(exc))

        latest_result = st.session_state.get("latest_agent_result")
        if latest_result:
            result_slot = st.container()
            feedback_status = latest_result.get("feedback_status", {})
            active_observation = peek_observation(latest_result["customer_id"])
            if active_observation is not None:
                offer_attempt = int(active_observation.get("offer_attempt", feedback_status.get("offer_attempt", 1)))
                offer_stage = "Primary Offer" if offer_attempt == 1 else "Replacement Offer"
                widget_suffix = f"{latest_result['customer_id']}_{offer_attempt}"

                st.subheader(f"{offer_stage} Feedback")
                if offer_attempt == 2:
                    st.info("The first offer was rejected. Capture feedback for the replacement offer to complete this interaction.")

                col1, col2 = st.columns([2, 3])
                with col1:
                    response = st.radio(
                        "Customer Response",
                        ["Accepted", "Rejected"],
                        horizontal=True,
                        key=f"feedback_response_{widget_suffix}",
                    )
                    accepted_offer = response == "Accepted"
                with col2:
                    feedback_reason = st.selectbox(
                        "Feedback Reason",
                        [
                            "Liked benefits",
                            "Too expensive",
                            "Not relevant",
                            "Prefers other offer",
                            "No clear reason",
                        ],
                        key=f"feedback_reason_{widget_suffix}",
                    )

                if st.button("Update & Learn", width="stretch", key=f"update_learn_{widget_suffix}"):
                    try:
                        latest_result = process_feedback(latest_result, accepted_offer, feedback_reason)
                        st.session_state["latest_agent_result"] = latest_result
                        updated_status = latest_result.get("feedback_status", {})
                        if updated_status.get("replacement_generated"):
                            st.success("Primary offer feedback recorded. Replacement offer is ready for response.")
                        elif updated_status.get("interaction_complete"):
                            st.success("Feedback recorded and the interaction is complete.")
                        else:
                            st.success("Feedback recorded and learning updated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            else:
                if feedback_status.get("interaction_complete"):
                    st.info("This customer interaction is complete. No active offer is waiting for feedback.")
                else:
                    st.warning("No active offer is waiting for feedback. Analyze the customer again to start a new interaction.")

            with result_slot:
                render_result_summary(latest_result)

    st.markdown("Built with Streamlit, TensorFlow, and LangChain.")


render_learning_agent_ui()
st.stop()


st.title("Bank Customer Retention Learning Agent")
st.caption("Performance element + learning element with reward-based feedback on top of the churn model.")

manual_tab, agent_tab = st.tabs(["Manual Prediction", "Learning Agent"])

with manual_tab:
    st.subheader("Direct churn prediction")
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)

    age = st.slider("Age", 18, 92)
    tenure = st.slider("Tenure", 0, 10)
    balance = st.number_input("Balance", min_value=0.0)
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=650.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
    num_of_products = st.slider("Number of Products", 1, 4)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active_member = st.selectbox("Is Active Member", [0, 1])

    if st.button("Predict Churn"):
        manual_customer = {
            "CustomerId": "manual_input",
            "CreditScore": credit_score,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
            "Geography": geography,
        }
        probability = predict_churn(manual_customer)
        manual_offer_payload = give_offer(manual_customer, probability, load_learning_state())
        manual_actions = [manual_offer_payload["offer_string"]]
        manual_actions.extend(
            f"Prioritize {feature} improvement for this customer."
            for feature in manual_offer_payload["selected_features"]
        )
        st.subheader("Prediction Result")
        st.write(f"Churn Probability: {probability:.2f}")
        st.progress(float(probability))
        st.json(
            {
                "customer_id": "manual_input",
                "churn_risk": round(probability, 4),
                "insights": build_insights(manual_customer, probability),
                "recommended_actions": manual_actions,
                "offer_context": {
                    "selected_features": manual_offer_payload["selected_features"],
                    "selected_feature_scores": manual_offer_payload["selected_feature_scores"],
                    "offers_importance": manual_offer_payload["offers_importance"],
                    "offer_scores": manual_offer_payload["offer_scores"],
                    "offer_rankings": manual_offer_payload["offer_rankings"],
                    "offer_labels": manual_offer_payload["offer_labels"],
                },
            }
        )

with agent_tab: #-? This tab still needs UX review if you want to change the agent-side interaction further
    st.subheader("Analyze an existing customer")
    st.write("Use a `CustomerId` from `Churn_Modelling.csv`, for example `15634602`.")

    customer_ids = sorted(customer_df["CustomerId"].astype(str).unique())
    customer_id = st.selectbox("Customer ID", customer_ids, index=0)
with st.expander("Parameter Visualizer", expanded=False):
    if PARAMETERS_CSV_PATH.exists():
        params_df = pd.read_csv(PARAMETERS_CSV_PATH)

        if not params_df.empty:
            params_df["feature_weights"] = params_df["feature_weights"].apply(json.loads)
            params_df["offer_weights"] = params_df["offer_weights"].apply(json.loads)

            # ---------- Create batch index (X-axis fix) ----------
            params_df["batch_step"] = range(1, len(params_df) + 1)

            # ---------- Feature weights trend ----------
            fw_df = pd.json_normalize(params_df["feature_weights"])
            fw_df["batch_step"] = params_df["batch_step"]

            st.subheader("Feature Weights Evolution")
            st.line_chart(fw_df.set_index("batch_step"))

            # ---------- Offer weights (latest heatmap style table) ----------
            st.subheader("Current Offer Strategy")
            latest_offer = pd.DataFrame(params_df.iloc[-1]["offer_weights"])
            st.dataframe(latest_offer.style.format("{:.2f}"))

            # ---------- Delta (change in last batch) ----------
            if len(params_df) > 1:
                prev_offer = pd.DataFrame(params_df.iloc[-2]["offer_weights"])
                curr_offer = pd.DataFrame(params_df.iloc[-1]["offer_weights"])

                delta_offer = (curr_offer - prev_offer).round(3)

                st.subheader("Recent Learning Update (Δ weights)")
                st.dataframe(delta_offer)

                # ---------- Feature delta ----------
                prev_feature = params_df.iloc[-2]["feature_weights"]
                curr_feature = params_df.iloc[-1]["feature_weights"]

                delta_feature = {
                    k: round(float(curr_feature[k]) - float(prev_feature[k]), 4)
                    for k in curr_feature
                }

                st.subheader("Feature Importance Change")
                st.json(delta_feature)

            # ---------- Quick insights ----------
            if len(params_df) > 1:
                st.subheader("Key Insights")

                # strongest feature
                latest_fw = params_df.iloc[-1]["feature_weights"]
                top_feature = max(latest_fw, key=lambda x: abs(latest_fw[x]))

                # most changed offer weight
                if len(params_df) > 1:
                    change_abs = delta_offer.abs()
                    max_change = change_abs.stack().idxmax()

                    st.write(f"• Most influential feature: **{top_feature}**")
                    st.write(f"• Largest update: **{max_change[1]} → {max_change[0]}**")

    customer_query = st.text_area(
        "Agent Query",
        value="Analyze the customer, estimate churn risk, and recommend retention actions.",
        height=100,
    )

    st.caption("Feature selection and offer scoring are computed locally; Groq is used to turn the scored offers into natural-language recommendations when available.")
    if LANGCHAIN_AVAILABLE and os.getenv("GROQ_API_KEY"):
        st.caption("Groq offer generation is enabled.")
        if LANGCHAIN_RUNTIME_ERROR:
            st.info(f"Groq status: {LANGCHAIN_RUNTIME_ERROR}")
    elif LANGCHAIN_AVAILABLE:
        st.caption("LangChain packages are available, but `GROQ_API_KEY` is missing. Offer text falls back to the local template.")
    else:
        st.caption(f"LangChain packages are unavailable: {LANGCHAIN_IMPORT_ERROR}. Offer text falls back to the local template.")
    if st.button("Analyze Customer"):
        try:
            result = analyze_customer(customer_id.strip(), customer_query.strip())
            st.session_state["latest_agent_result"] = result
            st.subheader("Churn Risk")
            st.write(result["churn_risk"])

            st.subheader("Insights")
            for insight in result["insights"]:
                st.write("•", insight)

            st.subheader("Recommended Offer")
            st.write(result["recommended_actions"][0])   # main offer

            st.subheader("Suggested Improvements")
            for action in result["recommended_actions"][1:]:
                st.write("•", action)
        except Exception as exc:
            st.error(str(exc))

    latest_result = st.session_state.get("latest_agent_result")
    if latest_result:
        st.subheader("Feedback")

        # ---------- FEEDBACK UI ----------
        st.subheader("Customer Feedback")

        col1, col2 = st.columns([2, 3])

        # ---------- Acceptance (clean UX) ----------
        with col1:
            response = st.radio(
                "Customer Response",
                ["Accepted", "Rejected"],
                horizontal=True
            )
            accepted_offer = (response == "Accepted")

        # ---------- Structured feedback ----------
        with col2:
            feedback_reason = st.selectbox(
                "Feedback Reason",
                [
                    "Liked benefits",
                    "Too expensive",
                    "Not relevant",
                    "Prefers other offer",
                    "No clear reason"
                ]
            )


        # ---------- Single action button ----------
        if st.button("Update & Learn", use_container_width=True):
            try:
                updated_result = process_feedback(latest_result,accepted_offer,feedback_reason)

                st.session_state["latest_agent_result"] = updated_result

                st.success("Feedback recorded and learning updated.")

                if not accepted_offer:
                    st.info("A replacement offer has been generated.")

                st.subheader("Updated Customer Status")

                # ---------- 1. Churn ----------
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Churn Risk (Before)",
                        f"{updated_result['churn_risk']:.3f}"
                    )

                with col2:
                    st.metric(
                        "Churn Risk (After)",
                        f"{updated_result['post_offer_churn']:.3f}",
                        delta=round(updated_result['post_offer_churn'] - updated_result['churn_risk'], 4)
                    )

                # ---------- 2. Insights ----------
                st.subheader("Key Insights")
                for insight in updated_result["insights"]:
                    st.write("•", insight)

                # ---------- 3. Main Offer ----------
                st.subheader("Recommended Offer")
                st.success(updated_result["recommended_actions"][0])

                # ---------- 4. Improvements ----------
                st.subheader("Focus Areas")
                for action in updated_result["recommended_actions"][1:]:
                    st.write("•", action)

                # ---------- 5. Offer Breakdown ----------
                st.subheader("Offer Strategy Breakdown")

                offer_df = pd.DataFrame([
                    {
                        "Offer": o["label"],
                        "Score": round(o["score"], 2),
                        "Probability": round(o["probability"], 4)
                    }
                    for o in updated_result["offer_context"]["offer_rankings"]
                ])

                st.dataframe(offer_df, use_container_width=True)

                # ---------- 6. Selected Features ----------
                st.subheader("Key Drivers")

                feature_df = pd.DataFrame({
                    "Feature": list(updated_result["offer_context"]["selected_feature_scores"].keys()),
                    "Importance": list(updated_result["offer_context"]["selected_feature_scores"].values())
                })

                st.bar_chart(feature_df.set_index("Feature"))

                # ---------- 7. Learning Feedback ----------
                st.subheader("Learning Update")

                st.info("System updated using customer feedback.")

                if updated_result["post_offer_churn"] < updated_result["churn_risk"]:
                    st.success("Retention strategy improved churn risk.")
                else:
                    st.warning("No improvement observed. System will adapt next iteration.")
            except Exception as exc:
                st.error(str(exc))
st.markdown("Built with Streamlit, TensorFlow, and LangChain.")
