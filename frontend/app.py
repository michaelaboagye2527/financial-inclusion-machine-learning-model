import streamlit as st
import requests

st.set_page_config(
    page_title="Financial Inclusion Predictor",
    layout="wide"
)
st.image(
    "frontend/images/banner1.png",
    use_container_width=True
)
API_URL = "http://localhost:8000/predict"

# Page title
#st.title("Financial Inclusion Prediction")
st.subheader("Bank Account Ownership Prediction")

st.write(
    "Enter the respondent's information below to predict "
    "whether the individual is likely to have a bank account."
)

st.divider()

# Categorical options
COUNTRIES = ["Rwanda", "Tanzania", "Kenya", "Uganda"]

YEARS = ["2016", "2017", "2018"]

LOCATION_TYPES = ["Rural", "Urban"]

CELLPHONE_ACCESS = ["Yes", "No"]

GENDERS = ["Female", "Male"]

RELATIONSHIPS = [
    "Head of Household",
    "Spouse",
    "Child",
    "Parent",
    "Other relative",
    "Other non-relatives"
]

MARITAL_STATUSES = [
    "Married/Living together",
    "Single/Never Married",
    "Widowed",
    "Divorced/Seperated",
    "Dont know"
]

EDUCATION_LEVELS = [
    "Primary education",
    "Secondary education",
    "No formal education",
    "Tertiary education",
    "Vocational/Specialised training",
    "Other/Dont know/RTA"
]

JOB_TYPES = [
    "Self employed",
    "Informally employed",
    "Farming and Fishing",
    "Remittance Dependent",
    "Formally employed Private",
    "Other Income",
    "No Income",
    "Formally employed Government",
    "Government Dependent",
    "Dont Know/Refuse to answer"
]


# Input form
with st.form("prediction_form"):

    st.subheader("Respondent Information")

    col1, col2 = st.columns(2)

    with col1:

        country = st.selectbox(
            "Country",
            COUNTRIES
        )

        year = st.selectbox(
            "Survey Year",
            YEARS
        )

        location_type = st.selectbox(
            "Location Type",
            LOCATION_TYPES
        )

        cellphone_access = st.selectbox(
            "Cellphone Access",
            CELLPHONE_ACCESS
        )

        household_size = st.number_input(
            "Household Size",
            min_value=0,
            step=1
        )

        age_of_respondent = st.number_input(
            "Age of Respondent",
            min_value=0,
            step=1
        )

    with col2:

        gender_of_respondent = st.selectbox(
            "Gender of Respondent",
            GENDERS
        )

        relationship_with_head = st.selectbox(
            "Relationship With Head",
            RELATIONSHIPS
        )

        marital_status = st.selectbox(
            "Marital Status",
            MARITAL_STATUSES
        )

        education_level = st.selectbox(
            "Education Level",
            EDUCATION_LEVELS
        )

        job_type = st.selectbox(
            "Job Type",
            JOB_TYPES
        )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Bank Account Ownership",
        use_container_width=True
    )


# Send data to FastAPI
if submitted:

    input_data = {
        "country": country,
        "year": year,
        "location_type": location_type,
        "cellphone_access": cellphone_access,
        "household_size": int(household_size),
        "age_of_respondent": int(age_of_respondent),
        "gender_of_respondent": gender_of_respondent,
        "relationship_with_head": relationship_with_head,
        "marital_status": marital_status,
        "education_level": education_level,
        "job_type": job_type
    }

    try:

        with st.spinner("Generating prediction..."):

            response = requests.post(
                API_URL,
                json=input_data,
                timeout=30
            )

            response.raise_for_status()

            result = response.json()

        prediction = result["prediction"]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:

            st.success(
                "The model predicts that this individual "
                "has a bank account."
            )

        else:

            st.warning(
                "The model predicts that this individual "
                "does not have a bank account."
            )

        st.metric(
            label="Predicted Class",
            value=prediction
        )

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the prediction API. "
            "Ensure that FastAPI is running."
        )

    except requests.exceptions.HTTPError as e:

        st.error(
            f"API returned an error: {e.response.text}"
        )

    except requests.exceptions.RequestException as e:

        st.error(f"Request failed: {e}")

    except (KeyError, ValueError) as e:

        st.error(f"Invalid API response: {e}")