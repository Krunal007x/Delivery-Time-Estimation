import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler with error handling
def load_model_and_scaler():
    try:
        with open("linear_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model or scaler file not found. Please ensure both files are in the working directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# Custom Styling
st.markdown(
    """
    <style>
        .main {background-color: #f0f2f6;}
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 24px;
        }
        .stTextInput, .stNumberInput {
            border-radius: 10px;
        }
        .stMarkdown {
            font-size: 20px;
            color: #2E3B4E;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title and Description
st.title("🌍📦 Estimated Delivery Time Predictor")
st.markdown("Enter your shipment details below to predict the estimated delivery time:")

# User Input Fields with tooltips
customer_zip = st.number_input("🏠 Customer Zip Code", min_value=1000, max_value=99999, step=1, help="Enter the 5-digit zip code of the customer")
product_weight = st.number_input("⚖️ Product Weight (in grams)", min_value=1, max_value=50000, step=1, help="Weight of the product being shipped")
product_length = st.number_input("📏 Product Length (in cm)", min_value=1, max_value=200, step=1, help="Length of the package")
product_height = st.number_input("📦 Product Height (in cm)", min_value=1, max_value=200, step=1, help="Height of the package")
product_width = st.number_input("📐 Product Width (in cm)", min_value=1, max_value=200, step=1, help="Width of the package")
seller_zip = st.number_input("🏪 Seller Zip Code", min_value=1000, max_value=99999, step=1, help="Enter the 5-digit zip code of the seller")

# Predict Button
if st.button("🚀 Predict Delivery Time"):
    input_features = np.array([
        [customer_zip, product_weight, product_length, product_height, product_width, seller_zip]
    ])
    
    # Optional: Debug info
    # st.write("🔍 Raw Input:", input_features)
    
    try:
        scaled_features = scaler.transform(input_features)
        prediction = model.predict(scaled_features)[0]
        st.success(f"📅 Estimated Delivery Time: **{prediction:.2f} days**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
