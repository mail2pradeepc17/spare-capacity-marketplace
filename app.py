import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np

# --- Configure Gemini API ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-pro')

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/spare_capacity.csv")
    return df

df = load_data()

# --- Helper Function: Match Query with Offers using Gemini ---
def match_offers(user_query, offers):
    prompt = f"""
    You are an AI assistant helping to find the best matching spare capacity offers.
    
    User Request: "{user_query}"
    
    Here are the available offers:
    {offers}
    
    Task:
    1. Identify which offers most closely match the user's request.
    2. Return a list of up to 5 offers with a relevance score (0-100%) and a short explanation.
    
    Format your response strictly in this JSON structure:
    [
      {{
        "id": <int>,
        "relevance_score": <int>,
        "reason": "<explanation>"
      }},
      ...
    ]
    """
    try:
        response = model.generate_content(prompt)
        return eval(response.text)  # Warning: Use safe parsing in production
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return []

# --- Streamlit UI ---
st.set_page_config(page_title="Open Innovation Spare Capacity Marketplace", layout="centered")

st.title("📦 Open Innovation Marketplace for Spare Capacity")
st.markdown("Find unused logistics capacity fast — powered by AI!")

# Input Section
user_request = st.text_input("Describe your need:", placeholder="E.g., Looking for 10 tons of truck space from Delhi to Kolkata")

if st.button("🔍 Find Matches"):
    if not user_request.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Matching offers..."):

            # Prepare offer descriptions
            offers_list = ""
            for idx, row in df.iterrows():
                offers_list += f"{idx+1}. [{row['Type']}] in {row['Location']} | {row['Description']} | Available: {row['Available_From']} to {row['Available_To']}\n"

            # Get matches from Gemini
            matches = match_offers(user_request, offers_list)

            if not matches:
                st.info("No relevant matches found.")
            else:
                st.subheader("✅ Top Matches Found:")
                for match in matches:
                    offer_idx = match["id"] - 1
                    offer = df.iloc[offer_idx]
                    score = match["relevance_score"]
                    reason = match["reason"]

                    with st.expander(f"Match #{match['id']} - {score}% Relevance"):
                        st.markdown(f"**Type:** {offer['Type']}")
                        st.markdown(f"**Location:** {offer['Location']}")
                        st.markdown(f"**Description:** {offer['Description']}")
                        st.markdown(f"**Availability:** {offer['Available_From']} to {offer['Available_To']}")
                        st.markdown(f"**AI Reasoning:** {reason}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Google Gemini API and Streamlit")