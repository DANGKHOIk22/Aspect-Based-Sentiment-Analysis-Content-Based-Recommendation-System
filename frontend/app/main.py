import streamlit as st
from api import predict
import pandas as pd
#from plugins.postgresql_operator import PostgresOperators
#from plugins.preprocessing import preprocess
# st.set_page_config(
#     page_title="Sentiment Analysis",
#     page_icon=":chart_with_upwards_trend:",
#     layout="wide",
#     initial_sidebar_state='auto'
# )
# def database(data):
#     warehouse_operator = PostgresOperators('postgres')
#     df = pd.DataFrame(data, columns=['Sentiment', 'Comment'])
#     warehouse_operator.save_data_to_postgres(
#         df,
#         "sentiment_data",
#         schema='warehouse',
#         if_exists='append'
#     )


st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 8px;
        }
        .stTextArea textarea {
            font-size: 16px;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for feedback
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = "Not rated yet"
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

def update_feedback():
    """Update feedback sentiment based on thumbs rating."""
    selected = st.session_state.get("thumb_rating")  # Get rating from session state
    if selected is not None:
        if selected == 1:  # Thumbs up
            st.session_state.user_feedback = "True"
        elif selected == 0:  # Thumbs down
            st.session_state.user_feedback = "False"
    else:
        st.session_state.user_feedback = "Not rated yet"

def main():
    st.title("📊 Aspect-based Sentiment Analysis")
    st.subheader("Model: DistilBERT | Dataset: SemEval-4 Restaurants")
    
    text_input = st.text_area("Enter a sentence:")
    
    if st.button("Analyze", type="primary"):
        results = predict(text_input)
        if "predictions" in results:
            preds = results["predictions"]
            
            st.markdown("### Analysis Result")
            st.write("**Term:**")
            st.markdown(preds["sentence_tags"])
            
            st.success(f'**Sentiment:** {preds["label"]}')
            
            # Reset feedback and mark analysis as done
            st.session_state.user_feedback = "Not rated yet"
            st.session_state.analysis_done = True
            st.session_state.preds = preds  # Store preds in session state
            data = [[st.session_state.user_feedback, text_input]]
            #database(data)
        else:
            st.error(results.get("error", "An error occurred during analysis."))
            st.session_state.analysis_done = False

    # Feedback section - only shows after analysis
    if st.session_state.analysis_done:
        st.write("Please provide your feedback:")
        st.feedback("thumbs", key="thumb_rating", on_change=update_feedback)
        st.write(f'**User Feedback Sentiment:** {st.session_state.user_feedback}')

if __name__ == "__main__":
    main()