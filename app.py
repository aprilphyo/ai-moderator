import streamlit as st

from src.predictor import predict_comment
st.set_page_config(
    page_title="AI Moderator",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Moderator")

st.write("Welcome to AI Moderator!")

st.write(
    """
    This application will use Artificial Intelligence to analyse
    user-generated comments and help moderators decide whether content
    should be:

    ✅ Allowed

    ⚠️ Reviewed

    ❌ Removed
    """
)

with st.form("moderation_form"):
    comment = st.text_input(
        "Enter a comment to analyse:",
        placeholder="Type a comment here..."
    )

    analyze_button = st.form_submit_button("Analyze")




if analyze_button:
    if not comment.strip():
        st.warning("Please enter a comment before analysing.")
    else:
        st.success("Analysis started!")

        result = predict_comment(comment)

        st.subheader("Analysis Result")
        st.metric("Risk Score", f"{result['score']}%")

        st.write("Risk Level:")

        if result["risk_level"] == "Low":
            st.success("🟢 Low Risk")
        elif result["risk_level"] == "Medium":
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

        st.write("Category Scores:")

        for category, score in result["category_scores"].items():
            st.progress(score / 100)
            st.write(
                f"**{category.replace('_', ' ').title()}** — {score}%"
            )

        st.write("Decision:")

        if result["decision"] == "✅ Allow":
            st.success(result["decision"])
        elif result["decision"] == "⚠️ Review":
            st.warning(result["decision"])
        else:
            st.error(result["decision"])

        st.write("Comment:")
        st.write(comment)