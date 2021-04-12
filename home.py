#app.pyimport app1
import model1
import model2
import default
import project
import conclusion
import streamlit as st
PAGES = {
    "Home":default,
    "Project Description":project,
    "Random Forest Classifier": model1,
    "Logistic Regression": model2,
    "Conclusion":conclusion
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()