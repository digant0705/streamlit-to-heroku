# app1.py
import streamlit as st
def app():
    st.title('Lead Conversion Classification')
    

    html_temp = """
            <div style ="background-color:powderblue;padding:13px">
            <h1 style ="color:black;text-align:center;">Conclusion </h1>
            <p>Both models are performing pretty good in classifying the data with an accuracy of over 90%.
                The scoring of the leads will help the company employees to reach out to the most 
               probable customers and increase the sales of the company and promote revenue generation. 
               This will decrease the customer acquisition cost and save millions for the company.
            </p>
            </div>

                    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)

    