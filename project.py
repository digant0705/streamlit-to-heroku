# app1.py
import streamlit as st
def app():
    st.title('Lead Conversion Classification')
    

    html_temp = """
            <div style ="background-color:powderblue;padding:13px">
            <h1 style ="color:black;text-align:center;">Abstract </h1>
            <p>Most of the businesses and companies today rely on online marketing and a huge part of sales comes
            from online platforms. A lot of investment goes into funding all these operations but they can only 
            prove worthy if they enable us to identify potential customers and provide some criteria to filter the
            traffic that comes to our webpage. It can save any business millions of dollars if they can focus their 
            efforts on only the leads that have high chances of converting. This project will attempt to classify the
            leads according to the probability of their conversion as well as score these leads so that the business 
            is able to follow up on those leads.
            </p>
            </div>

            <div style ="background-color:powderblue;padding:13px">
            <h1 style ="color:black;text-align:center;">Introduction </h1>
            <p> An online platform called X education provides educational courses tailored towards working 
            professionals. They market their courses on various social media platforms such as Google and Facebook which
            generates a huge chunk of footprint as well as have some organic traffic on their webpage. When someone fills
            a form on the webpage for more information, it is counted as lead. Additionally, they get leads through
            referrals by their previous customers. After these leads are identified, the company staff communicates with 
            them via email or phone call and try to convince them to buy the courses offered. This takes a lot of effort
            and time on the part of company employees but does not yield very good results. Out of 100 leads a day, only
            about 30 are converted to customers. The company wants to identify the leads that have high probability of 
            converting i.e., “Hot Leads”. This will help the company employees to communicate with these people first 
            and would also be beneficial for the business as it will be able to target the right audience.
            </p>
            </div>

                    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)

    