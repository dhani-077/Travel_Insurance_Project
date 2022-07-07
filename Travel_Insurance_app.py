import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Travel Insurance",
                   page_icon=":bar_chart:",
                   layout="wide")

st.title("Travel Insurance Prediction Project")
st.markdown("##")

"""
This travel insurance project is created to help the Tour and Travel company
to promote their travel insurance product to old customer. The new insurance
cover covid 19 protection. The project is to predict wether their customer
from 2019 before covid will buy a new travel insurance program or not, so 
the company will get valuable insight to help them make decision.

In this project I'm doing some exploratory data analysis and make a simple
dashboard to help finding a pattern and correlation about the customer decision,
and after that I'm developing machine learning app to predict it.

The data is retreived from kaggle: https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data

Let's take a look at the dataset and data visualisation

"""

def get_data():    
    df = pd.read_csv("TravelInsurancePrediction.csv")
    df.drop(df.columns[0], axis=1, inplace=True)
    return df
    
df = get_data()

st.dataframe(df)

st.title("Travel Insurance Dashboard")

# Dashboard Filter
option = st.selectbox(
    "Filter Chart By:",
    df.drop(["AnnualIncome","TravelInsurance"],axis=1).columns.values, index=1
    )

# Chart for Travel Insurance Customer
cust_by = df.query("TravelInsurance == 1").groupby(option).count()['TravelInsurance']

fig_insurance = px.bar(
    cust_by,
    x="TravelInsurance",
    y=cust_by.index,
    orientation="h",
    text_auto=True,
    title="<b>Travel Insurance Customer by {}</b>".format(option),
    color_discrete_sequence=["#008388"] * len(cust_by),
    template="plotly_white"
    )

fig_insurance.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
    )

# Chart for Non Travel Insurance Customer
nocust_by = df.query("TravelInsurance == 0").groupby(option).count()['TravelInsurance']

fig_noinsurance = px.bar(
    nocust_by,
    x="TravelInsurance",
    y=cust_by.index,
    orientation="h",
    text_auto=True,
    title="<b>Non Travel Insurance Customer by {}</b>".format(option),
    color_discrete_sequence=["#008100"] * len(nocust_by),
    template="plotly_white"
    )

fig_noinsurance.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
    )

# Display and layouting the dashboard
left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_insurance)
right_column.plotly_chart(fig_noinsurance)

b1, b2 = st.columns((2, 1))
income_slider = b1.slider(
    "Annual Income (Rupee)",
    min_value=min(df["AnnualIncome"]),
    max_value=max(df["AnnualIncome"]),
    value=(min(df["AnnualIncome"]),
           max(df["AnnualIncome"])), # Range Value
    step=10000
    )

def income():
    income_df = df[df["AnnualIncome"].between(min(income_slider), max(income_slider))]
    fig_income = px.scatter(
        income_df,
        x=income_df.index,
        y=income_df["AnnualIncome"],
        color="TravelInsurance",
        title="<b>Annual Income and Travel Insurance Corr</b>"
        )
    fig_income.update_layout(
        title={
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
        )
    return fig_income

b1.plotly_chart(income())
b2.subheader("SUMMARY")
b2.write("""
           From this dashboard we can see the column value that has most impact
           on customer decision to purchase travel insurance is the customer 
           annual income, has travelled abroad, and is frequent flyer.
           The other columns has little to do with customer purchasing decision.
           """)

# Make correlation between columns
def dummy():
    dummy = pd.get_dummies(df[["Employment Type",
                                "GraduateOrNot",
                                "FrequentFlyer", 
                                "EverTravelledAbroad"]]
                            )
    df_corr = pd.read_csv("TravelInsurancePrediction.csv")
    df_corr[["Employment Type",
              "GraduateOrNot",
              "FrequentFlyer",
              "EverTravelledAbroad"]] = dummy.drop(dummy.columns[[0,2,4,6]],
                                                      axis=1)
    return df_corr

st.title("Travel Insurance Matrix Corrrelation Heatmap")
nc1, nc2 = st.columns((2, 1))
fig_corr = px.imshow(dummy().corr(), text_auto=True)
nc1.plotly_chart(fig_corr)
nc2.write("""
          With this correlation we can be sure that the assumption from summary
          above doesn't change. So the value that will be used in this machine learning 
          is annual income, has travelled abroad, and is frequent flyer.
          """)
          
# Deploy the model into streamlit
st.title("Travel Insurance Customer Prediction")

st.write("""
         This is the machine learning model that has been created using the 
         column selected above. This model will be predict the customer
         decision. You can input the customer annual income and choose if
         the customer is frequent flyer and/or has travelled abroad, after
         that it will predict the customer if they interested or not interested
         to buy the travel insurance.
         """)
         
def ml_model():
    mc1, mc2, mc3 = st.columns(3)
    annual_income = mc1.number_input(
        "Annual Income (Rupee)",
        value=1300000,
        step=10000
        )
    frequent_flyer = mc2.selectbox("Frequent Flyer", ["No", "Yes"], index=1)
    if frequent_flyer == "No":
        frequent_flyer = 0
    else:
        frequent_flyer = 1
    ever_travelled = mc3.selectbox("Ever Travelled Abroad", ["No", "Yes"], index=1)
    if ever_travelled == "No":
        ever_travelled = 0
    else:
        ever_travelled = 1
    data = {"AnnualIncome": annual_income,
            "FrequentFlyer": frequent_flyer,
            "EverTravelledAbroad": ever_travelled}
    feat = pd.DataFrame(data, index=[0])
    return feat

df_model = ml_model()

# Make Machine Learning model
col_feat = ["AnnualIncome", "FrequentFlyer", "EverTravelledAbroad"]
X = dummy()[col_feat]
y = dummy()["TravelInsurance"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.write('Prediction Score :', clf.score(X_test, y_test))

result = ""

# Prediction action
if st.button("Predict Customer Decision"):
    result = clf.predict(df_model)
    if result == 0:
        result = 'Not Interested'
    else:
        result = 'Interested'
    st.success("This customer is {} purchasing travel insurance".format(result))

st.write("""
         We can see the prediction score is quite high at 80%, the model that
         been created is considered as good. Now that we have the model next step
         is using this to help improve the business.
         """)    

# Creating the business model
st.title("Travel Insurance Business Model")
st.write('''
         The data above isn't explained about the business model, how much does
         the insurance cost, how much the coverage is, how they reach the customer,
         and so on.

         So we make a simple asumption, based on my knowledge about travel insurance
         they have to cover if there is accident, health issue. From my 
         finding on the internet the chance accident bound to happen is about
         30%. We take this chance to generate a simple calculation.
         
         From the asumption above then we generate some simple calculation
         to predict how much profit margin the company will make. You can fill
         the box below to make a prediction how much profit you want to make.
         The equation is:\n
         P = Ix(M(SxO))-(AxC)
    ''')
with st.expander('See The equation'):
    st.write('''
             P = Profit Margin\n
             I = Insurance Price\n
             M = Model ML percentage\n
             S = Percentage people who will buy\n
             O = How much potential customer\n
             A = Accident Percentage\n
             C = Coverage from insurance\n
            ''')
def P(I,M,S,O,A,C):
    margin = I*(M*(S*O))-(A*C)
    return margin
M = 0.8
S = 0.36
A = 0.3

pc1, pc2, pc3 = st.columns(3)
O = pc1.number_input("Number of potential customer", value=100, step=1)
I = pc2.number_input("Insurance price", value=1250, step=50)
C = pc3.number_input("Insurance coverage", value=100000, step=1000)

hasil=""
if st.button("Calculate Profit"):
    hasil = round(P(I,M,S,O,A,C))
    if hasil > 0:    
        st.success("Company profit is {:,} rupee".format(hasil))
    else:
        st.success("Company loss is {:,} rupee".format(hasil))
    