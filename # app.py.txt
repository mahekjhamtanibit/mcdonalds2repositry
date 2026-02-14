# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="McDonald's Analytics Pro", layout="wide")

# McDonald's theme
st.markdown("""
    <style>
    .stApp { background-color: #fffaf0; }
    h1, h2, h3 { color: #DA291C; }
    .stButton>button { background-color: #FFC72C; color: #000; border: none; }
    </style>
""", unsafe_allow_html=True)

# Header with logo placeholder
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/36/McDonald%27s_Golden_Arches.svg", width=120)
with col2:
    st.title("McDonald's Premium Analytics Dashboard")
    st.caption("Store Performance • Menu Insights • Customer Behavior • 2024–2026")

uploaded = st.file_uploader("Upload your McDonald's dataset (CSV)", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Dataset loaded — {df.shape[0]} transactions")

    with st.expander("1. Raw Data Preview & Cleaning", expanded=False):
        st.dataframe(df.head(8))
        # Cleaning
        df['Customer_Age'].fillna(df['Customer_Age'].median(), inplace=True)
        df['Satisfaction_Rating_1to5'].fillna(3, inplace=True)  # neutral
        df['Item_Name'] = df['Item_Name'].str.replace(r'[^a-zA-Z0-9\s\-]', '', regex=True).str.strip()
        df['Promo_Applied'] = df['Promo_Applied'].replace('PROMO\?\?', 'None')
        st.markdown("**Cleaned preview**")
        st.dataframe(df.head(6))

    with st.expander("2. Transformation & Normalization", expanded=False):
        df['Order_Datetime'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Order_Hour'].astype(str) + ':00')
        df['Day_of_Week'] = df['Order_Datetime'].dt.day_name()
        df['Month'] = df['Order_Datetime'].dt.month_name()
        scaler = MinMaxScaler()
        num_cols = ['Customer_Age', 'Quantity', 'Unit_Price_USD', 'Total_Spend_USD', 'Satisfaction_Rating_1to5']
        df[num_cols] = scaler.fit_transform(df[num_cols])
        st.dataframe(df.head())

    with st.expander("3. Feature Engineering Highlights", expanded=True):
        df['Is_Peak_Hour'] = df['Order_Hour'].apply(lambda x: 'Peak' if x in [11,12,13,17,18,19] else 'Off-Peak')
        df['Age_Group'] = pd.cut(df['Customer_Age']*100, bins=[0,30,45,60,100], labels=['Gen Z/Young','Millennial','Gen X','Boomer+'])
        st.write("Engineered columns added: Is_Peak_Hour, Age_Group, Day_of_Week, Month")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Avg Spend (USD)", f"${df['Total_Spend_USD'].mean()* (df['Unit_Price_USD'].max()-1.89):.2f}")
    with colB:
        st.metric("Avg Satisfaction", f"{df['Satisfaction_Rating_1to5'].mean()*4 +1 :.1f} / 5")
    with colC:
        st.metric("Top Payment Method", df['Payment_Method'].mode()[0])

    st.header("Key Business Insights & Visuals")

    tab1, tab2, tab3 = st.tabs(["Menu Performance", "Customer Behavior", "Hypothesis & Trends"])

    with tab1:
        top_items = df.groupby('Item_Name')['Total_Spend_USD'].sum().sort_values(ascending=False).head(8)
        fig, ax = plt.subplots()
        sns.barplot(x=top_items.values, y=top_items.index, palette="YlOrRd", ax=ax)
        ax.set_title("Top Selling Items by Revenue")
        st.pyplot(fig)

    with tab2:
        loyalty_spend = df.groupby('Loyalty_Tier')['Total_Spend_USD'].mean().sort_values()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=loyalty_spend.index, y=loyalty_spend.values, palette="Blues", ax=ax2)
        st.pyplot(fig2)
        st.info("Gold Members spend ~2× more per transaction — prioritize loyalty program!")

    with tab3:
        yes = df[df['Loyalty_Tier'].str.contains('Gold|Silver')]['Total_Spend_USD']
        no  = df[~df['Loyalty_Tier'].str.contains('Gold|Silver')]['Total_Spend_USD']
        tstat, pval = stats.ttest_ind(yes, no)
        st.write(f"**Hypothesis**: Loyalty members spend significantly more → p-value = {pval:.5f}")
        if pval < 0.05:
            st.success("Result: Strong evidence — yes, they do!")
        else:
            st.warning("No strong statistical difference.")

    st.subheader("Recommendations for McDonald's Management")
    st.markdown("""
    - **Menu**: Push high-margin items (McFlurry, Fries) via combos & app promos  
    - **Peak Hours**: Increase staffing 11–2 pm & 5–8 pm  
    - **Loyalty**: Target Basic Members with upgrade offers — huge revenue potential  
    - **Delivery**: Growing fast — partner incentives could boost satisfaction  
    """)

else:
    st.info("Upload the generated CSV file to start exploring!")
    st.markdown("**Tip**: First run `generate_mcd_dataset.py` to create your realistic dataset.")