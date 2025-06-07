# importing dependencies
import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model=pickle.load(open('trained_model.pkl','rb'))

# loading the scaler
loaded_scaler=pickle.load(open('scaler.sav','rb'))

# set web page title and layout to wide/centered
st.set_page_config(page_title="Boston House Price Prediction",page_icon="https://miro.medium.com/v2/resize:fit:1024/0*YMZOAO8QE4bZ4_Rk.jpg",layout="wide")

# adjust padding accross title
st.markdown("""
        <style>
               .block-container{
                    padding-top: 1.23rem;
                }
        </style>
        """,unsafe_allow_html=True)
        
# creating function to predict house price
def predict_house_price(input_data):
    # changing input data to numpy 2d array
    input_data=np.array(input_data).reshape(1,-1)
    
    # standardize input data
    input_data_scaled=loaded_scaler.transform(input_data)
    
    predicted_price=loaded_model.predict(input_data_scaled)
    
    return predicted_price[0]

def main():
    # giving title
    st.title(':orange[House Price Prediction] :house:')
    
    # giving sub header and description
    col1,col2=st.columns([0.75,10])
    
    with col1:
        st.markdown("<span style='font-size:20px;font-weight:bold'>:blue[Boston]</span>", unsafe_allow_html=True)
    with col2:
        st.markdown("<span style='font-size:20px'>(Prediction will be done according to Boston Standard Metropolitan Statistical Area (SMSA) in 1970)</span>", unsafe_allow_html=True)
        
    # adding vertical space after title
    st.markdown("<br>", unsafe_allow_html=True)
    
    # divide 3 columns for each row
    col1,col2,col3=st.columns(3)
    
    # getting user input
    with col1:
        lstat=st.number_input("% lower status of the population",min_value=0.0,value=4.03,format="%.4f")
    with col2:
        rm=st.number_input("Average number of rooms per dwelling",min_value=0.0,value=7.185,format="%.4f")
    with col3:    
        dis=st.number_input("Weighted distances to ﬁve Boston employment centers",min_value=0.0,value=4.9671,format="%.4f")
    with col1:
        tax=st.number_input("Full-value property-tax rate per $10,000",min_value=0.0,value=242.0,format="%.4f")
    with col2:
        pt_ratio=st.number_input("Pupil-Teacher ratio by town 12",min_value=0.0,value=17.8,format="%.4f")
    
    price=0.0
    input_data=[lstat,rm,dis,tax,pt_ratio]
    
    # creating price prediction button 
    with col1:
        if st.button("Predict House Price"):
            price=predict_house_price(input_data)  
    with col1:
        st.success(f"Anticipated House Price : $ {price:.4f}K")
    
if __name__=="__main__":
    main()