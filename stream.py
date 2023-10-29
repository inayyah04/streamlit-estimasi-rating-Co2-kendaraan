import pickle
import streamlit as st 
import setuptools
from PIL import Image


# membaca model
co_model = pickle.load(open('estimasi_rating_karbondioksida_kendaraan.sav','rb'))
image = Image.open('banner.jpg')

#judul web
st.image(image, caption='')
st.title('Aplikasi Prediksi Rating Co2 Kendaraan')

col1, col2,col3=st.columns(3)
with col1:
    model_year = st.number_input('Input Model Year :')
with col2:
    engine_size  = st.number_input('Input Engine Size :')
with col3:
    Cylinders  = st.number_input('Cylinders :')
with col1:
    fuel_consumption_city = st.number_input('fuel_consumption_city :')
with col2:
    fuel_consumption_hwy = st.number_input('fuel_consumption_hwy :')
with col3:
    fuel_consumption_comb = st.number_input('fuel_consumption_comb :')
with col1:
    fuel_consumption_mpg = st.number_input('fuel_consumption_mpg :')
with col2:
    co2_emisi = st.number_input('co2_emisi :')

#code untuk estimasi
ins_est=''

#membuat button
with col1:
    if st.button('Estimasi Rating Co2'):
        co_pred = co_model.predict([[model_year,engine_size,Cylinders,fuel_consumption_city,fuel_consumption_hwy,fuel_consumption_comb,fuel_consumption_mpg,co2_emisi]])

        st.success(f'Estimasi Rating Co2 : {co_pred[0]:.2f}')