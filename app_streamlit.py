import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title='Klasifikasi Jeruk', page_icon=':tangerine:')

model = joblib.load('model_klasifikasi_jeruk.joblib')

st.title(':tangerine: Klasifikasi Jeruk')
st.markdown('ini adalah model untuk mengklasifikasikan jeruk')

diameter = st.slider('Diameter',4.0,10.0,6.5) 	
berat = st.slider('Berat',100.0,250.0,210.0) 	
tebal_kulit = st.slider('Tebal Kulit',0.2,1.0,0.8)
kadar_gula = st.slider('Kadar Gula',8.0,14.0,12.0)
asal_daerah = st.pills('Asal Daerah', ['Kalimantan', 'Jawa Barat', 'Jawa Tengah'], default='Kalimantan')	
warna = st.pills('Warna', ['kuning', 'hijau','oranye'], default='kuning') 	
musim_panen = st.pills('Musim Panen',['kemarau', 'hujan'], default='kemarau')


if st.button('Prediksi',type='primary'):
    data = pd.DataFrame({
            'diameter': [diameter],
            'berat': [berat],
            'tebal_kulit': [tebal_kulit],
            'kadar_gula': [kadar_gula],
            'asal_daerah': [asal_daerah],
            'warna': [warna],
            'musim_panen': [musim_panen]
        })

    prediksi = model.predict(data)[0]
    persentase = max(model.predict_proba(data)[0])
    st.success(f'Model Memprediksi **{prediksi}** dengan skor **{persentase:.2%}**')
    st.balloons()

st.divider()
st.caption('Created by Ahmad')
