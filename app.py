import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# KONFIGURASI HALAMAN
st.set_page_config(page_title="Supermarket Price Forecast Dashboard", layout="wide")
st.title("🛒 Dashboard Forecasting Harga Supermarket UK")
st.markdown("Dashboard ini menganalisis tren harga masa depan menggunakan model ARIMA dengan membandingkan produk Own Brand vs National Brand serta menganalisis tren harga pada kategori bahan pokok. Data yang digunakan bersumber dari Kaggle: [Time Series UK Supermarket Data](https://www.kaggle.com/datasets/declanmcalinden/time-series-uk-supermarket-data).")

# FUNGSI LOAD DATA
@st.cache_data
def load_data():
    df_ide1 = pd.read_csv("Forecast_Ide1_OwnBrand.csv")
    df_ide2 = pd.read_csv("Forecast_Ide2_TopCategories.csv")
    
    df_ide1['date'] = pd.to_datetime(df_ide1['date'])
    df_ide2['date'] = pd.to_datetime(df_ide2['date'])
    

    df_ide1['brand_label'] = df_ide1['brand_type'].apply(
        lambda x: "Own Brand" if "True" in str(x) else "National Brand"
    )
    return df_ide1, df_ide2

try:
    df_ide1, df_ide2 = load_data()
except FileNotFoundError:
    st.error("File CSV tidak ditemukan. Pastikan 'Forecast_Ide1_OwnBrand.csv' dan 'Forecast_Ide2_TopCategories.csv' ada di direktori yang sama.")
    st.stop()


tab1, tab2 = st.tabs(["Own Brand vs National Brand", "Prediksi Harga Kategori Bahan Pokok"])

# OWN BRAND VS NATIONAL BRAND
with tab1:
    st.header("Analisis Own Brand vs National Brand")
    
    supermarket_list = df_ide1['supermarket'].unique()
    selected_market = st.selectbox("Pilih Supermarket:", supermarket_list, key="market_ide1")
    
    df_market = df_ide1[df_ide1['supermarket'] == selected_market]
    
    col1, col2 = st.columns(2)
    
    # Prediksi Pergerakan Harga Harian
    with col1:
        st.subheader("Tren Forecast Harga Harian")
        fig1 = px.line(df_market, x='date', y='forecasted_prices', color='brand_label', 
                       title=f"Forecast Harga: Own Brand vs National Brand ({selected_market})",
                       labels={'forecasted_prices': 'Harga (£)', 'date': 'Tanggal'})
        st.plotly_chart(fig1, use_container_width=True)
        
    # Prediksi Selisih Harga
    with col2:
        st.subheader("Margin Harga")
        # Pivot data untuk menghitung selisih harian
        df_pivot = df_market.pivot_table(index='date', columns='brand_label', values='forecasted_prices').dropna()
        if 'Own Brand' in df_pivot.columns and 'National Brand' in df_pivot.columns:
            df_pivot['Price Gap (£)'] = df_pivot['National Brand'] - df_pivot['Own Brand']
            fig2 = px.area(df_pivot, y='Price Gap (£)', title="Selisih Harga National Brand terhadap Own Brand")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Data Own Brand / National Brand tidak lengkap untuk menghitung Price Gap.")

    st.subheader("Inflasi Internal & Ketahanan Supermarket")
    col3, col4 = st.columns(2)
    
    with col3:
        # Menghitung % kenaikan harga dari awal ke akhir periode forecast
        inflation_data = []
        for market in supermarket_list:
            df_temp = df_ide1[df_ide1['supermarket'] == market]
            for brand in ['Own Brand', 'National Brand']:
                df_b = df_temp[df_temp['brand_label'] == brand].sort_values('date')
                if not df_b.empty:
                    start_price = df_b['forecasted_prices'].iloc[0]
                    end_price = df_b['forecasted_prices'].iloc[-1]
                    growth = ((end_price - start_price) / start_price) * 100
                    inflation_data.append({'Supermarket': market, 'Brand': brand, 'Inflasi (%)': growth})
        
        df_infl = pd.DataFrame(inflation_data)
        fig3 = px.bar(df_infl, x='Supermarket', y='Inflasi (%)', color='Brand', barmode='group',
                      title="PrediksiKenaikan Harga per Supermarket")
        st.plotly_chart(fig3, use_container_width=True)
        

# ==========================================
# TAB 2: FORECAST KATEGORI TERTENTU
# ==========================================
with tab2:
    st.header("Analisis Kategori Bahan Pokok (Top 3)")
    
    category_list = df_ide2['category'].unique()
    selected_cat = st.selectbox("Pilih Kategori:", category_list, key="cat_ide2")
    
    col_a, col_b = st.columns(2)
    
    # 1. Price Match Kompetitor (Perbandingan Kategori antar Supermarket)
    with col_a:
        st.subheader(f"Price Match: {selected_cat}")
        df_cat = df_ide2[df_ide2['category'] == selected_cat]
        fig_a = px.line(df_cat, x='date', y='forecasted_prices', color='supermarket',
                        title=f"Perang Harga Kategori {selected_cat}",
                        labels={'forecasted_prices': 'Harga (£)'})
        st.plotly_chart(fig_a, use_container_width=True)
        
    # 2. Pola Fluktuasi Mingguan (Seasonality)
    with col_b:
        st.subheader(f"Pola Fluktuasi Mingguan Aktual ({selected_cat})")
        df_cat_actual = df_cat.dropna(subset=['actual_prices']).copy()
        df_cat_actual['Hari'] = df_cat_actual['date'].dt.day_name()
        hari_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig_b = px.box(df_cat_actual, x='Hari', y='actual_prices', color='supermarket',
                       category_orders={'Hari': hari_order},
                       title="Apakah ada diskon spesifik di akhir pekan?")
        st.plotly_chart(fig_b, use_container_width=True)

    # 3. Sensitivitas Kategori
    st.subheader("Sensitivitas Kategori (Laju Kenaikan Harga)")
    col_c, col_d = st.columns(2)
    
    with col_c:
        # Menghitung kategori mana yang naiknya paling tajam (menggunakan slope rasio)
        sens_data = []
        for cat in category_list:
            df_temp_cat = df_ide2[df_ide2['category'] == cat].groupby('date')['forecasted_prices'].mean().reset_index()
            if not df_temp_cat.empty:
                start_p = df_temp_cat['forecasted_prices'].iloc[0]
                end_p = df_temp_cat['forecasted_prices'].iloc[-1]
                growth_cat = ((end_p - start_p) / start_p) * 100
                sens_data.append({'Kategori': cat, 'Proyeksi Kenaikan (%)': growth_cat})
                
        df_sens = pd.DataFrame(sens_data).sort_values('Proyeksi Kenaikan (%)', ascending=False)
        fig_c = px.bar(df_sens, x='Kategori', y='Proyeksi Kenaikan (%)', color='Kategori',
                       title="Kategori dengan Proyeksi Kenaikan Harga Tertinggi")
        st.plotly_chart(fig_c, use_container_width=True)
