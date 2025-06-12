import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from os.path import join, exists
import os
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import ElasticNet

st.set_page_config(
    page_title="Stock/ETF Visualizer",
    layout="wide"
)

@st.cache_data
def get_available_symbols(folder):
    return sorted([f.replace('.csv', '') for f in os.listdir(folder) if f.endswith('.csv')])

@st.cache_data
def load_data(symbol, folder):
    df = pd.read_csv(join(folder, f"{symbol}.csv"))
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

def get_stock_news():
    try:
        url = "https://stocknewsapi.com/api/v1/category?section=alltickers&items=3&page=1&token=pt5lmjz7hyjiqjnrjjtzdtknlp0u2z6ynl40atr4"
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()
        return json_data.get("data", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}. Please check your internet connection or API token.")
        return []
    except json.JSONDecodeError:
        st.error("Error decoding news response. The API might have returned an invalid JSON.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching news: {e}")
        return []

@st.cache_data
def get_cluster_map(csv_path="cluster_assignments.csv"):
    try:
        if not exists(csv_path):
            st.error(f"Cluster assignments file not found: {csv_path}")
            return {}
        df = pd.read_csv(csv_path)
        if 'cluster' not in df.columns:
            st.error(f"Error: 'cluster' column not found in cluster assignments CSV. Columns found: {df.columns.tolist()}")
            return {}
        valid_clusters = {0, 9, 12, 15}
        df['cluster'] = df['cluster'].apply(lambda x: x if x in valid_clusters else 'outliers')
        cluster_map = dict(zip(df['ticker'], df['cluster']))
        return cluster_map
    except Exception as e:
        st.error(f"Error loading cluster assignments: {e}")
        return {}

@st.cache_resource
def load_model(model_path, model_type):
    try:
        if not exists(model_path):
            st.warning(f"Model file not found: {model_path}. Skipping {model_type} prediction.")
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.warning(f"Error loading {model_type} model from {model_path}: {e}. Skipping prediction with this model.")
        return None

def predict_next_day_close(ticker, filtered_prediction_df, model_dir="cluster_models"):
    cluster_map = get_cluster_map()
    if not cluster_map:
        return None

    cluster = cluster_map.get(ticker, 'outliers')

    if cluster == 'outliers' and ticker not in cluster_map:
        st.info(f"No specific cluster assignment found for ticker **{ticker}**. Using a generic 'outliers' category for model lookup (if models for 'outliers' exist).")

    if filtered_prediction_df.empty:
        st.warning(f"No data available for prediction for {ticker} in the selected date range.")
        return None

    latest_data = filtered_prediction_df.iloc[-1]
    today_close = latest_data['Close']

    features_to_drop = ['Date', 'Target', 'Price_Up']
    actual_features_to_drop = [col for col in features_to_drop if col in latest_data.index]

    try:
        features = latest_data.drop(actual_features_to_drop, errors='ignore').values.reshape(1, -1)
        feature_names_used = latest_data.drop(actual_features_to_drop, errors='ignore').index.tolist()
    except Exception as e:
        st.error(f"Error preparing features for prediction: {e}")
        st.info(f"Available columns in prediction data: {latest_data.index.tolist()}")
        return None

    predictions = {
        'ElasticNet': {'prediction': None, 'change': None},
        'LGBM': {'prediction': None, 'change': None}
    }
    
    en_model_path = join(model_dir, f"cluster_{cluster}_en.pkl")
    en_model = load_model(en_model_path, 'ElasticNet')
    if en_model:
        try:
            en_pred = en_model.predict(features)[0]
            predictions['ElasticNet']['prediction'] = en_pred
            predictions['ElasticNet']['change'] = 'Increase' if en_pred > today_close else 'Decrease'
        except Exception as e:
            st.warning(f"ElasticNet prediction failed for cluster {cluster}: {e}. Model might expect different number of features (expected: {getattr(en_model, 'n_features_in_', 'unknown')}, got: {features.shape[1]}).")
            
    lgb_model_path = join(model_dir, f"cluster_{cluster}_lgb.pkl")
    lgb_model = load_model(lgb_model_path, 'LGBM')
    if lgb_model:
        try:
            lgb_pred = lgb_model.predict(features)[0]
            predictions['LGBM']['prediction'] = lgb_pred
            predictions['LGBM']['change'] = 'Increase' if lgb_pred > today_close else 'Decrease'
        except Exception as e:
            st.warning(f"LGBM prediction failed for cluster {cluster}: {e}. Model might expect different number of features (expected: {getattr(lgb_model, 'n_features_in_', 'unknown')}, got: {features.shape[1]}).")
            
    actual_next_day_close = None
    if 'Target' in filtered_prediction_df.columns and len(filtered_prediction_df) > 1:

        actual_next_day_close = latest_data['Target']


    result = {
        'Ticker': ticker,
        'Cluster': cluster,
        'Today Close': today_close,
        'Actual Next Day Close': actual_next_day_close,
        'Predictions': predictions,
        'Features Used': pd.DataFrame({'Feature': feature_names_used, 'Value': features.tolist()[0]})
    }
    
    return result

def display_prediction_results(prediction_data):
    if prediction_data is None:
        st.info("No prediction data to display. Please ensure a valid stock/ETF is selected and models are available.")
        return
    
    st.subheader(f"📈 Next Day Close Prediction for {prediction_data['Ticker']}")
    
    st.write(f"**Cluster Assignment:** {prediction_data['Cluster']}")
    st.write(f"**Today's Close Price:** ${prediction_data['Today Close']:.2f}")

    if prediction_data['Actual Next Day Close'] is not None:
        st.write(f"**Actual Next Day Close (from dataset):** ${prediction_data['Actual Next Day Close']:.2f}")
    else:
        st.info("Actual next day close not available in the dataset.")

    with st.expander("📊 Features Used for Prediction"):
        st.table(prediction_data['Features Used'])

    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("### ElasticNet Prediction")
        en_pred = prediction_data['Predictions']['ElasticNet']
        if en_pred['prediction'] is not None:
            st.metric(
                label="Predicted Close",
                value=f"${en_pred['prediction']:.2f}",
                delta=f"{en_pred['change']}",
                delta_color="normal"
            )
        else:
            st.warning("ElasticNet prediction not available (model not loaded or prediction failed).")
    
    with cols[1]:
        st.markdown("### LGBM Prediction")
        lgb_pred = prediction_data['Predictions']['LGBM']
        if lgb_pred['prediction'] is not None:
            st.metric(
                label="Predicted Close",
                value=f"${lgb_pred['prediction']:.2f}",
                delta=f"{lgb_pred['change']}",
                delta_color="normal"
            )
        else:
            st.warning("LGBM prediction not available (model not loaded or prediction failed).")

st.title("📊 Stock/ETF Dashboard")

mode = st.sidebar.radio("Select Mode", ["Visualization and Prediction", "Stock Market News"])

if mode == "Visualization and Prediction":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization Settings")
    data_type = st.sidebar.radio("Visualization Dataset Type", ["Stocks", "ETFs"], index=0)
    vis_folder = "stocks" if data_type == "Stocks" else "etfs"
    vis_symbols = get_available_symbols(vis_folder)

    if not vis_symbols:
        st.error(f"No valid datasets found in '{vis_folder}' folder for visualization.")
        st.stop()

    selected_vis_symbol = st.sidebar.selectbox("Select Asset for Visualization", vis_symbols, index=0)
    vis_df = load_data(selected_vis_symbol, vis_folder)

    if vis_df.empty:
        st.error("Selected visualization dataset is empty.")
        st.stop()

    min_date_vis = vis_df['Date'].min().to_pydatetime()
    max_date_vis = vis_df['Date'].max().to_pydatetime()

    date_range_vis = st.sidebar.slider(
        "Select Date Range for Visualization",
        min_value=min_date_vis,
        max_value=max_date_vis,
        value=(min_date_vis, max_date_vis),
        format="YYYY-MM-DD"
    )

    filtered_vis_df = vis_df[(vis_df['Date'] >= date_range_vis[0]) & (vis_df['Date'] <= date_range_vis[1])]

    if filtered_vis_df.empty or len(filtered_vis_df) < 2:
        st.warning("Not enough data in selected date range for visualization. Please choose a broader range.")
        st.stop()

    st.header("Stock/ETF Price Visualization")
    plot_type = st.selectbox(
        "Chart Type",
        ["Candlestick", "Moving Averages", "Close vs Volume Overlay"]
    )

    fig = None

    if plot_type == "Candlestick":
        if all(col in filtered_vis_df.columns for col in ["Open", "High", "Low", "Close"]):
            fig = go.Figure(go.Candlestick(
                x=filtered_vis_df['Date'],
                open=filtered_vis_df['Open'],
                high=filtered_vis_df['High'],
                low=filtered_vis_df['Low'],
                close=filtered_vis_df['Close']
            ))
            fig.update_layout(title=f"{selected_vis_symbol} Candlestick Chart")
        else:
            st.warning("Candlestick chart requires 'Open', 'High', 'Low', 'Close' columns.")
    elif plot_type == "Moving Averages":
        if len(filtered_vis_df) < 25:
            st.warning("Need at least 25 data points to show Moving Averages. Consider a broader date range.")
        else:
            filtered_vis_df['MA_10'] = filtered_vis_df['Close'].rolling(10).mean()
            filtered_vis_df['MA_25'] = filtered_vis_df['Close'].rolling(25).mean()
            fig = px.line(filtered_vis_df, x="Date", y=["Close", "MA_10", "MA_25"], title=f"{selected_vis_symbol} with MA 10 & 25")
    elif plot_type == "Close vs Volume Overlay":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_vis_df['Date'], y=filtered_vis_df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Bar(x=filtered_vis_df['Date'], y=filtered_vis_df['Volume'], name='Volume', yaxis='y2'))
        fig.update_layout(
            title=f"{selected_vis_symbol} Close Price and Volume",
            yaxis=dict(title="Close"),
            yaxis2=dict(title="Volume", overlaying='y', side='right'),
        )

    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Metrics for Visualization Data")
    latest_row_vis = filtered_vis_df.iloc[-1]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Date", latest_row_vis['Date'].date().isoformat())
    col2.metric("Current Close", f"${latest_row_vis['Close']:.2f}")
    col3.metric("High (Range)", f"${filtered_vis_df['Close'].max():.2f}")
    col4.metric("Low (Range)", f"${filtered_vis_df['Close'].min():.2f}")
    col5.metric("Current Volume", f"{int(latest_row_vis['Volume']):,}")

    st.markdown("---")
    st.header("🔮 Price Prediction")
    st.write("Prediction uses datasets from the `processed_datasets` folder.")

    prediction_symbols = get_available_symbols("processed_datasets")

    if not prediction_symbols:
        st.error("No valid datasets found in 'processed_datasets' folder for prediction.")
        st.stop()

    if selected_vis_symbol in prediction_symbols:
        selected_pred_symbol = selected_vis_symbol
    else:
        selected_pred_symbol = prediction_symbols[0]
        st.info(f"Selected visualization asset '{selected_vis_symbol}' not found in 'processed_datasets'. Defaulting prediction to '{selected_pred_symbol}'.")

    st.subheader(f"Predicting for: **{selected_pred_symbol}**")

    pred_df = load_data(selected_pred_symbol, "processed_datasets")

    if pred_df.empty:
        st.error("Selected prediction dataset is empty.")
        st.stop()

    min_date_pred = pred_df['Date'].min().to_pydatetime()
    max_date_pred = pred_df['Date'].max().to_pydatetime()

    start_date_pred_slider = max(date_range_vis[0], min_date_pred)
    end_date_pred_slider = min(date_range_vis[1], max_date_pred)
    
    if start_date_pred_slider > end_date_pred_slider:
        start_date_pred_slider = min_date_pred
        end_date_pred_slider = max_date_pred

    date_range_pred = st.slider(
        "Select Date Range for Prediction Data (applied to prediction features)",
        min_value=min_date_pred,
        max_value=max_date_pred,
        value=(start_date_pred_slider, end_date_pred_slider),
        format="YYYY-MM-DD",
        key="prediction_date_range"
    )
    
    filtered_pred_df = pred_df[(pred_df['Date'] >= date_range_pred[0]) & (pred_df['Date'] <= date_range_pred[1])]

    if filtered_pred_df.empty:
        st.warning(f"No data for prediction in the selected date range for {selected_pred_symbol}. Please adjust the slider.")
    else:
        prediction_data = predict_next_day_close(selected_pred_symbol, filtered_pred_df)
        display_prediction_results(prediction_data)

elif mode == "Stock Market News":
    st.subheader("📰 Latest Stock Market News")
    st.info("Currently displaying general news from the API. Keyword input is for reference only and does not filter the API query.")
    keyword_input = st.text_input("Enter keyword for reference only", "stock market")
    
    articles = get_stock_news()
    if not articles:
        st.info("Could not fetch news. Please check the API token or try again later.")
    else:
        for article in articles[:5]:
            st.markdown(f"### [{article.get('title', 'No Title')}]({article.get('news_url', '#')})")
            st.write(article.get("text", "No description available."))
            st.markdown(f"**Source**: {article.get('source_name', 'Unknown')} | **Date**: {article.get('date', 'N/A')}")
            st.markdown("---")