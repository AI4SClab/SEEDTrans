# SEEDTrans: Interpretable Day-ahead Photovoltaic Power Forecasting

![License](https://img.shields.io/badge/license-MIT-blue.svg)

SEEDTrans is a Transformer-based framework for **interpretable day-ahead photovoltaic (PV) power forecasting**, incorporating **multi-level series decomposition**, **expert-driven variable grouping**, and **cross-scale semantic fusion**. It achieves state-of-the-art accuracy while enhancing interpretability in extreme weather scenarios like El Niño.

---

## 🌞 Highlights

- 📊 **Interpretable Forecasting**: Integrates learnable wavelet transforms and seasonal-trend decomposition.
- 🎯 **Adaptive Grouping**: Groups meteorological variables dynamically, guided by expert priors.
- ⛅ **Extreme Weather Robustness**: Tested under El Niño conditions with adaptive representation.
- 🧠 **Transformer Core**: Encoder-decoder structure with multi-level cross-fusion and full attention.

---

## 🏗️ Model Architecture

The framework includes:

- **Adaptive Variable Grouping (AVG)**  
  Learns to identify and reweight variables critical to short-term fluctuation and long-term trend.
  
- **Wavelet-based Decomposition (WTDU)**  
  Extracts fine-grained frequency-aware features.

- **Seasonal-Trend Decomposition (STDU)**  
  Disentangles seasonal and long-term trends.

- **Cross-Fusion Strategy**  
  Fuses features across scales and decomposition levels.

![Framework](docs/images/framework.png) *(replace with your actual image)*

---

## 🧪 Experimental Results

SEEDTrans significantly outperforms baselines like ARIMA, LSTM, ConvLSTM, and iTransformer:

| Model        | RMSE ↓ | MAE ↓ | MAPE ↓ |
|--------------|--------|-------|--------|
| ARIMA        | 1.161  | 0.816 | 0.442  |
| FC-LSTM      | 1.049  | 0.842 | 0.405  |
| CNN-BiLSTM   | 0.502  | 0.271 | 0.177  |
| **SEEDTrans**| **0.439** | **0.223** | **0.129** |

*(See paper for full benchmark and ablation results)*

---

## 📁 Dataset

We use:

- 6 PV stations in Hebei, China (15-min resolution, 1 year)
- Stanford PV Plant, USA (30-min resolution, 2 years)

Each includes NWP data (irradiance, temperature, humidity, etc.) and historical PV output.

---

## 📦 Installation

```bash
git clone https://github.com/AI4SClab/SEEDTrans.git
cd SEEDTrans
pip install -r requirements.txt

# Training
python train.py --config configs/seedtrans_stanford.yaml

# Inference
python predict.py --checkpoint checkpoints/best_model.pth
