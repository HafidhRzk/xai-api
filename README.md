# 📘 Sales Performance XAI API
Explainable AI using SHAP, LIME, and Anchors

## 1. Deskripsi
Aplikasi REST API berbasis FastAPI untuk memprediksi performa sales (High/Mid/Low) dan menjelaskan hasil prediksi menggunakan metode Explainable AI (XAI): SHAP, LIME, dan Anchors.  
Aplikasi ini digunakan untuk kebutuhan eksperimen dan komparasi XAI dalam penelitian tesis.

## 2. Struktur Folder
xai-api/
├── main.py
├── train_model.py
├── sales_classifier.pkl
├── requirements.txt
└── README.md

## 3. Environment
Python versi 3.9–3.10 direkomendasikan.

### Virtual Environment
python -m venv .venv
source .venv/bin/activate

### Install Dependency
pip install -r requirements.txt

## 4. Training Model
Training dilakukan satu kali atau setiap dataset diperbarui.

python train_model.py

Model akan tersimpan sebagai sales_classifier.pkl dan otomatis menggantikan file lama.

## 5. Menjalankan API
uvicorn main:app --reload

Dokumentasi API:
http://127.0.0.1:8000/docs

## 6. Endpoint
POST /analyze

Endpoint ini melakukan:
- Prediksi performa sales
- Penjelasan menggunakan SHAP, LIME, dan Anchors

## 7. Contoh Request
{
  "salesname": "John Doe",
  "month": "January",
  "year": "2025",
  "performance": {
    "attendance": {"ontime": 20, "late": 5},
    "visit": 17,
    "productSold": 86,
    "salesValue": 79730200
  }
}

## 8. Alur Sistem
1. Data historis digunakan untuk training
2. Model dan scaler disimpan
3. Data baru diprediksi melalui API
4. Hasil dijelaskan oleh SHAP, LIME, dan Anchors

## 9. Catatan Tesis
Fokus penelitian adalah perbandingan kualitas interpretasi SHAP, LIME, dan Anchors, bukan hanya akurasi model.
