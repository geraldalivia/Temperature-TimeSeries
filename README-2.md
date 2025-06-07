# Laporan Proyek Machine Learning - Naufal Arsapradhana

## Domain Proyek

Prediksi harga properti berbasis time-series merupakan tantangan penting dalam sektor real estat, termasuk di Indonesia. Dengan memanfaatkan data historis seperti tanggal penjualan, kode pos, tipe properti, dan jumlah kamar tidur, kita dapat membangun model prediktif yang membantu berbagai pemangku kepentingan dalam pengambilan keputusan.

🧩 **Mengapa Masalah Ini Penting untuk Diselesaikan?**

- **Mendukung Pengambilan Keputusan yang Lebih Baik**  
  Prediksi harga properti yang akurat memungkinkan investor, pembeli, dan pengembang untuk membuat keputusan yang lebih tepat terkait waktu dan lokasi investasi.

- **Mengurangi Ketidakpastian Pasar**  
  Dengan model prediktif, kita dapat mengurangi ketidakpastian pasar dengan menyediakan estimasi harga yang lebih andal.

- **Efisiensi Penilaian Properti**  
  Model machine learning dapat mempercepat proses penilaian harga properti secara otomatis berdasarkan data historis.

🔧 **Bagaimana Masalah Ini Dapat Diselesaikan?**

- **Pra-pemrosesan Data**: Mengubah data tanggal menjadi datetime, mengatasi nilai hilang, dan melakukan label encoding untuk data kategorikal.
- **Ekstraksi Fitur Time-Series**: Ekstraksi tahun, bulan, dan hari dari tanggal penjualan.
- **Pemilihan Model**: Menggunakan Random Forest dan XGBoost, serta melakukan hyperparameter tuning pada XGBoost.
- **Evaluasi Model**: Menggunakan MAE, MSE, RMSE, dan R² sebagai metrik evaluasi.

📚 **Referensi Terkait**:

- Zhao, Sijian. "Applying Machine Learning and Time Series to Predict Real Estate Valuations." Atlantis Press, 2024.
- Tyagi, Varun. "House Price Prediction with Machine Learning." Medium, 2023.
- Zhang, Cheng et al. "Deep Learning Models for Price Forecasting of Financial Time Series: A Review." arXiv:2305.04811, 2023.

---

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi harga properti berdasarkan data historis yang tersedia?
2. Model machine learning mana yang memberikan akurasi terbaik dalam memprediksi harga properti?

### Goals

1. Membangun model prediksi harga properti berdasarkan fitur historis.
2. Membandingkan performa Random Forest dan XGBoost untuk mendapatkan model terbaik.

### Solution Statements

- Menggunakan dua model machine learning: Random Forest dan XGBoost.
- Melakukan analisa pada hasil pengukuran metrik MAE, MSE, RMSE, dan R² untuk mengevaluasi performa model.

---

## Data Understanding

Dataset yang digunakan berisi informasi harga properti di Australia. Dataset ini bersifat time-series dan memiliki fitur-fitur berikut:

- `datesold` : tanggal penjualan properti
- `postcode` : kode pos lokasi properti
- `price` : harga penjualan properti (target variabel)
- `propertyType` : tipe properti (unit, house)
- `bedrooms` : jumlah kamar tidur

EDA (Exploratory Data Analysis) telah dilakukan untuk melihat beberapa informasi berikut: 
- Data yang hilang dan terduplikasi, 
- Distribusi harga properti, 
- Distribusi jumlah kamar tidur, 
- Perbandingan harga berdasarkan jenis properti, 
- Kode pos dengan properti harga tertinggi.

Dataset ini terdiri atas 29.580 baris dan 5 kolom.

Tautan sumber dataset: [House Property Sales Time Series](https://www.kaggle.com/datasets/htagholdings/property-sales?select=raw_sales.csv)

---

## Data Preparation

- Mengonversi kolom `datesold` menjadi datetime dan memecahnya menjadi kolom `year`, `month`, dan `day` karena model machine learning tidak bisa menangani objek tipe tanggal secara langsung, jadi konversi ini penting agar informasi waktu dapat dimanfaatkan sebagai fitur numerik oleh model.
- Label encoding untuk fitur kategorikal seperti `propertyType` karena model tree-based seperti Random Forest dan XGBoost tidak membutuhkan representasi vektor yang rumit (seperti one-hot encoding). Mereka mampu menangani hubungan ordinal maupun non-ordinal hanya dengan label encoding, serta tetap dapat membagi data berdasarkan nilai kategori ini dalam pembuatan pohon keputusan.
- Membersihkan outlier dengan menerapan Interquartile Method karena outlier dapat memengaruhi performa model dengan memperbesar error (terutama pada metrik MSE dan RMSE). Dengan menggunakan metode IQR, kita mendeteksi dan menghapus nilai-nilai ekstrem berdasarkan distribusi data, khususnya untuk variabel numerik seperti harga (price) sehingga model menjadi lebih stabil dan general.
- Mendefinisikan kolom fitur dan target, di mana variabel fitur akan membawa pengaruh dalam menentukan variabel target, yaitu `price`.
- Split data dengan proporsi training dan test set (80:20) umum digunakan karena memberikan cukup banyak data untuk pelatihan, sambil menyisakan cukup data untuk evaluasi model secara akurat.
- Feature scaling tidak diperlukan karena model tree-based tidak sensitif terhadap skala.

---

## Modeling

### Perbandingan Model: Random Forest vs XGBoost

Dalam proyek ini, dua model tree-based digunakan untuk melakukan prediksi harga properti: **Random Forest** dan **XGBoost**. Keduanya memiliki keunggulan dalam menangani data tabular dan fitur kategorikal yang telah diproses dengan baik. Namun, masing-masing memiliki karakteristik unik yang memengaruhi performa dan penggunaannya.

### 🔍 Random Forest
Model ini digunakan untuk memprediksi nilai target (seperti harga properti, misalnya) berdasarkan fitur-fitur input menggunakan pendekatan Random Forest yang kuat dan andal terhadap overfitting serta bekerja baik pada data nonlinear. Beberapa parameter yang digunakan, yaitu:
- n_estimators=100 berarti model membangun 100 pohon keputusan (semakin banyak pohon, biasanya hasil lebih stabil dan akurat).
- random_state=42 digunakan untuk memastikan hasil yang konsisten setiap kali kode dijalankan (reproducibility).
- fit(X_train, y_train) melatih model dengan data pelatihan (X_train sebagai fitur dan y_train sebagai target).
- predict(X_test) menghasilkan prediksi (rf_pred) dari data uji X_test.

#### ✅ Kelebihan:
- **Tahan terhadap overfitting**: Karena menggabungkan banyak pohon secara acak dan mengandalkan voting rata-rata.
- **Mudah digunakan**: Cukup efektif dengan tuning minimal.
- **Tidak sensitif terhadap skala fitur dan outlier**.
- **Memberikan feature importance** yang dapat digunakan untuk interpretasi.

#### ❌ Kekurangan:
- **Kurang efisien** jika jumlah pohon sangat besar, terutama pada dataset besar.
- **Kurang optimal** dalam menangkap hubungan kompleks antar fitur jika dibandingkan dengan XGBoost.

### ⚙️ XGBoost (Extreme Gradient Boosting)
Model ini adalah algoritma boosting yang sangat efektif dalam menangani data tabular dan sering memberikan performa lebih tinggi dibandingkan model lain dalam berbagai kompetisi machine learning seperti di Kaggle. Cocok untuk masalah regresi dengan data yang kompleks dan nonlinier. Beberapa parameter yang digunakan, yaitu:
- n_estimators=100 berarti model akan membangun 100 pohon secara bertahap, di mana setiap pohon berusaha memperbaiki kesalahan dari pohon sebelumnya.
- learning_rate=0.1 menentukan seberapa besar kontribusi tiap pohon baru terhadap model akhir. Nilai kecil membuat proses pembelajaran lebih stabil.
- random_state=42 menjamin hasil yang konsisten saat dijalankan berulang.
- fit(X_train, y_train) melatih model pada data pelatihan.
- predict(X_test) digunakan untuk menghasilkan prediksi dari data uji.

#### ✅ Kelebihan:
- **Akurasi tinggi**: Telah memenangkan banyak kompetisi data science.
- **Efisien dan cepat**: Menggunakan teknik boosting yang dioptimalkan.
- **Kemampuan regularisasi (L1 dan L2)** untuk mengontrol overfitting lebih baik.
- **Fleksibel dalam tuning** dan menangani data tidak seimbang.

#### ❌ Kekurangan:
- **Lebih kompleks dan banyak parameter** yang harus dituning untuk performa optimal.
- **Cenderung lebih sensitif terhadap data kotor** seperti missing value dan encoding yang buruk.
- **Sulit diinterpretasikan** dibandingkan Random Forest.

### 📊 Ringkasan Perbandingan

| Aspek                          | Random Forest                              | XGBoost                                        |
|-------------------------------|--------------------------------------------|------------------------------------------------|
| Akurasi                       | Cukup tinggi                               | Sangat tinggi                                  |
| Overfitting                   | Rendah                                     | Dapat dikontrol dengan regularisasi            |
| Interpretabilitas             | Lebih mudah                                | Cenderung sulit (black-box)                    |
| Waktu pelatihan               | Lebih cepat untuk tuning sederhana         | Optimasi cepat, tetapi tuning bisa lebih lama  |
| Kompleksitas model            | Lebih sederhana                            | Lebih kompleks                                 |
| Sensitivitas terhadap data    | Tahan terhadap outlier & data mentah       | Perlu data bersih & pra-pemrosesan yang baik   |
| Penggunaan di industri        | Luas dan sebagai baseline                  | Umum pada kompetisi & aplikasi presisi tinggi  |

### 📌 Kesimpulan

Jika kita mencari **model yang cepat, stabil, dan mudah digunakan**, **Random Forest** adalah pilihan yang sangat baik, terutama untuk baseline model. Namun, jika kita membutuhkan **akurası tinggi dan bersedia melakukan tuning lebih dalam**, maka **XGBoost** adalah pilihan terbaik untuk performa prediktif maksimal.

---

## Evaluation
Proses evaluasi dilakukan dengan menjalankan fungsi evaluate_model() dengan menggunakan metrik evaluasi berikut:
- **MAE (Mean Absolute Error)**: rata-rata kesalahan absolut prediksi terhadap nilai sebenarnya.
- **MSE (Mean Squared Error)**: rata-rata kuadrat kesalahan.
- **RMSE (Root Mean Squared Error)**: akar dari MSE.
- **R² (R-squared)**: proporsi variansi target yang dapat dijelaskan oleh fitur.

### Hasil Evaluasi
Berikut adalah tabel hasil evaluasi dari model Random Forest dan XGBoost yang diukur dalam 4 metrik evaluasi:
| Model          | MAE (Rp)     | MSE (Rp²)          | RMSE (Rp)   | R² Score |
|----------------|--------------|---------------------|-------------|----------|
| Random Forest  | 75,750.01    | 10,479,933,946.53   | 102,371.55  | 0.6685   |
| XGBoost        | 68,153.09    | 8,614,280,786.20    | 92,813.15   | 0.7275   |

Hasil evaluasi dari kedua model divisualisasikan dalam bentuk bar chart untuk setiap metrik, sehingga memudahkan dalam melihat model mana yang memberikan performa terbaik secara keseluruhan. Selain itu, dengan membandingkan sebaran titik pada kedua plot menggunakan scatter plot, kita bisa menilai seberapa baik masing-masing model mampu memprediksi harga properti: semakin rapat titik-titik terhadap garis, semakin baik akurasi model tersebut.

---

## Kesimpulan
Berdasarkan evaluasi di atas dapat disimpulkan bahwa solusi yang diterapkan dengan menggunakan dua model machine learning yang berbada serta melakukan analisa pada hasil pengukran metrik evaluasi terbukti mampu menjawab pertanyaan yang ada dalam section Business Understanding, yaitu:

1. Bagaimana cara memprediksi harga properti berdasarkan data historis yang tersedia?
    Prediksi harga dilakukan dengan mendefinisikan fitur yang akan menjadi dasar perhitungan variabel target, yaitu harga properti. Selanjutnya, model dilatih pada data yang telah melalui tahap data preparation untuk dapat mengenali pola yang ada. Pola inilah yang menjadi dasar perhitungan prediksi harga properti berdasarkan kriteria yang ada, seperti jumlah kamar tidur, tipe properti, sampai wilayah properti berada.
2. Model machine learning mana yang memberikan akurasi terbaik dalam memprediksi harga properti?
    Berdasarjan hasil evaluasi, model XGBoost memberikan akurasi yang lebih baik jika dibandingkan dengan Random Forest Regressor, di mana model ini memberikan nilai MAE dan RMSE terkecil serta R² tertinggi.

Selain itu, hasil evaluasi ini juga telah mampu mencapai Goals yang ditetapkan untuk membangun model prediksi harga properti berdasarkan fitur historis dan membandingkan performa Random Forest dan XGBoost untuk mendapatkan model terbaik.

---

## Visualisasi

Dua visualisasi scatter plot menunjukkan akurasi masing-masing model:

![Scatter Plot Image](scatter_plot.png)

Titik yang mendekati garis merah diagonal menunjukkan prediksi yang akurat.

---

## Inferensi
Pada proses inferensi, saya ingin mengukur seberapa relevan dan akurat sebuah model dalam memberikan prediksi berdasarkan data yang dibuat seperti berikut:
| Postcode | Bedrooms | Year | Month | Day   | Property Type |
|----------|----------|------|-------|-------|---------------|
| 2615     | 3        | 2025 | 5     | 13    | House         |
| 2607     | 2        | 2025 | 2     | 11    | Unit          |

Dari data di atas, harga yang diprediksi adalah berikut:
| Model          | Harga Data 1   | Harga Data 2 | 
|----------------|----------------|--------------|
| Random Forest  | $ 537,254.33   | $ 306,946.25 |
| XGBoost        | $ 527,184.31   | $ 333,614.84 |

---

_Terima kasih telah membaca laporan proyek ini._