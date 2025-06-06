# Laporan Proyek Machine Learning - Geralda Livia

## Domain Proyek

Keberlanjutan telah menjadi fokus utama manajemen dan energi yang berfokus pada lingkungan, meningkatkan efisiensi, hingga pengelolaan sumber daya alam (Lemence et al., 2024, 1-2). Salah satu kontributor penting dalam upaya keberlanjutan ini adalah peramalan suhu. Peramalan suhu yang akurat, jika diterapkan secara efektif dapat memberikan kemampuan untuk memprediksi suhu yang dapat memainkan peran dalam mengoptimalkan penggunaan energi, meningkatkan efisiensi proses, dan mengurangi emisi.

Peramalan suhu berbasis deret waktu adalah landasan bagi kontrol suhu dan efisiensi energi yang berkelanjutan. Baik dalam manajemen gedung pintar, operasional pertanian yang sensitif iklim, sistem pendingin dan pemanas di industri manufaktur, atau bahkan pengelolaan konsumsi energi di rumah tangga, kemampuan untuk memprediksi suhu yang akan datang memungkinkan perencanaan dan penyesuaian yang lebih baik. Dengan memanfaatkan data historis suhu, kita dapat membangun model prediktif yang membantu berbagai stakeholder dalam pengambilan keputusan yang lebih cerdas dan proaktif. Hal ini memastikan bahwa proses dan sistem dapat diatur secara optimal untuk menjaga suhu yang tepat tanpa pemborosan energi yang tidak perlu, sehingga secara langsung berkontribusi pada efisiensi operasional, pengurangan biaya, dan pengurangan jejak karbon. 


🔎 **Mengapa Masalah Ini Penting untuk Diselesaikan?**

- Mendukung Pengambilan Keputusan yang Lebih Baik: Prediksi suhu yang akurat memungkinkan sektor seperti pertanian (jadwal tanam/panen), energi (optimalisasi produksi dan distribusi listrik), dan kesehatan (prediksi gelombang panas/dingin) untuk membuat keputusan yang lebih tepat dan efisien.

- Mengurangi Ketidakpastian Operasional: Dengan model prediktif, kita dapat mengurangi ketidakpastian terkait kondisi iklim, menyediakan estimasi suhu yang lebih andal untuk perencanaan dan mitigasi risiko.

- Efisiensi Penggunaan Energi: Kontrol suhu adalah landasan efisiensi energi. Model machine learning dapat mempercepat proses penilaian dan penyesuaian suhu secara otomatis berdasarkan data historis, memastikan proses berjalan tanpa pemborosan energi yang tidak perlu, sehingga berkontribusi pada efisiensi operasional dan pengurangan jejak karbon.


**Bagaimana Masalah Ini Dapat Diselesaikan?**

- Pra-pemrosesan Data: Mengonversi data waktu ke Datetime dan melakukan normalisasi data suhu.

- Pembuatan Sequence Time-Series: Mengubah data deret waktu menjadi pasangan input (urutan suhu masa lalu) dan output (suhu yang akan datang) menggunakan teknik sliding windows.

- Pemilihan Model: Menggunakan model Deep Learning (LSTM) dan model Ensemble Learning (XGBoost), serta melakukan training pada keduanya (Caraka et al., 2024).

- Evaluasi Model: Menggunakan MAE, MSE, dan RMSE sebagai metrik evaluasi untuk membandingkan kinerja kedua model.

**Referensi**:
1. Caraka, R. E., Gio, P. U., Yang, V. V., Firdaus, F. A., Riantika, R. A., Laura, G. Y., Fikriya, A., Darmawan, G., & Pontoh, R. P. S. (2024, October 30). International Conference on Computer, Control, Informatics and its Applications (IC3INA). LSTM-Driven Forecasting of Surface Temperature Trends in Indonesia as Insights Into Climate Change, 9(10), 478 - 483. https://doi.org/10.1109/IC3INA64086.2024.10732048
2. Lemence, A. L. G., Cravioto, J., & McLellan, B. C. (2024, June 25). International Conference on Green Energy and Applications (ICGEA). Review of Social Sustainability Assessments of Electricity Generating Systems, 17(6058), 223 - 228. https://doi.org/10.1109/ICGEA60749.2024.10560748


## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi suhu berikutnya berdasarkan data suhu historis yang tersedia?
2. Model machine learning mana (LSTM atau XGBoost) yang memberikan akurasi terbaik dalam memprediksi suhu?

### Goals

1. Membangun model prediksi suhu berdasarkan fitur historis dan pola temporal.
2. Membandingkan performa model LSTM dan XGBoost untuk mendapatkan model terbaik dalam peramalan suhu.

### Solution statements

1. Menggunakan dua model machine learning: Long Short-Term Memory (LSTM) dan XGBoost Regressor.
2. Melakukan analisa pada hasil pengukuran metrik MAE, MSE, dan RMSE untuk mengevaluasi performa model.


## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).


Dataset yang digunakan berisi informasi suhu per jam. Dataset ini bersifat time-series dan memiliki fitur utama berikut:

Timestamp: waktu pengukuran suhu.

Hourly_Temp: nilai suhu per jam (target variabel).

EDA (Exploratory Data Analysis) telah dilakukan untuk melihat beberapa informasi berikut:

Struktur data dan tipe kolom.

Statistik deskriptif variabel suhu.

Pengecekan nilai yang hilang.

Visualisasi tren suhu seiring waktu untuk mengidentifikasi pola atau anomali.

Dataset ini, dalam contoh ini, terdiri atas 100 baris dan 2 kolom (jika menggunakan data dummy yang dibuat).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
