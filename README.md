# Laporan Proyek Machine Learning - Geralda Livia

## Domain Proyek

Keberlanjutan telah menjadi fokus utama manajemen dan energi yang berfokus pada lingkungan dalam upaya meningkatkan efisiensi, hingga pengelolaan sumber daya alam (Lemence et al., 2024, 1-2). Salah satu kontributor penting dalam upaya keberlanjutan ini adalah peramalan suhu. Peramalan suhu yang akurat, jika diterapkan secara efektif dapat memberikan kemampuan untuk memprediksi suhu yang dapat memainkan peran dalam mengoptimalkan penggunaan energi, meningkatkan efisiensi proses, dan mengurangi emisi.

Peramalan suhu berbasis deret waktu adalah landasan bagi kontrol suhu dan efisiensi energi yang berkelanjutan. Baik dalam manajemen gedung pintar, sistem pendingin dan pemanas di industri manufaktur, atau bahkan pengelolaan konsumsi energi di rumah tangga, kemampuan untuk memprediksi suhu yang akan datang memungkinkan perencanaan dan penyesuaian yang lebih baik. Dengan memanfaatkan data historis suhu ruangan, kita dapat membangun model prediktif yang membantu berbagai stakeholder dalam pengambilan keputusan yang lebih cerdas dan proaktif. Hal ini memastikan bahwa proses dan sistem dapat diatur secara optimal untuk menjaga suhu ruangan yang tepat tanpa pemborosan energi yang tidak perlu, sehingga secara langsung berkontribusi pada efisiensi operasional, pengurangan biaya, dan pengurangan jejak karbon. 


🔎 **Mengapa Masalah Ini Penting untuk Diselesaikan?**

- Mendukung Pengambilan Keputusan yang Lebih Baik: Prediksi suhu ruangan yang akurat memungkinkan untuk optimasi energi di berbagai sektor misalnya optimalisasi produksi dan distribusi listrik. Selain itu, dapat memberikan informasi lebih dini tentang suhu ruangan kedepan yang berguna untuk mitigasi dini pada hal-hal diluar kendali kita serta membuat keputusan yang lebih tepat dan efisien.

- Mengurangi Ketidakpastian Operasional: Dengan model prediktif, kita dapat mengurangi ketidakpastian terkait kondisi iklim, menyediakan estimasi suhu yang lebih andal untuk perencanaan dan mitigasi risiko.

- Efisiensi Penggunaan Energi: Kontrol suhu adalah landasan efisiensi energi. Model machine learning dapat mempercepat proses penilaian dan penyesuaian suhu secara otomatis berdasarkan data historis, memastikan proses berjalan tanpa pemborosan energi yang tidak perlu, sehingga berkontribusi pada efisiensi operasional dan pengurangan jejak karbon.


🔖 **Bagaimana Masalah Ini Dapat Diselesaikan?**

- Pra-pemrosesan Data: Mengonversi data waktu ke Datetime dan melakukan normalisasi data suhu.

- Pembuatan Sequence Time-Series: Mengubah data deret waktu menjadi pasangan input (urutan suhu masa lalu) dan output (suhu yang akan datang) menggunakan teknik sliding windows.

- Pemilihan Model: Menggunakan model Deep Learning (LSTM) dan model Ensemble Learning (XGBoost), serta melakukan training pada keduanya (Caraka et al., 2024).

- Evaluasi Model: Menggunakan MAE, MSE, dan RMSE sebagai metrik evaluasi untuk membandingkan kinerja kedua model.

**Referensi**:
1. Caraka, R. E., Gio, P. U., Yang, V. V., Firdaus, F. A., Riantika, R. A., Laura, G. Y., Fikriya, A., Darmawan, G., & Pontoh, R. P. S. (2024, October 30). International Conference on Computer, Control, Informatics and its Applications (IC3INA). LSTM-Driven Forecasting of Surface Temperature Trends in Indonesia as Insights Into Climate Change, 9(10), 478 - 483. https://doi.org/10.1109/IC3INA64086.2024.10732048
2. Lemence, A. L. G., Cravioto, J., & McLellan, B. C. (2024, June 25). International Conference on Green Energy and Applications (ICGEA). Review of Social Sustainability Assessments of Electricity Generating Systems, 17(6058), 223 - 228. https://doi.org/10.1109/ICGEA60749.2024.10560748

---

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

---

## Data Understanding

Dataset dihasilkan dengan bantuan data Perangkat IOT yang mewakili nilai suhu udara ruangan terhadap waktu. Dataset ini bersifat time-series dan memiliki fitur sebagai berikut:

Fitur:
- `Unamed: 0` : hanya index nomor saja.
- `Hourly _Temp` : berisi nilai rata-rata Suhu Udara Pasokan dalam derajat celcius per jam,
- `Datetime` : menunjukkan tanggal dan Jam perekaman data

EDA (Exploratory Data Analysis) telah dilakukan untuk melihat beberapa informasi berikut:
- Pengecekan nilai yang hilang.
- Pengecekan nilai duplikat.
- Distribusi suhu

Dataset ini terdiri atas 7056 baris dan 3 kolom. Dengan tidak adanya nilai yang hilang (no missing value) dan tidak ada data yang terduplikasi (no duplicated value)

Sumber dataset : [Time Series Room Temperature Data](https://www.kaggle.com/datasets/vitthalmadane/ts-temp-1)

---

## Data Preparation

- Mengubah nama kolom pada dataset. Kolom yang diubah adalah `Unnamed: 0` menjadi `Number`, kolom `Datetime` menjadi `Timestamp`, dan kolom `Hourly_Temp` menjadi `Temperature`.
- Mengonversi kolom `Timestamp` menjadi tipe data datetime untuk memudahkan analisis dan menampilkan visualisasi tren suhu seiring waktu untuk mengidentifikasi pola atau anomali
- Melakukan normalisasi data pada Fitur target `Temperature` menggunakan MinMaxScaler ke rentang (-1, 1). Hal ini dilakukan untuk membangun model deep learning seperti LSTM yang berguna agar pelatihan lebih stabil dan cepat konvergen.
- Membuat Sequence Sliding Windows yang tujuannya adalah data diubah menjadi format sequence (input X, target Y) menggunakan teknik sliding windows dengan seq_length (panjang urutan historis) 6, yang memungkinkan model belajar dependensi temporal.
- Pembagian Data menjadi set pelatihan (training set) dan set pengujian (testing set) dengan proporsi 80:20 (test_size=0.2), tanpa pengacakan (shuffle=False) untuk mempertahankan urutan temporal data karena hal ini krusial untuk peramalan deret waktu.
- Penyesuaian bentuk data untuk XGBoost. Karena XGBoost mengharapkan input 2D (sampel, fitur) sementara LSTM membutuhkan 3D (sampel, timesteps, fitur), data X_train dan X_test di-reshape dari `(X_train.shape[0], seq_length, 1)` menjadi `(X_train, seq_length)` dan `(X_test.shape[0],seq_length, 1)` menjadi `(X_test, seq_length)`. Selainn itu, pada variabel target y_train dan y_test juga di-flatten dari `(X_train, 1)` menjadi `(X_train, )` untuk `y_train`, sedangkan `(X_test, 1)` menjadi `(X_test, )` untuk `y_test` .

---

## Modeling

### Perbandingan Model: LSTM vs XGBoost

Dalam proyek ini, dua model machine learning digunakan untuk melakukan prediksi suhu: **Long Short-Term Memory (LSTM)** dan **XGBoost**. Keduanya memiliki pendekatan yang berbeda dalam menangani deret waktu, dan perbandingan performa mereka akan memberikan wawasan berharga.

### 📌 LSTM (Long Short-Term Memory)

LSTM adalah jenis Recurrent Neural Network (RNN) yang dirancang khusus untuk mengatasi masalah vanishing/exploding gradient (gradien menghilang) dan menangani dependensi jangka panjang dalam data sekuensial. Unit dasar LSTM memiliki tiga gerbang (gates) utama dan sebuah status sel (cell state), yang secara kolektif mengatur informasi yang masuk, keluar, dan disimpan dalam sel. Ini memungkinkan LSTM untuk mengingat informasi penting dan melupakan informasi yang tidak relevan selama periode waktu yang panjang. Dalam projek ini, model dibangun untuk prediksi deret waktu, yang ingin diprediksi adalah nilai suhu berikutnya berdasarkan data sebelumnya. 

Quick Workflow of LSTM
> Pada proyek prediksi suhu, setiap kali LSTM memproses satu timestep dalam sequence, LSTM akan:
1. Forget Gate akan menentukan berapa banyak memori suhu lama yang relevan yang harus dipertahankan.
2. Input Gate memutuskan berapa banyak informasi suhu baru dari timestep saat ini yang harus ditambahkan ke memori.
3. State Cell diperbarui berdasarkan keputusan gerbang lupa dan input.
4. Output Gate menggunakan status sel yang diperbarui untuk menghasilkan hidden state baru, yang kemudian akan menjadi input untuk timestep berikutnya dan juga dapat diteruskan ke lapisan Dense (lapisan keluaran) untuk menghasilkan prediksi suhu aktual.

```
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(units=1))
```

Beberapa catatan penting terkait code di atas yaitu:
- Sequential() : Membuat model sekuensial yang lapisannya ditambahkan satu per satu secara berurutan.
- model.add(LSTM(...)) : Menambahkan lapisan LSTM (Long Short-Term Memory) ke model.
- units=50 : Jumlah neuron/unit di LSTM layer adalah 50.
- activation='relu' : Menggunakan fungsi aktivasi ReLU
- seq_length : Jumlah langkah waktu (timesteps) yang digunakan untuk memprediksi nilai selanjutnya.
- 1 : Setiap timestep hanya berisi satu suhu.
- model.add(Dense(...)) : Menambahkan lapisan Dense (fully connected) sebagai output layer.
- units=1 : Model hanya akan memprediksi satu nilai suhu pada timestamp berikutnya.

**Kelebihan:**
- Mampu menangkap dependensi jangka panjang: Ideal untuk data deret waktu yang panjang dan kompleks.
- Belajar pola temporal secara otomatis: Tidak memerlukan rekayasa fitur input secara eksplisit.
- Fleksibel dalam menangani berbagai jenis pola deret waktu.

**Kekurangan:**
- Membutuhkan lebih banyak data: Cenderung berkinerja optimal dengan dataset yang besar.
- Waktu pelatihan lebih lama: Terutama pada dataset yang besar dan arsitektur yang dalam.
- Kurang interpretatif: Sulit untuk memahami "mengapa" model membuat prediksi tertentu.
- Daya komputasi tinggi: Membutuhkan GPU untuk pelatihan yang efisien pada model yang lebih besar.


### 📌 XGBoost (Extreme Gradient Boosting)

XGBoost adalah algoritma boosting berbasis pohon keputusan yang sangat efisien dan kuat. XGBoost bekerja berdasarkan kerangka kerja Gradient Boosting. Model ini bekerja dengan membangun serangkaian pohon keputusan secara sekuensial, di mana setiap pohon baru belajar untuk memperbaiki kesalahan (residu) yang dibuat oleh kombinasi semua pohon sebelumnya. Proses "belajar dari kesalahan" ini, dikombinasikan dengan teknik regularisasi yang canggih, memungkinkan XGBoost untuk mencapai akurasi prediksi yang sangat tinggi dan kinerja yang efisien, menjadikannya pilihan populer untuk berbagai masalah regresi dan klasifikasi pada data tabular. Untuk peramalan deret waktu, XGBoost memerlukan fitur lagged atau temporal yang direkayasa secara eksplisit sebagai input.

Quick Workflow of XGBoost
> Dalam projek prediksi suhu ini, perlu melakukan rekayasa fitur lagged/input, misalnya, seq_length suhu sebelumnya menjadi input model.
1. Setiap pohon keputusan di XGBoost akan belajar hubungan antara fitur-fitur lagged yang didapat dari input model sebelumnya. Misalnya suhu 1 jam yang lalu, 2 jam yang lalu, dan seterusnya, serta kesalahan prediksi suhu.
2. Pohon pertama mungkin membuat prediksi suhu yang belum baik atau kasar.
3. Pohon kedua akan belajar untuk memperbaiki di mana pohon pertama yang salah.
4. Pohon ketiga akan memperbaiki kesalahan gabungan dari dua pohon pertama, dan seterusnya.
5. Setiap pohon baru akan belajar untuk mengoreksi kesalahan dari model sebelumnya. Hingga akhirnya memiliki prediksi yang lebih baik karena pohon baru telah mencoba untuk mengurangi kesalahan dari prediksi sebelumnya.
   
```
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_xgb, y_train_xgb)
```

Beberapa catatan penting terkait code di atas yaitu:
- XGBRegressor(): library xgboost.
- n_estimators=100: hyperparameter yang menentukan jumlah pohon keputusan (decision trees) yang akan dibangun oleh algoritma boosting.
- learning_rate=0.1: hyperparameter yang mengontrol ukuran langkah saat mengoptimalkan model.
- random_state=42: generator angka acak.
- n_jobs=-1: hyperparameter yang mengatur jumlah thread CPU yang akan digunakan untuk pelatihan.

**Kelebihan:**
- Akurasi tinggi: kinerja prediktifnya yang kuat.
- Efisien dan cepat: Dioptimalkan untuk kinerja dan kecepatan pelatihan.
- Kemampuan regularisasi: Memiliki mekanisme bawaan untuk mengontrol overfitting.
- Fleksibel: Dapat menangani berbagai jenis data tabular dan masalah.

**Kekurangan:**
- Memerlukan rekayasa fitur manual: Agar efektif untuk deret waktu, perlu setup secara eksplisit sehingga membuat fitur seperti nilai lagged atau rata-rata bergerak (dalam proyek ini, seq_length data menjadi fitur untuk XGBoost).
- Lebih sensitif terhadap data kotor: Kualitas data input sangat mempengaruhi kinerja.
- Sulit diinterpretasikan: Mirip dengan model ensemble lainnya, sulit untuk melihat kontribusi setiap pohon secara individual.


### Ringkasan Perbandingan

| Aspek                         | Random Forest                              | XGBoost                                        |
|-------------------------------|--------------------------------------------|------------------------------------------------|
| Kemampuan Temporal            | Belajar dependensi otomatis, cocok         | Memerlukan rekayasa fitur (lagged/Input)       |
|                               | untuk panjang                              | Sangat tinggi                                  |
| Akurasi                       | Sangat baik pada pola kompleks             | Sangat baik pada berbagai pola,                |
|                               |                                            | ering kompetitif                               |
| Waktu Pelatihan               | Cenderung lebih lama                       | Optimasi cepat                                 |
| Membutuhkan Data              | Lebih banyak data untuk kinerja optimal    | Cukup baik pada dataset menengah               |
| Interpretabilitas             | Rendah (black-box)                         | Sedang (fitur penting dapat dianalisis)        |         
| Sensitivitas terhadap Skala   | Sensitif (memerlukan normalisasi)          | Tidak sensitif (tetapi normalisasi tetap baik) |

### 📝 Kesimpulan

Jika kita mencari model yang secara otomatis dapat menangkap dependensi temporal yang kompleks dan jangka panjang, **LSTM** adalah pilihan yang kuat, terutama pada dataset deret waktu yang besar. Namun, jika kita membutuhkan model yang cepat, efisien, dan memberikan akurasi tinggi dengan rekayasa fitur yang moderat, maka **XGBoost** adalah alternatif yang sangat kompetitif. Pilihan model terbaik seringkali bergantung pada karakteristik spesifik data, kompleksitas pola, dan kebutuhan akan kecepatan atau interpretasi.

---

## Evaluation

Proses evaluasi dilakukan pada data pengujian (X_test, y_test) menggunakan metrik evaluasi berikut:
-**MAE (Mean Absolute Error)**: Rata-rata kesalahan absolut antara prediksi dan nilai sebenarnya.
-**MSE (Mean Squared Error)**: Rata-rata kuadrat kesalahan antara prediksi dan nilai sebenarnya.
-**RMSE (Root Mean Squared Error)**: Akar kuadrat dari MSE. Satuannya sama dengan unit target.

### Hasil Evaluasi
Berikut adalah tabel hasil evaluasi dari model LSTM dan XGBoost yang diukur dalam 3 metrik evaluasi pada skala normalisasi dan skala asli data:
| Metrik                     | LSTM           | XGBoost          | 
|----------------------------|----------------|------------------|
| MSE (Normalized)           | 0.0142         | 0.0135           | 
| RMSE (Normalized)          | 0.1193         | 0.1161           |
| MAE (Normalized)           | 0.0508         | 0.0552           |
| MSE (Original Scale)       | 3.4536         | 3.2711           |
| RMSE (Original Scale)      | 1.8584         | 1.8086           |
| MAE (Original Scale)       | 0.7910         | 0.8597           |

Hasil evaluasi dari kedua model juga divisualisasikan dalam bentuk grafik garis yang membandingkan prediksi masing-masing model dengan nilai aktual pada data pengujian, sehingga memudahkan dalam melihat model mana yang memberikan performa terbaik secara visual.

---

## Kesimpulan

Berdasarkan Evaluasi sebelumnya dapat disimpulkan bahwa dengan dua model machine learning yang berbeda, kedua model machine learning tersebut dapat memberikan solusi yang didukung dengan analisa pada hasil pengukuran metrik evaluasi yang terbukti mampu menjawab pertanyaan yang ada dalam bagian Business Understanding, yaitu:

1. Bagaimana cara memprediksi suhu berikutnya berdasarkan data suhu historis yang tersedia?
> Prediksi suhu dilakukan dengan terlebih dahulu menyiapkan data historis ke dalam format sequence (urutan) yang relevan. Selanjutnya, model (LSTM atau XGBoost) dilatih pada data yang telah melalui tahap pra-pemrosesan untuk dapat mengenali pola temporal dan tren yang ada. Pola inilah yang menjadi dasar perhitungan prediksi suhu berikutnya berdasarkan riwayat suhu sebelumnya.

2. Model machine learning mana yang memberikan akurasi terbaik dalam memprediksi suhu?
> Berdasarkan hasil evaluasi pada metrik MSE dan RMSE (baik pada skala normalisasi maupun skala asli), model XGBoost memberikan akurasi yang sedikit lebih baik dibandingkan dengan model LSTM, menunjukkan kesalahan rata-rata kuadrat yang lebih kecil. Namun, perlu dicatat bahwa perbedaan kinerja antara kedua model tidak terlalu signifikan, dan keduanya mampu menangkap pola suhu dengan baik.

✏️ Selain itu, hasil evaluasi ini juga telah mampu mencapai Goals yang ditetapkan untuk membangun model prediksi suhu berdasarkan fitur historis dan membandingkan performa LSTM dan XGBoost untuk mendapatkan model terbaik.

---

## Visualisasi

Dua visualisasi grafik garis yang membandingkan prediksi masing-masing model dengan nilai aktual pada data pengujian.

![Perbandingan Prediksi Kedua Model](https://drive.google.com/uc?export=view&id=1TYBdeTWhHtjCs78FFe4dQavTghoKrgMd)

Dapat dilihat bahwa kedua model mampu menangkap pola suhu ruangan dengan baik.

---

## Inferensi

Pada proses inferensi, model yang telah dilatih digunakan untuk memprediksi suhu berikutnya berdasarkan input data historis. Berikut adalah contoh hasil inferensi untuk memprediksi suhu berikutnya, baik dari data terakhir yang tersedia maupun dari input manual:

Berikut data input manual: <br>
`Data suhu : [25.0, 24.5, 24.0, 23.5, 23.0, 22.5]`

Pada Data Terakhir itu didapatkan dari `seq_length = 6`, panjang sequence input, yaitu 6. Maksudnya data suhu terakhir yang tersedia dari seluruh dataset asli.

Berikut adalah tabel hasil inferensi kedua model

| Model          |    Data Terakhir    |   Input Manual   | 
|----------------|---------------------|------------------|
| LSTM           |       22.83 °C      |     21.94 °C     |
| XGBoost        |       23.10 °C      |        -         |

Catatan: untuk inferensi input manual hanya menggunakan model LSTM saja

---

_Terima kasih telah membaca laporan proyek ini._
