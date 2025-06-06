# Laporan Proyek Machine Learning - Geralda Livia

## Domain Proyek

Keberlanjutan telah menjadi fokus utama manajemen dan energi yang berfokus pada lingkungan, meningkatkan efisiensi, hingga pengelolaan sumber daya alam (Lemence et al., 2024, 1-2). Salah satu kontributor penting dalam upaya keberlanjutan ini adalah peramalan suhu. Peramalan suhu yang akurat, jika diterapkan secara efektif dapat memberikan kemampuan untuk memprediksi suhu yang dapat memainkan peran dalam mengoptimalkan penggunaan energi, meningkatkan efisiensi proses, dan mengurangi emisi.

Peramalan suhu adalah landasan bagi kontrol suhu yang proaktif dan efisiensi energi yang berkelanjutan. Baik dalam manajemen gedung pintar, operasional pertanian yang sensitif iklim, sistem pendingin dan pemanas di industri manufaktur, atau bahkan pengelolaan konsumsi energi di rumah tangga, kemampuan untuk memprediksi suhu yang akan datang memungkinkan perencanaan dan penyesuaian yang lebih baik. Hal memastikan bahwa proses dan sistem dapat diatur secara optimal untuk menjaga suhu yang tepat tanpa pemborosan energi yang tidak perlu, sehingga secara langsung berkontribusi pada efisiensi operasional, pengurangan biaya, dan pengurangan jejak karbon.

🔎 **Mengapa Masalah Ini Penting untuk Diselesaikan?**

- Mendukung Pengambilan Keputusan yang Lebih Baik
Prediksi harga properti yang akurat memungkinkan investor, pembeli, dan pengembang untuk membuat keputusan yang lebih tepat terkait waktu dan lokasi investasi.

Mengurangi Ketidakpastian Pasar
Dengan model prediktif, kita dapat mengurangi ketidakpastian pasar dengan menyediakan estimasi harga yang lebih andal.

Efisiensi Penilaian Properti
Model machine learning dapat mempercepat proses penilaian harga properti secara otomatis berdasarkan data historis.


**Bagaimana Masalah Ini Dapat Diselesaikan?**
Pra-pemrosesan Data: Mengubah data tanggal menjadi datetime, mengatasi nilai hilang, dan melakukan label encoding untuk data kategorikal.
Ekstraksi Fitur Time-Series: Ekstraksi tahun, bulan, dan hari dari tanggal penjualan.
Pemilihan Model: Menggunakan Random Forest dan XGBoost, serta melakukan hyperparameter tuning pada XGBoost.
Evaluasi Model: Menggunakan MAE, MSE, RMSE, dan R² sebagai metrik evaluasi.

**Referensi**:
1. Caraka, R. E., Gio, P. U., Yang, V. V., Firdaus, F. A., Riantika, R. A., Laura, G. Y., Fikriya, A., Darmawan, G., & Pontoh, R. P. S. (2024, October 30). International Conference on Computer, Control, Informatics and its Applications (IC3INA). LSTM-Driven Forecasting of Surface Temperature Trends in Indonesia as Insights Into Climate Change, 9(10), 478 - 483. https://doi.org/10.1109/IC3INA64086.2024.10732048
2. Lemence, A. L. G., Cravioto, J., & McLellan, B. C. (2024, June 25). International Conference on Green Energy and Applications (ICGEA). Review of Social Sustainability Assessments of Electricity Generating Systems, 17(6058), 223 - 228. https://doi.org/10.1109/ICGEA60749.2024.10560748

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

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
