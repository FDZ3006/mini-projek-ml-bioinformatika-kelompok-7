# Analisis Sensitivitas dan Kesalahan Klasifikasi Tanda Retinopati Diabetik Menggunakan k-Nearest Neighbors dan Decision Tree

**Mini Project Machine Learning Bioinformatika — Kelompok 7**

> Proyek ini merupakan eksperimen klasifikasi machine learning pada fitur hasil pengolahan citra retina. Proyek ini bukan alat diagnosis klinis. Hasil k-NN di bawah bersifat sementara; notebook k-NN merupakan konversi dari PDF dan belum dijalankan ulang.

## Latar belakang

Retinopati diabetik merupakan komplikasi diabetes yang dapat merusak pembuluh darah retina dan mengganggu penglihatan. Proposal Kelompok 7 mengangkat potensi analisis fitur citra fundus untuk mengenali tanda retinopati secara lebih konsisten. Dataset berisi fitur hasil ekstraksi citra retina, bukan foto fundus mentah. Proyek merencanakan perbandingan k-Nearest Neighbors (k-NN) dan Decision Tree dengan perhatian khusus pada sensitivitas kelas positif, false negative, dan interpretasi fitur.

## Tujuan

Sesuai proposal, proyek bertujuan:

1. Membangun model k-NN dan Decision Tree untuk klasifikasi tanda retinopati diabetik.
2. Membandingkan sensitivitas, false negative, dan metrik evaluasi kedua model pada data uji.
3. Menganalisis feature importance serta dampak fitur pemeriksaan awal terhadap kinerja klasifikasi.

## Dataset dan fitur

Dataset **Diabetic Retinopathy Debrecen** berasal dari [UCI Machine Learning Repository, ID 329](https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen) (DOI: [10.24432/C5XP4P](https://doi.org/10.24432/C5XP4P)). Data awal mencakup 1.151 observasi, 19 fitur prediktor, dan satu label biner. Label 0 berarti tidak ada tanda retinopati diabetik dan label 1 berarti ada tanda.

Kelompok fitur meliputi kualitas citra dan pre-screening, enam fitur microaneurysms, delapan fitur exudates, jarak makula–diskus optik, diameter diskus optik, serta keluaran AM/FM. Dataset berisi representasi numerik hasil pengolahan citra fundus, bukan citra mentah.

## Referensi

Antal, B. dan Hajdu, A. (2014). *An ensemble-based system for automatic screening of diabetic retinopathy*. Knowledge-Based Systems, 60, 20–27. [https://doi.org/10.1016/j.knosys.2013.12.023](https://doi.org/10.1016/j.knosys.2013.12.023).

## Metodologi dan alur pengerjaan

1. **EDA — Fajar:** pemeriksaan distribusi kelas, kualitas data, statistik dan sebaran fitur, serta korelasi dan interpretasi biologis awal. Lihat [notebook EDA asli](notebooks/eda/Analisis_EDA_Diabetic_Retinopathy_Kelompok_7.ipynb).
2. **Preprocessing — Isnaini:** audit data, penghapusan lima baris duplikat identik, stratified train-test split 80:20, dan StandardScaler yang di-fit pada data latih. Hasil: 1.146 observasi bersih, 916 data latih, dan 230 data uji. Lihat [notebook pipeline EDA dan preprocessing](notebooks/preprocessing/Pipeline_EDA_dan_Preprocessing_Diabetic_Retinopathy_Kelompok_7.ipynb).
3. **k-NN — Elvin:** berkas notebook yang tersedia merupakan konversi PDF ke notebook dengan Gemini. Notebook belum dijalankan ulang sehingga reproduksibilitasnya belum diverifikasi. CSV, TXT, grafik, dan PDF hasil evaluasi tersedia sebagai hasil sementara.
4. **Decision Tree — Abid:** belum ditemukan notebook maupun hasil Decision Tree di workspace saat repo ini disiapkan.
5. **Perbandingan dan dokumentasi — Rakan:** perbandingan final kedua algoritma belum tersedia karena hasil Decision Tree belum ada.

Notebook pipeline menggabungkan materi EDA dan preprocessing; notebook EDA asli juga dipertahankan sebagai pekerjaan Fajar.

## Hasil sementara k-NN

Angka berikut dicocokkan dengan CSV evaluasi dan ringkasan TXT yang tersedia. Evaluasi menggunakan test set 230 observasi.

| Skenario | k | Accuracy | Precision | Recall / sensitivitas | F1 | ROC-AUC | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 fitur | 21 | 63,04% | 68,32% | 56,56% | 61,88% | 0,7050 | 69 | 76 | 32 | 53 |
| 17 fitur | 17 | 58,26% | 63,54% | 50,00% | 55,96% | 0,6781 | 61 | 73 | 35 | 61 |

Pada test set tersebut, sensitivitas 19 fitur berarti model mengenali 69 dari 122 kasus positif (56,56%) dan melewatkan 53 kasus positif (FN; 43,44%). Skenario 17 fitur mengenali 61 dari 122 kasus positif (50,00%) dan melewatkan 61 (FN; 50,00%). Ini menunjukkan masih banyak kasus positif yang terlewat pada kedua skenario. Angka tersebut adalah hasil sementara, bukan bukti kinerja klinis.

Hasil tuning CSV menunjukkan k=21 untuk 19 fitur dan k=17 untuk 17 fitur memiliki F1 CV tertinggi pada skenario masing-masing. Pemilihan ini perlu ditegaskan dalam laporan karena nilai recall CV tertinggi terjadi pada k lain. Notebook k-NN belum dijalankan ulang; angka hasil yang disajikan adalah angka dari artefak CSV/TXT yang sudah tersedia.

Artefak evaluasi: [ringkasan k-NN](hasil/knn/ringkasan_knn.txt), [tabel evaluasi test set](hasil/knn/evaluasi_uji_knn.csv), [hasil tuning CV](hasil/knn/hasil_cv_tuning_k.csv), dan [PDF konversi beserta kode/output](hasil/knn/hasil_uji_k_NN.pdf). Grafik: [kurva CV](hasil/knn/01_kurva_cv_knn.png), [confusion matrix](hasil/knn/02_confusion_matrix_knn.png), dan [kurva ROC](hasil/knn/03_kurva_roc_knn.png).

Artefak EDA tersedia pada [tabel distribusi kelas](hasil/eda/distribusi_kelas.csv), [audit kualitas data](hasil/eda/audit_kualitas_data.csv), [statistik deskriptif](hasil/eda/statistik_deskriptif_fitur.csv), serta [grafik distribusi kelas](hasil/eda/01_distribusi_kelas.png) dan [matriks korelasi](hasil/eda/04_matriks_korelasi.png). Artefak preprocessing utama meliputi [ringkasan split](hasil/preprocessing/03_ringkasan_stratified_split.csv), [verifikasi scaling](hasil/preprocessing/04_verifikasi_standard_scaler.csv), dan tabel train/test scaled serta unscaled di [folder hasil preprocessing](hasil/preprocessing/).

## Keterbatasan dan status

- Eksperimen memakai fitur tabular dari citra yang sudah diproses, bukan citra retina mentah.
- Hasil k-NN masih sementara dan notebook hasil konversi belum direproduksi.
- Decision Tree belum tersedia; perbandingan final kedua algoritma belum dapat diklaim selesai.
- Sensitivitas k-NN yang tercatat masih terbatas dan false negative masih banyak.
- Proyek ditujukan untuk pembelajaran dan analisis, bukan diagnosis atau keputusan klinis.

## Struktur repository

```text
README.md
.gitignore
requirements.txt
notebooks/
  eda/
  preprocessing/
  knn/
hasil/
  eda/
  preprocessing/
  knn/
data/
  README.md
```

Proposal asli tersimpan di workspace lokal dan tidak dimasukkan ke repository publik karena memuat NIM anggota. Ringkasan latar belakang, tujuan, metodologi, dan pembagian tugas pada README merujuk pada proposal tersebut.

## Mendapatkan data

Dataset mentah tidak disertakan di repository. Lihat [petunjuk dataset](data/README.md) untuk sumber UCI dan cara memperolehnya. Notebook EDA mengambil dataset menggunakan ucimlrepo; notebook lain dapat menggunakan hasil preprocessing yang sudah disediakan.

## Dependensi dan menjalankan notebook

Gunakan Python dengan Jupyter Notebook/JupyterLab. Pasang dependensi yang tercantum di [requirements.txt](requirements.txt), lalu buka notebook yang diperlukan. Notebook k-NN mengandalkan CSV preprocessing; jalankan pipeline preprocessing lebih dahulu atau gunakan artefak CSV di hasil/preprocessing/. Notebook k-NN tersedia sebagai hasil konversi dan **belum diuji ulang**, sehingga petunjuk ini belum merupakan jaminan eksekusi berhasil. Notebook pipeline dapat mengambil dataset UCI saat dijalankan dan memerlukan akses jaringan.

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Notebook dapat dibuka melalui tautan pada bagian metodologi. Jalankan hanya setelah menyiapkan dependensi dan dataset sesuai petunjuk.

## Anggota dan pembagian pekerjaan

| Anggota | Tanggung jawab |
|---|---|
| Fajar Dhuha Zafran | EDA dan analisis biologis |
| Isnaini Tiana Hayu | Dataset dan preprocessing |
| Elvin Ramadhani | k-NN |
| Abid Ra'fad Aziz | Decision Tree |
| Rakan Fawwaz Irawadi | Perbandingan model, kesimpulan, dan dokumentasi |

## Progres project

- [x] EDA tersedia beserta output.
- [x] Pipeline EDA dan preprocessing tersedia beserta output.
- [x] Artefak evaluasi k-NN tersedia.
- [ ] Reproduksi notebook k-NN hasil konversi.
- [ ] Decision Tree dan evaluasinya.
- [ ] Perbandingan final kedua model serta kesimpulan.
