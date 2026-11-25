# Dataset

Project menggunakan **Diabetic Retinopathy Debrecen** dari [UCI Machine Learning Repository, dataset ID 329](https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen). DOI dataset: [10.24432/C5XP4P](https://doi.org/10.24432/C5XP4P).

Dataset mentah tidak disalin ke repository. Unduh melalui halaman resmi UCI sesuai ketentuan dan sitasi yang berlaku. Notebook EDA mengambil data secara programatis menggunakan library ucimlrepo dan ID 329; penggunaan cara tersebut memerlukan koneksi jaringan saat notebook dijalankan.

Data berisi 1.151 baris, 19 fitur prediktor, dan label biner (0 = tidak ada tanda retinopati diabetik; 1 = ada tanda). Fitur merupakan hasil pengolahan citra fundus retina, bukan foto fundus mentah. File train/test hasil preprocessing yang disimpan pada ../hasil/preprocessing/ adalah artefak olahan untuk dokumentasi dan reproduksi model, bukan salinan dataset mentah.
