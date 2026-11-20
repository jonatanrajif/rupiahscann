# RupiahScanner - Deteksi Keaslian Uang Rupiah Berbasis YOLO

Aplikasi web untuk mendeteksi keaslian uang Rupiah menggunakan model **YOLO (You Only Look Once)** yang dilatih khusus untuk mengenali fitur keamanan pada uang kertas. Pengguna dapat mengambil foto melalui kamera atau mengunggah gambar, kemudian sistem akan menganalisis dan menampilkan tingkat keaslian secara real-time.

## Fitur

- **Deteksi Real-time** dengan kamera perangkat (laptop/HP)
- **Upload Gambar** untuk analisis dari galeri
- **Hasil Analisis** berupa persentase keaslian dan potensi palsu
- **Filter Jumlah Bounding Box** untuk memastikan fitur uang terdeteksi lengkap
- **Tampilan Responsif** dengan Tailwind CSS
- **Dukungan Multi-kamera** (pilih kamera yang tersedia)
- **Feedback Visual** dengan progres bar dan ikon status

## Teknologi yang Digunakan

### Backend

- **Flask** – Framework web Python
- **Ultralytics YOLO** – Model deteksi objek
- **OpenCV** – Pemrosesan gambar
- **TensorFlow** – Backend untuk model TFLite
- **Flask-CORS** – CORS support

### Frontend

- **HTML5 + Tailwind CSS** – Tampilan modern dan responsif
- **Font Awesome** – Ikon
- **Vanilla JavaScript** – Interaksi kamera dan komunikasi API

## Struktur Folder

```
.
├── app.py                 # Aplikasi Flask utama
├── index.html             # Antarmuka pengguna
├── requirements.txt       # Dependensi Python
├── Model/
│   └── best_int8.tflite   # Model YOLO terlatih (format TFLite)
├── README.md              # Dokumentasi
└── (file debug)           # gambar_dari_web.jpg, yolo_hasil_deteksi.jpg
```

## Instalasi

### Prasyarat

- Python 3.8 atau lebih baru
- Pip (package manager)

### Langkah-langkah

1. **Clone repositori**

   ```bash
   git clone https://github.com/username/rupiahscanner.git
   cd rupiahscanner
   ```

2. **Buat virtual environment (opsional tetapi disarankan)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # atau
   venv\Scripts\activate     # Windows
   ```

3. **Install dependensi**

   ```bash
   pip install -r requirements.txt
   ```

4. **Siapkan model**
   - Letakkan file model `best_int8.tflite` di folder `Model/`.
   - Jika menggunakan model `.pt`, pastikan nama file `best.pt` dan sesuaikan kode di `app.py`.

## Cara Menjalankan

1. Jalankan server Flask:

   ```bash
   python app.py
   ```

2. Buka browser dan akses:

   ```
   http://localhost:5001
   ```

3. Aplikasi siap digunakan. Klik **"Mulai Kamera"** untuk mengakses kamera, atau **"Upload File"** untuk memilih gambar dari perangkat.

## Penggunaan

1. **Pilih kamera** (jika ada lebih dari satu) dari dropdown.
2. **Tekan "Mulai Kamera"** – tampilan langsung akan muncul.
3. **Arahkan uang kertas** ke kamera dengan pencahayaan cukup dan pastikan seluruh uang terlihat.
4. **Tekan "Ambil Foto"** – sistem akan memproses gambar dan menampilkan:
   - **Status** (ASLI / PALSU / TIDAK DIKENALI)
   - **Tingkat Keaslian** dan **Potensi Palsu** dalam bentuk persentase.
5. Alternatif: **Upload File** untuk menganalisis gambar yang sudah ada.

## Catatan untuk Pengembang

- Model dilatih dengan `imgsz=640`. Pastikan preprocessing di `app.py` menggunakan ukuran yang sama (sudah diatur di `model.predict(imgsz=640)`).
- Parameter `conf=0.3` dan `iou=0.7` dapat disesuaikan di `proses_inferensi` untuk mengubah sensitivitas deteksi.
- Logika filter jumlah bounding box (5 untuk sisi depan, 4 untuk sisi belakang) dapat dimodifikasi sesuai kebutuhan.

## Kontribusi

Pull request diterima. Untuk perubahan besar, buka isu terlebih dahulu untuk mendiskusikan apa yang ingin diubah.

---

**Dibuat untuk Tugas Akhir – Deteksi Keaslian Uang Rupiah**
