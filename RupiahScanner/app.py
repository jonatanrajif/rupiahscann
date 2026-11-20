from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2
import numpy as np
import base64
import os
import json
import uuid
import zipfile
import io
from ultralytics import YOLO

app = Flask(__name__, instance_relative_config=False)  # nonaktifkan instance folder
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "detections.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- MODEL TABEL ----------
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_filename = db.Column(db.String(200), nullable=False)
    result_class = db.Column(db.String(20))
    confidence_asli = db.Column(db.Float)
    confidence_palsu = db.Column(db.Float)
    bboxes_json = db.Column(db.Text)
    is_verified = db.Column(db.Boolean, default=False)
    verified_class = db.Column(db.String(20))
    verified_confidence = db.Column(db.Float)

    def __repr__(self):
        return f'<Detection {self.id} - {self.result_class}>'

# ---------- MIGRASI OTOMATIS (DIPERBAIKI) ----------
with app.app_context():
    db.create_all()  # buat tabel jika belum ada
    import sqlite3
    conn = sqlite3.connect('detections.db')
    cursor = conn.cursor()
    # Cek apakah tabel detection ada
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detection'")
    if cursor.fetchone():
        cursor.execute("PRAGMA table_info(detection)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        new_columns = {
            'bboxes_json': 'TEXT',
            'is_verified': 'BOOLEAN DEFAULT 0',
            'verified_class': 'VARCHAR(20)',
            'verified_confidence': 'FLOAT'
        }
        for col, col_type in new_columns.items():
            if col not in existing_columns:
                cursor.execute(f"ALTER TABLE detection ADD COLUMN {col} {col_type}")
                print(f"[MIGRASI] Kolom '{col}' ditambahkan.")
    else:
        print("[MIGRASI] Tabel 'detection' belum ada, dilewati.")
    conn.commit()
    conn.close()

# ---------- KONFIGURASI YOLO ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "best_int8.tflite")
model = YOLO(MODEL_PATH, task="detect")

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------- FUNGSI INFERENSI ----------
def proses_inferensi(image_array):
    print("\n--- GERBANG VERIFIKASI MUTLAK JUMLAH BOUNDING BOX ---")
    results = model.predict(source=image_array, conf=0.5, imgsz=640, verbose=False)
    boxes = results[0].boxes
    total_box_terdeteksi = len(boxes)
    print(f"[DEBUG LOG] Jumlah Bounding Box: {total_box_terdeteksi}")

    if total_box_terdeteksi == 0:
        print(">> Tidak ada fitur uang yang terdeteksi.")
        return "tidak_dikenali", 0.0, []

    # gambar_luaran_yolo = results[0].plot()
    # cv2.imwrite("yolo_hasil_deteksi.jpg", gambar_luaran_yolo)

    best_box_index = boxes.conf.argmax().item()
    kelas_id = int(boxes.cls[best_box_index].item())
    confidence = boxes.conf[best_box_index].item()
    nama_kelas_utama = results[0].names[kelas_id]

    is_sisi_depan = False
    is_sisi_belakang = False
    if "depan" in nama_kelas_utama.lower():
        is_sisi_depan = True
    elif "belakang" in nama_kelas_utama.lower():
        is_sisi_belakang = True
    else:
        if total_box_terdeteksi >= 5:
            is_sisi_depan = True
        else:
            is_sisi_belakang = True

    if is_sisi_depan:
        print(f"[DEPAN] Target: 5 Box | Terdeteksi: {total_box_terdeteksi}")
        if total_box_terdeteksi < 5:
            print(">> DITOLAK: Box kurang dari 5.")
            return "tidak_dikenali", 0.0, []
    elif is_sisi_belakang:
        print(f"[BELAKANG] Target: 4 Box | Terdeteksi: {total_box_terdeteksi}")
        if total_box_terdeteksi < 4:
            print(">> DITOLAK: Box kurang dari 4.")
            return "tidak_dikenali", 0.0, []

    print(f">> DITERIMA: '{nama_kelas_utama}'")

    bboxes = []
    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            bboxes.append([x1, y1, x2, y2, cls_id, conf])

    return nama_kelas_utama, confidence, bboxes

# ---------- FUNGSI SIMPAN KE DATABASE ----------
def simpan_ke_database(gambar_cv, hasil_keaslian, persen_asli, persen_palsu, bboxes):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    filename = f"{timestamp}_{uid}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(filepath, gambar_cv)

    bboxes_json = json.dumps(bboxes) if bboxes else None

    detection = Detection(
        image_filename=filename,
        result_class=hasil_keaslian,
        confidence_asli=persen_asli,
        confidence_palsu=persen_palsu,
        bboxes_json=bboxes_json,
        is_verified=False
    )
    db.session.add(detection)
    db.session.commit()
    print(f"[DATABASE] Data tersimpan: {filename} | {hasil_keaslian}")

# ---------- ROUTE UTAMA ----------
@app.route('/')
def home():
    return send_file('index.html')

# ---------- ROUTE DETEKSI ----------
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({"status": "error", "message": "Tidak ada gambar yang dikirim"}), 400

        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # cv2.imwrite("gambar_dari_web.jpg", img)

        nama_kelas, confidence, bboxes = proses_inferensi(img)

        if nama_kelas == "tidak_dikenali":
            simpan_ke_database(img, "TIDAK_DIKENALI", 0.0, 0.0, bboxes)
            return jsonify({
                "status": "success",
                "keaslian": "SIAP",
                "asli": 0,
                "palsu": 0
            })

        status_keaslian = nama_kelas.upper()
        conf_percent = confidence * 100

        if status_keaslian == "ASLI":
            persen_asli = conf_percent
            persen_palsu = 100.0 - conf_percent
        else:
            persen_palsu = conf_percent
            persen_asli = 100.0 - conf_percent

        simpan_ke_database(img, status_keaslian, persen_asli, persen_palsu, bboxes)

        return jsonify({
            "status": "success",
            "keaslian": status_keaslian,
            "asli": round(persen_asli, 1),
            "palsu": round(persen_palsu, 1)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------- HISTORY (JSON) ----------
@app.route('/history')
def history():
    detections = Detection.query.order_by(Detection.timestamp.desc()).all()
    result = []
    for d in detections:
        result.append({
            'id': d.id,
            'timestamp': d.timestamp.isoformat(),
            'image_filename': d.image_filename,
            'result_class': d.result_class,
            'confidence_asli': d.confidence_asli,
            'confidence_palsu': d.confidence_palsu,
            'bboxes': json.loads(d.bboxes_json) if d.bboxes_json else [],
            'is_verified': d.is_verified,
            'verified_class': d.verified_class
        })
    return jsonify(result)

# ---------- ADMIN (WEB) ----------
@app.route('/admin')
def admin():
    detections = Detection.query.order_by(Detection.timestamp.desc()).all()
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Admin - Data Deteksi</title>
    <style>
        body { font-family: Arial; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .asli { color: green; font-weight: bold; }
        .palsu { color: red; font-weight: bold; }
        .unknown { color: gray; }
        .btn { padding: 4px 8px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn-verify { background: #28a745; }
        .btn-unverify { background: #dc3545; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { opacity:0.8; }
    </style>
    <script>
        function verify(id, correctClass) {
            fetch('/verify/' + id, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ verified_class: correctClass })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') location.reload();
                else alert('Gagal: ' + data.message);
            });
        }
        function deleteRecord(id) {
            if (confirm('Yakin ingin menghapus data ini? Tindakan tidak dapat dibatalkan.')) {
                fetch('/delete/' + id, { method: 'DELETE' })
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'success') { alert('Terhapus!'); location.reload(); }
                    else alert('Gagal: ' + data.message);
                })
                .catch(err => alert('Error: ' + err));
            }
        }
        function exportDataset() {
            window.location.href = '/export';
        }
    </script>
    </head>
    <body>
    <h2>📊 Riwayat Deteksi & Verifikasi</h2>
    <button onclick="exportDataset()" style="margin-bottom:10px; padding:8px 16px; background: #17a2b8; color:white; border:none; border-radius:4px; cursor:pointer;">⬇️ Ekspor Dataset (YOLO)</button>
    <table>
        <tr><th>ID</th><th>Waktu</th><th>Gambar</th><th>Hasil Prediksi</th><th>% Asli</th><th>% Palsu</th><th>Verifikasi</th><th>Aksi</th></tr>
    """
    for d in detections:
        cls_class = "asli" if d.result_class == "ASLI" else "palsu" if d.result_class == "PALSU" else "unknown"
        verified_text = d.verified_class if d.is_verified else "Belum"
        action_buttons = ""
        if not d.is_verified:
            action_buttons = f"""
                <button class="btn btn-verify" onclick="verify({d.id}, 'ASLI')">Verif Asli</button>
                <button class="btn btn-verify" onclick="verify({d.id}, 'PALSU')">Verif Palsu</button>
                <button class="btn btn-verify" onclick="verify({d.id}, 'TIDAK_DIKENALI')">Verif Tidak</button>
            """
        else:
            action_buttons = f"""
                <button class='btn btn-unverify' onclick="verify({d.id}, null)">Batalkan</button>
                <button class='btn btn-danger' onclick="deleteRecord({d.id})">🗑️ Hapus</button>
            """
        img_link = f"/uploads/{d.image_filename}"
        html += f"""
        <tr>
            <td>{d.id}</td>
            <td>{d.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
            <td><a href="{img_link}" target="_blank">{d.image_filename}</a></td>
            <td class="{cls_class}">{d.result_class}</td>
            <td>{d.confidence_asli:.1f}%</td>
            <td>{d.confidence_palsu:.1f}%</td>
            <td>{verified_text}</td>
            <td>{action_buttons}</td>
        </tr>
        """
    html += "</table></body></html>"
    return html

# ---------- VERIFIKASI ----------
@app.route('/verify/<int:detection_id>', methods=['POST'])
def verify_detection(detection_id):
    try:
        data = request.json
        verified_class = data.get('verified_class')
        detection = Detection.query.get(detection_id)
        if not detection:
            return jsonify({"status": "error", "message": "ID tidak ditemukan"}), 404

        if verified_class is None:
            detection.is_verified = False
            detection.verified_class = None
        else:
            detection.is_verified = True
            detection.verified_class = verified_class

        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------- HAPUS DATA ----------
@app.route('/delete/<int:detection_id>', methods=['DELETE'])
def delete_detection(detection_id):
    try:
        detection = Detection.query.get(detection_id)
        if not detection:
            return jsonify({"status": "error", "message": "ID tidak ditemukan"}), 404

        filepath = os.path.join(UPLOAD_FOLDER, detection.image_filename)
        if os.path.exists(filepath):
            os.remove(filepath)

        db.session.delete(detection)
        db.session.commit()
        return jsonify({"status": "success", "message": "Data berhasil dihapus"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------- EKSPOR DATASET (YOLO) ----------
@app.route('/export')
def export_dataset():
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        detections = Detection.query.filter(Detection.is_verified == True).all()
        for det in detections:
            img_path = os.path.join(UPLOAD_FOLDER, det.image_filename)
            if not os.path.exists(img_path):
                continue
            zf.write(img_path, arcname=f"images/{det.image_filename}")

            if det.bboxes_json:
                bboxes = json.loads(det.bboxes_json)
                label_lines = []
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    continue
                h, w = img_cv.shape[:2]
                for box in bboxes:
                    x1, y1, x2, y2, cls_id, conf = box
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    if det.is_verified and det.verified_class:
                        class_mapping = {"ASLI":0, "PALSU":1, "TIDAK_DIKENALI":2}
                        cls_id = class_mapping.get(det.verified_class, cls_id)
                    label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                label_filename = os.path.splitext(det.image_filename)[0] + ".txt"
                zf.writestr(f"labels/{label_filename}", "\n".join(label_lines))

    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='dataset_yolo.zip')

# ---------- SERVING FILE UPLOAD ----------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ---------- MAIN ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)