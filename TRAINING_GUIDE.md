# Eğitim Rehberi — M4 Air Üzerinde Uçtan Uca

Bu rehber, genişletilmiş veriyle modeli MacBook Air M4 (24 GB) üzerinde
eğitmek, kalibre etmek ve CoreML'e çıkarmak için gereken her adımı sırayla
verir. Veri indirme ve temizleme adımları hazırlık sırasında bir kez
koşulmuştur; yeniden koşmak güvenlidir (indirilenler önbelleklenir).

## 0. Ortam

```bash
cd "/Users/keremoztopuz/Desktop/Masaüstü - Mac/senior_design_project_ai_model"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sistem Python'u (3.9) da çalışır; venv sürüm çakışmalarını önler.

## 1. Veri (hazırlıkta koşuldu; yeniden üretmek istersen)

```bash
python data_prep/fetch_scin.py          # SCIN: Acne + Eczema (kimlik gerekmez)
python data_prep/fetch_kaggle_sets.py   # DermNet takviyesi (~/.kaggle/kaggle.json ister)
python data_prep/clean_and_split.py     # dedup + sızıntısız 70/15/15 split
```

`clean_and_split.py` eski `orchestration_data/`'yı
`orchestration_data_v1_backup/`a taşır ve sınıf/split sayım tablosu basar.
Sonunda `audit_split_leakage()` koşar — `TEMIZ` görmelisin.

## 2. Eğitim

```bash
cd src
python train.py
```

- İlk 5 epoch yalnızca sınıflandırma kafası eğitilir, sonra son ConvNeXt
  aşaması açılır (kod bunu kendisi yapar).
- En iyi model val AUROC ile `outputs/model/best_model.pth`e, en iyi Top-1
  ayrıca `best_top1_model.pth`e kaydedilir.
- Epoch metrikleri `outputs/logs/metrics_history.csv`ye yazılır — eğitimi
  izlemek için bu dosyaya bak.
- Erken durdurma: 8 epoch iyileşme yoksa durur.

Öneri: kapağı açık, şarjda ve gece koş. Air fansız olduğu için uzun yükte
kısılma (throttling) normaldir; sonucu etkilemez, sadece süreyi uzatır.

## 3. Kalibrasyon

```bash
python calibrate.py
```

- Sınıf başına sigmoid eşikleri `outputs/model/thresholds.json`a,
- Ölçülmüş **temperature** değeri `outputs/model/temperature.json`a yazılır.

Uygulamadaki `CameraViewModel.processedScore` şu an elle seçilmiş 1.8
kullanıyor; buradaki ölçülmüş değer app entegrasyonunda onun yerini almalı.

## 4. Değerlendirme

```bash
python evaluate.py --tta
```

`--tta` yatay çevirme ortalaması alır (bedava birkaç puan). Grafikler
`outputs/images/` altına düşer.

## 5. CoreML Export

```bash
cd ../export
python export.py               # FP16, ~53 MB
python export.py --palettize   # 8-bit, ~27 MB (önce FP16 ile doğrula)
```

Export sonunda PyTorch ve CoreML logitleri karşılaştırılır; `max |Δ|`
0.15'in altında olmalı. Çıktı: `outputs/coreml/skin_disease.mlpackage`.
Uygulama sözleşmesi değişmedi: aynı 384×384 girdi, aynı normalizasyon,
aynı 4 logit — Swift tarafında değişiklik gerekmez (sınıf sırası:
Acne, Eczema, Eye_Bags, Wrinkles).

## M4 Air 24 GB Fizibilite

**Sonuç: rahatça eğitilir.**

| Kaynak | İhtiyaç | Mevcut |
|---|---|---|
| Model + gradyan + AdamW durumu (FP32) | ~0.5 GB | 24 GB birleşik |
| Aktivasyonlar (batch 16 @ 384²) | ~4–6 GB | " |
| Veri önbelleği + işletim sistemi | ~4 GB | " |

Bellek toplamda ~10 GB'ın altında kalır; batch 32'ye çıkmak bile mümkün
(`config.py`'de `BATCH_SIZE`). MPS desteği kodda hazır, AMP Apple Silicon'da
otomatik kapalı (FP32 — doğru davranış).

- Ölçülen adım süresi (bu makinede, MPS, batch 16 @384²): **1.7 s/batch ≈ 9.4 görüntü/sn**
  (son ConvNeXt aşaması + kafa eğitilirken; MPS bellek kullanımı ~0.4 GB)
- Tahmini epoch süresi (genişletilmiş ~960 train görüntüsüyle): **~2 dk**
- Erken durdurmalı tam koşu: **~1 saat** (50 epoch tavanında ~1.7 saat;
  fansız kasada kısılmayla en kötü ~2.5 saat)

Optuna taraması (30 deneme) bu makinede ~gece ölçeğidir; önce tek reçeteyle
eğit, gerekirse sonra tara.

## Veri Kaynakları ve Lisans Notu

| Kaynak | Sınıflar | Lisans |
|---|---|---|
| SCIN (Google/Stanford) | Acne, Eczema | CC-BY 4.0 (ticari OK) |
| DermNet (Kaggle mirror) | Acne, Eczema | Belirsiz / non-commercial eğilimli |
| Mevcut Roboflow havuzu | Eye_Bags, Wrinkles | Roboflow Universe (sete göre değişir) |

DermNet ve eski havuz ticari dağıtım için gri alandadır; App Store'da satış
büyüdüğünde eğitim setini SCIN + lisansı doğrulanmış kaynaklarla yeniden
kurmak en temizi.
