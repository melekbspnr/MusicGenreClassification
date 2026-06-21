# Music Genre Classification

GTZAN veri kümesini ve ek bir Turkish Rap sınıfını kullanarak müzik türü sınıflandırma yaklaşımlarını karşılaştıran Google Colab tabanlı bir makine öğrenmesi çalışmasıdır.

## Notebook'lar

- `music_genre_classification.ipynb`: Veri indirme, ses dönüştürme, özellik çıkarımı, model eğitimi ve değerlendirme akışını içerir.
- `agentic_career_assistant.ipynb`: Proje çıktılarını örnek girdi olarak kullanan, Gemini destekli çok ajanlı bir kariyer asistanı denemesidir.

## Kullanılan yöntemler

- MFCC, Mel Spectrogram, Chroma, Zero Crossing Rate ve Spectral Centroid
- Random Forest, SVM, CNN, Gradient Boosting, KNN ve Voting Classifier
- Confusion matrix ve sınıflandırma raporları
- GTZAN veri kümesine özel Turkish Rap örnekleri ekleme

## Çalıştırma

Notebook'lar Google Colab dosya yollarını kullanır. Colab üzerinde sırayla çalıştırılması önerilir.

1. Kaggle hesabınızdan bir API token dosyası (`kaggle.json`) oluşturun.
2. `music_genre_classification.ipynb` notebook'unu açın.
3. Notebook istediğinde `kaggle.json` dosyasını geçici Colab oturumuna yükleyin.
4. Hücreleri sırayla çalıştırın.

LLM notebook'u için API anahtarını notebook içine yazmayın. Google Colab'ın **Secrets** bölümüne `GEMINI_API_KEY` adıyla ekleyip çalışma ortamına aktarın:

```python
import os
from google.colab import userdata

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```

Paylaşılan veya sürüm kontrolüne alınan notebook'larda anahtar değerini saklamayın.

## Yerel bağımlılıklar

Notebook'ları yerel Jupyter ortamında çalıştırmak isterseniz:

```bash
pip install -r requirements.txt
```

Colab'a özgü Drive bağlama ve dosya yükleme hücrelerinin yerel ortama uyarlanması gerekir.

## Veri ve model dosyaları

Veri kümeleri, ses dosyaları, çıkarılmış özellikler ve eğitilmiş modeller boyutları ve lisans koşulları nedeniyle repoya dahil edilmez. Bunlar notebook çalıştırıldığında kullanıcının Google Drive alanında oluşturulur.

## Güvenlik

Bu repoda API anahtarı veya `kaggle.json` bulunmamalıdır. Daha önce notebook içinde kullanılan anahtarlar iptal edilip yenilenmelidir.
