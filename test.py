import cv2
import numpy as np
import matplotlib.pyplot as plt

# Canny kenar tespiti fonksiyonu
def canny_edge_detection(img, low_threshold=100, high_threshold=200):
    """Canny kenar tespiti uygular."""
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges

# Kutu sayma yöntemi ile fraktal boyut hesaplama fonksiyonu
def box_count(img, box_size):
    """Verilen bir görüntü için kutu sayma yöntemini uygular."""
    img_size = img.shape[0]  # Görüntünün boyutu (kare olarak varsayıyoruz)
    num_boxes = img_size // box_size  # Kutu sayısını hesapla
    count = 0
    
    for i in range(num_boxes):
        for j in range(num_boxes):
            if np.any(img[i * box_size:(i + 1) * box_size, j * box_size:(j + 1) * box_size] > 0):
                count += 1
    
    return count

def fractal_dimension(img, min_box_size=1):
    """Görüntünün fraktal boyutunu kutu sayma yöntemiyle hesaplar."""
    sizes = []
    counts = []
    
    img_size = img.shape[0]
    box_sizes = np.unique(np.logspace(0, np.log2(img_size), base=2, num=10).astype(int))
    box_sizes = box_sizes[box_sizes >= min_box_size]
    
    for box_size in box_sizes:
        count = box_count(img, box_size)
        sizes.append(box_size)
        counts.append(count)
    
    # Log-log ölçeğinde regresyon yap
    log_sizes = np.log(1.0 / np.array(sizes))
    log_counts = np.log(np.array(counts))
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    
    return -coeffs[0]  # Fraktal boyut eğimin negatifidir

# Görüntüyü yükle ve gri tonlamaya çevir
img_path = 'python/PHOTO-2025-02-20-17-56-09.jpg'  # Gerçek resim yolunu buraya yazın
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Canny kenar tespiti uygula
edges = canny_edge_detection(img)

# Fraktal boyutunu hesapla
dim = fractal_dimension(edges)

# Sonucu yazdır
print(f"Fraktal Boyutu: {dim:.4f}")

# Görüntüyü ve kenarları görselleştir
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Orijinal Görüntü")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title(f"Canny Kenar Tespiti\nFraktal Boyut: {dim:.4f}")
plt.axis("off")

plt.show()

