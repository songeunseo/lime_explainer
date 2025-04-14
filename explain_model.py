import tensorflow as tf
from tensorflow.keras.models import load_model
from lime import lime_image
import numpy as np
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries, felzenszwalb
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 모델 로드
model = load_model('cats_vs_dogs_model.h5')
# LIME 설명기 초기화
explainer = lime_image.LimeImageExplainer()

# 예측 함수
def predict_fn(images):
    # 이미지 전처리 (컬러 정보 유지)
    if len(images.shape) == 3:  # 단일 이미지인 경우
        images = np.expand_dims(images, axis=0)
    images = images.astype('float32') / 255.0
    # 모델 예측
    return model.predict(images)

# 테스트 이미지 선택
def select_test_image(animal_type=None):
    # 테스트 디렉토리에서 랜덤으로 이미지 선택
    test_dir = 'data/dogs_vs_cats/test'
    import random
    import os
    
    # animal_type이 지정되지 않은 경우에만 랜덤 선택
    if animal_type is None:
        animal = random.choice(['cats', 'dogs'])
    else:
        animal = animal_type
    
    img_path = os.path.join(test_dir, animal, random.choice(os.listdir(os.path.join(test_dir, animal))))
    
    # 이미지 로드 및 전처리 (컬러 유지)
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    
    # 실제 라벨 설정 (0: 고양이, 1: 개)
    true_label = 0 if animal == 'cats' else 1
    
    return img_array, true_label

# 테스트 이미지 선택 (개 이미지로 고정)
image, true_label = select_test_image('dogs')

# 모델 예측
pred = model.predict(np.expand_dims(image, axis=0))
pred_label = 1 if pred[0][0] > 0.5 else 0
confidence = pred[0][0] if pred_label == 1 else 1 - pred[0][0]

print(f"True Label: {'Dog' if true_label == 1 else 'Cat'}")
print(f"Predicted: {'Dog' if pred_label == 1 else 'Cat'}")
print(f"Confidence: {confidence:.2f}")

# LIME 설명 생성
explanation = explainer.explain_instance(
    image,
    predict_fn,
    top_labels=2,
    hide_color=0,
    num_samples=2000,
    random_seed=42
)

# 디버깅을 위한 정보 출력
print("Explanation top labels:", explanation.top_labels)
print("Explanation local_exp:", explanation.local_exp)

# 설명 시각화
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=False,
    num_features=20,
    hide_rest=False,
    min_weight=0.01
)

# 마스크 정보 출력
print("Mask shape:", mask.shape)
print("Mask min value:", np.min(mask))
print("Mask max value:", np.max(mask))
print("Mask unique values:", np.unique(mask))

# 결과 시각화
plt.figure(figsize=(15, 5))

# 1. 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(image.astype('uint8'))
plt.title(f'Original Image (True: {"Dog" if true_label == 1 else "Cat"})')
plt.axis('off')

# 2. LIME 히트맵
plt.subplot(1, 3, 2)
if np.any(mask):  # 마스크가 비어있지 않은 경우에만 표시
    plt.imshow(mask, cmap='RdBu_r', interpolation='bilinear')
    plt.colorbar()
else:
    plt.text(0.5, 0.5, 'No importance map generated', 
             horizontalalignment='center', verticalalignment='center')
plt.title('LIME Importance Heatmap')
plt.axis('off')

# 3. 오버레이
plt.subplot(1, 3, 3)
plt.imshow(image.astype('uint8'))
if np.any(mask):  # 마스크가 비어있지 않은 경우에만 오버레이
    plt.imshow(mask, cmap='RdBu_r', alpha=0.6)
plt.title(f'Overlay (Predicted: {"Dog" if pred_label == 1 else "Cat"})')
plt.axis('off')

plt.tight_layout()
plt.savefig('lime_explanation.png', dpi=300, bbox_inches='tight')
plt.show()