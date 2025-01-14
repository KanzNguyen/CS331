import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from torch import nn
import joblib
from skimage import data, color
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy import ndimage
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Hàm xử lý hình ảnh
def process_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 23, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    img_cnt = img_rgb.copy()
    img_cnt = cv2.drawContours(img_cnt, cnts, -1, (0, 255, 255), 4)

    l, r, t, b = [], [], [], []
    for contour in cnts:
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
        l.append(leftmost)
        r.append(rightmost)
        t.append(topmost)
        b.append(bottommost)

    leftmost = min(l, key=lambda x: x[0])
    rightmost = max(r, key=lambda x: x[0])
    topmost = min(t, key=lambda x: x[1])
    bottommost = max(b, key=lambda x: x[1])

    img_pnt = img_cnt.copy()
    img_pnt = cv2.circle(img_pnt, leftmost, 5, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, rightmost, 5, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, topmost, 5, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, bottommost, 5, (255, 255, 0), -1)

    ADD_PIXELS = 0
    new_img = img_rgb[topmost[1]-ADD_PIXELS:bottommost[1]+ADD_PIXELS,
                      leftmost[0]-ADD_PIXELS:rightmost[0]+ADD_PIXELS].copy()
    resize_img = cv2.resize(new_img, (224, 224))

    return img_rgb, gray, thresh, img_cnt, img_pnt, new_img, resize_img

# Hàm tiền xử lý ảnh cho MobileNet và PCA
def preprocess_image(image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.fromarray(image)
    img = transform(img).unsqueeze(0).to(device)
    return img

def preprocess_image_pca(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.fromarray(image)
    img = transform(img).numpy().flatten()
    return img

# Load pre-trained MobileNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mobilenet_v3_large_model = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3_large_model = nn.Sequential(*list(mobilenet_v3_large_model.children())[:-1])
mobilenet_v3_large_model = mobilenet_v3_large_model.to(device)
mobilenet_v3_large_model.eval()

resnet18_model = models.resnet18(pretrained=True)
resnet18_model = nn.Sequential(*list(resnet18_model.children())[:-1])
resnet18_model = resnet18_model.to(device)
resnet18_model.eval()

# Load other models and PCA

# resnet with pca
model1_resnet_pca = joblib.load('resnet 18/model1.pkl')
model2_resnet_pca = joblib.load('resnet 18/model2.pkl')
model3_resnet_pca = joblib.load('resnet 18/model3.pkl')
ensemble_model_svm_resnet_pca = joblib.load('resnet 18/ensemble_model_svm.pkl')
pca_model_resnet_pca = joblib.load('resnet 18/pca_model.pkl')
svm_model_resnet_pca = joblib.load('resnet 18/svm.pkl')
scaler_model_resnet_pca = joblib.load('resnet 18/scaler.pkl')
# resnet no pca
model1_resnet = joblib.load('resnet 18 no pca/model1.pkl')
model2_resnet = joblib.load('resnet 18 no pca/model2.pkl')
model3_resnet = joblib.load('resnet 18 no pca/model3.pkl')
ensemble_model_svm_resnet = joblib.load('resnet 18 no pca/ensemble_model_svm.pkl')
pca_model_resnet = joblib.load('resnet 18 no pca/pca_model.pkl')
svm_model_resnet = joblib.load('resnet 18 no pca/svm.pkl')
scaler_model_resnet = joblib.load('resnet 18 no pca/scaler.pkl')
#mobilenet with pca
model1_mobilenet_pca = joblib.load('mobilenet V3/model1.pkl')
model2_mobilenet_pca = joblib.load('mobilenet V3/model2.pkl')
model3_mobilenet_pca = joblib.load('mobilenet V3/model3.pkl')
ensemble_model_svm_mobilenet_pca = joblib.load('mobilenet V3/ensemble_model_svm.pkl')
pca_model_mobilenet_pca = joblib.load('mobilenet V3/pca_model.pkl')
svm_model_mobilenet_pca = joblib.load('mobilenet V3/svm.pkl')
scaler_model_mobilenet_pca = joblib.load('mobilenet V3/scaler.pkl')
#mobilenet no pca
model1_mobilenet = joblib.load('mobilenet V3 no pca/model1.pkl')
model2_mobilenet = joblib.load('mobilenet V3 no pca/model2.pkl')
model3_mobilenet = joblib.load('mobilenet V3 no pca/model3.pkl')
ensemble_model_svm_mobilenet = joblib.load('mobilenet V3 no pca/ensemble_model_svm.pkl')
pca_model_mobilenet = joblib.load('mobilenet V3 no pca/pca_model.pkl')
svm_model_mobilenet = joblib.load('mobilenet V3 no pca/svm.pkl')
scaler_model_mobilenet = joblib.load('mobilenet V3 no pca/scaler.pkl')

def read_and_convert_to_gray(image_path):
    # Đọc ảnh bằng OpenCV
    image = cv2.imread(image_path)
    # Chuyển đổi ảnh sang thang độ xám
    gray_image = color.rgb2gray(image)
    return gray_image
def extract_features(image, model):
    with torch.no_grad():
        feature = model(image).squeeze().cpu().numpy()  # Đảm bảo chuyển tensor về CPU
    return feature
# Giao diện ứng dụng Streamlit
st.title("MRI Image Classification")
uploaded_file = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="1. Original image", width=300)

    # Xử lý hình ảnh
    original, gray, thresh, contours, extreme_points, cropped, resized = process_image("temp.png")

    # Hiển thị các bước xử lý
    st.subheader("Preprocessing image")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(gray, caption="2. Grayscale image", use_container_width=True)
        st.image(extreme_points, caption="5. Extreme points", use_container_width=True)
    with col2:
        st.image(thresh, caption="3. Thresholding", use_container_width=True)
        st.image(cropped, caption="6. Cropped image", use_container_width=True)

    with col3:
        st.image(contours, caption="4. Contours", use_container_width=True)
        st.image(resized, caption="7. Resized image (224x224)", use_container_width=True)

    model_choice = st.radio("Select classification method:", (   "Resnet18 + SVM", "Resnet18 + PCA + Ensemble Models", "Resnet18 + PCA + SVM", "Resnet18 + Ensemble Models"
                                                                ,"MobilenetV3 + SVM", "MobilenetV3 + PCA + Ensemble Models", "MobilenetV3 + PCA + SVM", "MobilenetV3 + Ensemble Models"))

    if model_choice == "MobilenetV3 + SVM":
        img = preprocess_image(resized, device)
        image_features_resnet = extract_features(img, mobilenet_v3_large_model)
        image_features_resnet = scaler_model_mobilenet.transform(image_features_resnet.reshape(1, -1))
        y_pred = svm_model_mobilenet.predict(image_features_resnet)
        predicted_label = y_pred[0]
        img = load_img("temp.png")
        img_array = img_to_array(img)
        st.subheader(f"Prediction label: {y_pred[0]}")
        st.image(img_array.astype('uint8'), caption=f"Predicted Label: {y_pred[0]}", use_container_width=True)
    elif model_choice == "Resnet18 + SVM":
        img = preprocess_image(resized, device)
        image_features_resnet = extract_features(img, resnet18_model)
        image_features_resnet = scaler_model_resnet.transform(image_features_resnet.reshape(1, -1))
        y_pred = svm_model_resnet.predict(image_features_resnet)
        predicted_label = y_pred[0]
        img = load_img("temp.png")
        img_array = img_to_array(img)
        st.subheader(f"Prediction label: {y_pred[0]}")
        st.image(img_array.astype('uint8'), caption=f"Predicted Label: {y_pred[0]}", use_container_width=True)
    elif model_choice == "MobilenetV3 + Ensemble Models":
        img = preprocess_image(resized, device)
        image_features_resnet = extract_features(img, mobilenet_v3_large_model)
        features_combined = image_features_resnet.reshape(1, -1)
        y_pred1_prob = model1_mobilenet.predict(features_combined).flatten()
        y_pred2_prob = model2_mobilenet.predict(features_combined).flatten()
        y_pred3_prob = model3_mobilenet.predict(features_combined).flatten()
        y_pred_probs = np.column_stack((y_pred1_prob, y_pred2_prob, y_pred3_prob))
        y_pred_ensemble_svm = ensemble_model_svm_mobilenet.predict(y_pred_probs)
        predicted_label = y_pred_ensemble_svm[0]
        st.subheader(f"Prediction label: {predicted_label}")
        st.image("temp.png", caption="Result Image", use_container_width=True)
    elif model_choice == "Resnet18 + Ensemble Models":
        img = preprocess_image(resized, device)
        image_features_resnet = extract_features(img, resnet18_model)
        features_combined = image_features_resnet.reshape(1, -1)
        y_pred1_prob = model1_resnet.predict(features_combined).flatten()
        y_pred2_prob = model2_resnet.predict(features_combined).flatten()
        y_pred3_prob = model3_resnet.predict(features_combined).flatten()
        y_pred_probs = np.column_stack((y_pred1_prob, y_pred2_prob, y_pred3_prob))
        y_pred_ensemble_svm = ensemble_model_svm_resnet.predict(y_pred_probs)
        predicted_label = y_pred_ensemble_svm[0]
        st.subheader(f"Prediction label: {predicted_label}")
        st.image("temp.png", caption="Result Image", use_container_width=True)
    elif model_choice == "MobilenetV3 + PCA + SVM":
        img = preprocess_image(resized, device)
        image_features_resnet = extract_features(img, mobilenet_v3_large_model)
        image_features_pca = preprocess_image_pca(resized)
        image_features_pca = pca_model_mobilenet_pca.transform(image_features_pca.reshape(1, -1))
        features_combined = np.hstack((image_features_resnet.reshape(1, -1), image_features_pca))
        features_combined = scaler_model_mobilenet_pca.transform(features_combined.reshape(1, -1))
        y_pred = svm_model_mobilenet_pca.predict(features_combined)
        predicted_label = y_pred[0]
        img = load_img("temp.png")
        img_array = img_to_array(img)
        st.subheader(f"Prediction label: {y_pred[0]}")
        st.image(img_array.astype('uint8'), caption=f"Predicted Label: {y_pred[0]}", use_container_width=True)
    elif model_choice == "Resnet18 + PCA + SVM":
        img = preprocess_image(resized, device)
        image_features_resnet = extract_features(img, resnet18_model)
        image_features_pca = preprocess_image_pca(resized)
        image_features_pca = pca_model_resnet_pca.transform(image_features_pca.reshape(1, -1))
        features_combined = np.hstack((image_features_resnet.reshape(1, -1), image_features_pca))
        features_combined = scaler_model_resnet_pca.transform(features_combined.reshape(1, -1))
        y_pred = svm_model_resnet_pca.predict(features_combined)
        predicted_label = y_pred[0]
        img = load_img("temp.png")
        img_array = img_to_array(img)
        st.subheader(f"Prediction label: {y_pred[0]}")
        st.image(img_array.astype('uint8'), caption=f"Predicted Label: {y_pred[0]}", use_container_width=True)
    elif model_choice == "MobilenetV3 + PCA + Ensemble Models":
        img = preprocess_image(resized, device)
        image_features_resnet = extract_features(img, mobilenet_v3_large_model)
        image_features_pca = preprocess_image_pca(resized)
        image_features_pca = pca_model_mobilenet_pca.transform(image_features_pca.reshape(1, -1))
        features_combined = np.hstack((image_features_resnet.reshape(1, -1), image_features_pca))
        y_pred1_prob = model1_mobilenet_pca.predict(features_combined).flatten()
        y_pred2_prob = model2_mobilenet_pca.predict(features_combined).flatten()
        y_pred3_prob = model3_mobilenet_pca.predict(features_combined).flatten()
        y_pred_probs = np.column_stack((y_pred1_prob, y_pred2_prob, y_pred3_prob))
        y_pred_ensemble_svm = ensemble_model_svm_mobilenet_pca.predict(y_pred_probs)
        predicted_label = y_pred_ensemble_svm[0]
        st.subheader(f"Prediction label: {predicted_label}")
        st.image("temp.png", caption="Result Image", use_container_width=True)
    elif model_choice == "Resnet18 + PCA + Ensemble Models":
        img = preprocess_image(resized, device)
        image_features_resnet = extract_features(img, resnet18_model)
        image_features_pca = preprocess_image_pca(resized)
        image_features_pca = pca_model_resnet_pca.transform(image_features_pca.reshape(1, -1))
        features_combined = np.hstack((image_features_resnet.reshape(1, -1), image_features_pca))
        y_pred1_prob = model1_resnet_pca.predict(features_combined).flatten()
        y_pred2_prob = model2_resnet_pca.predict(features_combined).flatten()
        y_pred3_prob = model3_resnet_pca.predict(features_combined).flatten()
        y_pred_probs = np.column_stack((y_pred1_prob, y_pred2_prob, y_pred3_prob))
        y_pred_ensemble_svm = ensemble_model_svm_resnet_pca.predict(y_pred_probs)
        predicted_label = y_pred_ensemble_svm[0]
        st.subheader(f"Prediction label: {predicted_label}")
        st.image("temp.png", caption="Result Image", use_container_width=True)
    if predicted_label == 1:
            st.warning("Label is 1. Do you want to segment the image?")
            choice = st.radio("Select an option:", ("Yes", "No"), index=1)
        
            if choice == "Yes":
                st.write("You chose to segment the image.")
                # Add logic for image segmentation here
                image_11 = color.rgb2gray(original)
                st.image(image_11, caption="1", use_container_width=True)
    
                # Lấy kích thước và chuẩn bị dữ liệu
                row, col = image_11.shape
                image_11_2d = np.reshape(image_11, (row * col, 1))
                
                # Áp dụng Mean Shift
                bandwidth = estimate_bandwidth(image_11_2d, quantile=0.2, n_samples=500)
                mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                mean_shift_labels = mean_shift.fit_predict(image_11_2d)
                
                # Tạo ảnh phân đoạn
                segmented_image_11 = mean_shift_labels.reshape((row, col))
                
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.imshow(segmented_image_11, cmap='viridis')
                ax.axis('off')
                
                st.pyplot(fig)
                st.caption("Segmented Image with Mean Shift Clustering")
                row, col = image_11.shape
                image_11_2d = np.reshape(image_11, (row * col, 1))
                # Áp dụng Mean Shift
                bandwidth = estimate_bandwidth(image_11_2d, quantile=0.2, n_samples=50)
                mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                mean_shift_labels = mean_shift.fit_predict(image_11_2d)
                segmented_image_11 = mean_shift_labels.reshape((row, col))
                
                # Xử lý kích thước cụm
                unique, counts = np.unique(segmented_image_11, return_counts=True)
                cluster_sizes = dict(zip(unique, counts))
                top_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)[:2]
                
                top_left_cluster = segmented_image_11[0, 0]
                if top_left_cluster not in top_clusters:
                    top_clusters.append(top_left_cluster)
                    top_clusters = sorted(top_clusters, key=cluster_sizes.get, reverse=True)[:2]
                
                # Gán màu cho cụm
                segmented_colored = np.zeros((row, col, 3), dtype=np.uint8)
                for r in range(row):
                    for c in range(col):
                        if segmented_image_11[r, c] in top_clusters:
                            original_color = plt.cm.viridis(segmented_image_11[r, c] / len(unique))[:3]
                            segmented_colored[r, c] = (np.array(original_color) * 255).astype(np.uint8)
                        else:
                            segmented_colored[r, c] = [255, 0, 0]
                 # Hiển thị ảnh với Streamlit
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                ax.imshow(segmented_colored)
                ax.axis('off')
                # Streamlit UI
                st.title("Segmented Image with Highlighted Clusters")
                st.pyplot(fig)
    
    
                red_channel = segmented_colored[:, :, 0]
    
                red_mask = red_channel == 255
                
                labeled, num_features = ndimage.label(red_mask)
                
                component_sizes = ndimage.sum(red_mask, labeled, range(num_features + 1))
                
                largest_component_label = np.argmax(component_sizes)
                
                largest_component_mask = labeled == largest_component_label
                
                final_segmented_colored = np.copy(segmented_colored)
                final_segmented_colored[~largest_component_mask] = [0, 0, 0]
    
                # Hiển thị với Streamlit
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                ax.imshow(final_segmented_colored)
                ax.axis('off')
                # Streamlit UI
                st.title("Largest Component in Red Regions")
                st.pyplot(fig)
                st.caption("Only the largest connected component in red regions is preserved.")
                image_11_final = image_11.copy()

                red_channel = final_segmented_colored[:, :, 0]
                red_mask = red_channel == 255
                
                image_11_final = np.uint8(image_11 * 255)
                
                image_11_final_rgb = cv2.cvtColor(image_11_final, cv2.COLOR_GRAY2RGB)
                
                row, col = image_11_final.shape
                
                for r in range(row):
                    for c in range(col):
                        if red_mask[r, c]:
                            image_11_final_rgb[r, c] = [255, 0, 0]
                # Hiển thị với Streamlit
                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                ax.imshow(image_11_final_rgb)
                ax.axis('off')
    
                # Streamlit UI
                st.title("Final Image")
                st.pyplot(fig)
                
    else:
                st.write("You chose not to segment the image.")