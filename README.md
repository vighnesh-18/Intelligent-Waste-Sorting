
# 🧠 Smart Waste Classification

An AI-powered system for automated **multi-stage waste classification** using deep learning, transfer learning, and explainable AI (XAI) methods. This project supports scalable waste management through smarter, faster classification.

<p align="center">
  <img src="./75c15a50-0bc6-4fe1-add3-88902bad44be.png" width="80%" alt="Architecture diagram"/>
</p>

> 📝 [View Published Research Paper](https://www.researchgate.net/publication/387951109)

---

### 📂 Project Pipeline Overview

| Stage        | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| ♻️ Preprocessing | Resize & normalize images, prepare class splits for training/testing.        |
| 📊 Stage 1 Classification | Categorize waste into **Biodegradable** vs **Non-Biodegradable**. |
| 🧪 Stage 2 Classification | Classify into **9 major categories** like Glass, Metal, Plastic, etc. |
| 🔍 Stage 3 Classification | Fine-grained classification across **36 detailed waste types**. |
| 🧠 XAI Phase      | Interpret models using SHAP, Grad-CAM, Guided CAM, and saliency maps.   |
| 💻 Deployment     | Hardware-software implementation with monitoring interface.               |

---

### 🔗 Google Colab Notebooks

- 📁 [Dataset Preprocessing](https://colab.research.google.com/drive/1IWp2x9ZdXSrj0cmZ-aYfKIx4PWctxSD3#scrollTo=zb5ZpDZ7fo0w)
- 🧠 [Final Training & Testing](https://colab.research.google.com/drive/1NI3zYoxvWT4sJDhcMPj0z4ELl5k-7QGk#scrollTo=RCq5UFtLmMvJ)

---

### ⚙️ Technologies Used

- Python 3.x 🐍
- TensorFlow & Keras 🤖
- OpenCV & Image Preprocessing 📷
- SHAP, Grad-CAM 🔍
- Google Colab ☁️

---

### 🧰 Features

- ✅ Multi-stage waste classification (4 → 9 → 36 classes)
- ✅ Transfer learning with custom CNNs (PL-CNN, DP-CNN)
- ✅ Explainable AI integration (SHAP, Grad-CAM, etc.)
- ✅ Hardware & software integration
- ✅ Reproducible Colab notebooks

---

### 🖼️ Sample Output (XAI Phase)

> Add here sample saliency map or classification result if you have one (optional)

---

### 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/vighnesh-18/Smart-Waste-Classification.git
   cd Smart-Waste-Classification
   ```
2. Open notebooks in [Google Colab](https://colab.research.google.com)
3. Run `Dataset Preprocessing` → then `Model Final Training & Testing`

---

### 📚 Citation

If this project or paper helped your work, please consider citing it:

```bibtex
@article{vighnesh2025smartwaste,
  title={Smart Waste Classification using Deep Learning and XAI},
  author={Vighnesh},
  journal={ResearchGate},
  year={2025},
  url={https://www.researchgate.net/publication/387951109}
}
```

---

### 👤 Author

- [Vighnesh](https://github.com/vighnesh-18)

---

### 📄 License

This project is licensed under the MIT License.
