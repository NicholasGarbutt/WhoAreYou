# WhoAreYou 👤🔍

**WhoAreYou** is a Python-based facial recognition tool that uses a user’s webcam to identify people in real time. The project aims to improve everyday facial recognition by checking whether someone is known personally (from a private photo collection) or has an online presence.

---

## 🚀 Features

- 📷 Real-time webcam face detection  
- 🧠 Facial recognition against your personal photo database  
- 🌐 Optional search for public online presence  
- ⚡ Fast and lightweight Python implementation  
- 🔒 Local-first design (your data stays on your machine)  
- 🧩 Modular structure for easy extension  

---

## 🧠 How It Works

1. The webcam captures a live image.  
2. Faces are detected and encoded.  
3. The encoding is compared against:
   - Your local known-faces database  
   - (Optional) online/public sources  
4. The system returns:
   - ✅ Match found (known person)  
   - ❓ Unknown person  
   - 🌐 Possible online presence  

---

## 🛠️ Tech Stack

- Python 3.x  
- OpenCV  
- face_recognition  
- NumPy  
- (Optional) requests / web scraping tools  

---

# Face Recognition Project Roadmap

## Phase 1 — Core Functionality

- [ ] Basic webcam face detection  
- [ ] Local face database matching  
- [ ] Real-time bounding boxes  
- [ ] Confidence scoring  

---

## Phase 2 — Accuracy Improvements

- [ ] Better face encoding pipeline  
- [ ] Multi-image per person support  
- [ ] Adjustable tolerance threshold  
- [ ] Performance optimisations  

---

## Phase 3 — User Experience

- [ ] Simple GUI interface  
- [ ] Camera selection  
- [ ] Settings/config file  
- [ ] Result history log  

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/WhoAreYou.git
cd WhoAreYou

