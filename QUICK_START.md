# 🚀 QUICK START GUIDE - TL;DR Version

## For People Who Just Want It Running NOW!

---

## ⚡ 3-Minute Setup (Windows)

### Step 1: Install Python
1. Download: https://www.python.org/downloads/
2. Run installer
3. ✅ **CHECK "Add Python to PATH"**
4. Click "Install Now"

### Step 2: Open PowerShell in Project Folder
1. Open project folder in File Explorer
2. Shift + Right-click in folder
3. Choose "Open PowerShell window here"

### Step 3: Run Auto-Setup
```powershell
.\start.ps1
```

Choose option 1 (Launch Web App)

**Done!** Browser opens automatically. 🎉

---

## ⚡ 3-Minute Setup (Mac/Linux)

### Step 1: Install Python
```bash
brew install python@3.11  # Mac
sudo apt install python3  # Linux
```

### Step 2: Navigate to Project
```bash
cd /path/to/your/project
```

### Step 3: Run Auto-Setup
```bash
chmod +x start.sh
./start.sh
```

Choose option 1 (Launch Web App)

**Done!** Browser opens automatically. 🎉

---

## 📱 Alternative: Manual Setup

If auto-setup doesn't work:

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Mac/Linux

# Install packages
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 🎯 Quick Commands Reference

### Run Web App:
```bash
streamlit run app.py
```

### Train Models (No Web Interface):
```bash
python satellite_predictor.py
```

### Generate Diagrams:
```bash
python generate_diagrams.py
```

### Stop App:
- Press `Ctrl + C` in terminal

---

## 🔧 Common Fixes

### "Python not recognized"
**Fix:** Reinstall Python, check "Add to PATH"

### "Cannot run scripts" (Windows)
**Fix:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Port already in use"
**Fix:**
```bash
streamlit run app.py --server.port 8502
```

### "Module not found"
**Fix:**
```bash
pip install -r requirements.txt
```

---

## 📊 Using the App

### 1. Load Data
- Go to "Home & Data Overview"
- Upload CSV files (GEO and MEO)

### 2. Train Models
- Go to "Train Models"
- Click "Start Training"
- Wait 10-15 minutes

### 3. Make Predictions
- Go to "Make Predictions"
- Select satellite type
- Click "Generate Predictions"

### 4. Evaluate
- Go to "Analysis & Evaluation"
- Click "Run Evaluation"
- Check if p-value > 0.05 ✅

---

## 🌐 Deploy Online (Free)

### Streamlit Cloud:
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repository
4. Deploy!

### Hugging Face:
1. Create account at huggingface.co
2. Create new Space (Streamlit)
3. Upload files
4. Auto-deploys!

---

## 📁 Files You Need

**Required:**
- ✅ app.py
- ✅ satellite_predictor.py
- ✅ requirements.txt
- ✅ DATA_GEO_Train.csv
- ✅ DATA_MEO_Train.csv

**Optional:**
- 📖 README.md
- 📊 FLOWCHART.md
- 🎓 BEGINNER_GUIDE.md
- 🚀 start.ps1 / start.sh
- 📈 generate_diagrams.py

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Install Python | 5 min |
| Install packages | 5-10 min |
| First-time setup | 15-20 min |
| Training models | 10-20 min |
| Making predictions | < 1 min |

**Total First Run:** ~30-45 minutes

**Subsequent Runs:** ~2 minutes (just launch app!)

---

## 💾 System Requirements

**Minimum:**
- 8 GB RAM
- 2 GB free space
- Internet connection
- Python 3.8+

**Recommended:**
- 16 GB RAM
- 5 GB free space
- Fast internet
- Python 3.11

---

## 🎯 Success Checklist

Before demo/presentation:

- [ ] Python installed and working
- [ ] All packages installed
- [ ] Data files loaded successfully
- [ ] Models trained (saved in `models/` folder)
- [ ] App runs without errors
- [ ] Predictions generate correctly
- [ ] Evaluation shows good p-values
- [ ] Screenshots/diagrams ready
- [ ] Team knows how to navigate app
- [ ] Backup plan if internet fails

---

## 📞 Emergency Troubleshooting

**App won't start?**
→ Check terminal for error messages
→ Verify all files are present
→ Reinstall packages

**Training fails?**
→ Reduce batch size to 16
→ Reduce epochs to 30
→ Check CSV files are correct format

**Out of memory?**
→ Close other programs
→ Reduce batch size
→ Use smaller dataset

**Slow performance?**
→ Normal for first training
→ Saved models load instantly
→ Consider using GPU

---

## 🎓 Key Concepts (Explain in Presentation)

**LSTM Model:**
- Type of neural network
- Learns patterns over time
- Good for time-series prediction

**7-Day Sequence:**
- Uses past 7 days
- Predicts 8th day
- Like weather forecasting

**Shapiro-Wilk Test:**
- Tests if errors are random
- p > 0.05 = model works well
- Proves systematic errors removed

**4 Error Components:**
- X, Y, Z position errors
- Clock error
- All measured in meters

---

## 🏆 Presentation Tips

1. **Demo Live:**
   - Show data loading
   - Show visualizations
   - Run a quick prediction

2. **Explain Results:**
   - Show p-value > 0.05
   - Explain what it means
   - Show prediction accuracy

3. **Show Diagrams:**
   - Architecture diagram
   - Data flow diagram
   - Training curves

4. **Highlight Features:**
   - Automatic preprocessing
   - Outlier treatment
   - Interactive visualizations
   - Easy to use interface

---

## 🔗 Important Links

**Documentation:**
- Full README: `README.md`
- Beginner Guide: `BEGINNER_GUIDE.md`
- Technical Details: `FLOWCHART.md`

**Resources:**
- Python: https://python.org
- Streamlit: https://streamlit.io
- TensorFlow: https://tensorflow.org

**Deploy:**
- Streamlit Cloud: https://share.streamlit.io
- Hugging Face: https://huggingface.co/spaces

---

## 📋 Pre-Demo Checklist

**30 minutes before:**
- [ ] Test app on demo computer
- [ ] Verify internet connection
- [ ] Close unnecessary programs
- [ ] Have backup slides ready
- [ ] Charge laptop fully
- [ ] Test projector connection

**5 minutes before:**
- [ ] Launch app
- [ ] Load sample data
- [ ] Have a prediction ready
- [ ] Open evaluation results
- [ ] Have diagrams ready
- [ ] Brief team on roles

---

## 🎤 Demo Script

**Opening (30 sec):**
"We've built an AI system that predicts satellite errors using LSTM neural networks..."

**Show Data (1 min):**
"Here's our 7-day GEO and MEO satellite data with X, Y, Z errors..."

**Show Training (30 sec):**
"Our model trains on these patterns with 3 LSTM layers..."

**Show Predictions (1 min):**
"It predicts the 8th day errors with high accuracy..."

**Show Evaluation (1 min):**
"Shapiro-Wilk test confirms our model removed systematic errors..."

**Conclusion (30 sec):**
"This helps ISRO improve navigation satellite accuracy..."

**Total:** 4 minutes (leaves 1 minute for questions in 5-min slot)

---

## 🚨 If Everything Breaks

**Plan B:**
1. Use screenshots/video of working app
2. Walk through code explanations
3. Show diagrams and flowcharts
4. Explain methodology in detail
5. Have PDF of results ready

**Remember:**
- Judges care about approach, not just working demo
- Explain your thought process
- Show you understand the problem
- Demonstrate technical knowledge

---

## ✨ Bonus Points

**Impress judges by:**
- Explaining LSTM architecture
- Discussing outlier treatment strategy
- Showing Shapiro-Wilk test understanding
- Mentioning deployment scalability
- Suggesting future improvements

---

**You're ready! Break a leg! 🎭🚀**

---

*Quick Reference Card*  
*ISRO SIH 2025 - PS 25176*  
*Print this and keep handy during demo!*
