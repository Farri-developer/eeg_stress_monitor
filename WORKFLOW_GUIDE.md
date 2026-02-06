# Updated Workflow Guide - NASA-TLX Integration

## 🎯 Two Complete Workflows Available

### Manual Workflow (Step-by-Step with TLX)

```
┌─────────────────────────────────────────────────┐
│        MANUAL WORKFLOW BUTTONS                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  🩺 1. Baseline BP        ▶ 2. Start Stream   │
│                                                 │
│  ⏺ 3. Start Recording    ⏹ 4. Stop Recording  │
│                                                 │
│  ⏹ 5. Stop Stream        💓 6. Question-End BP│
│                                                 │
│  📋 7. NASA-TLX Form     📊 8. Generate Dataset│
│                                                 │
│  🚀 Complete Automated Session (full auto)     │
│                                                 │
│  🚪 Exit                                        │
└─────────────────────────────────────────────────┘
```

### Button Flow (Manual)

```
Measure BP → Start Stream → Record → Stop → Stop Stream → BP Again → Fill TLX → Generate
   ↓            ↓             ↓        ↓         ↓            ↓         ↓          ↓
 Button 1    Button 2     Button 3  Button 4  Button 5   Button 6  Button 7   Button 8
```

---

## 📋 New NASA-TLX Form (Button 7)

### When to Use
- **After** Question-End BP is measured (Button 6)
- **Before** generating dataset (Button 8)
- Button is **disabled** until BP measurements are complete

### What It Asks

The form includes 6 dimensions rated 0-100:

| Dimension | Question |
|-----------|----------|
| **Mental Demand** | How mentally demanding was the task? |
| **Physical Demand** | How physically demanding was the task? |
| **Temporal Demand** | How hurried or rushed was the pace? |
| **Performance** | How successful were you? |
| **Effort** | How hard did you work? |
| **Frustration** | How stressed and annoyed were you? |

### Example Scores

```
Mental Demand:    ████████████████░░░░  75/100
Physical Demand:  ████░░░░░░░░░░░░░░░░  20/100
Temporal Demand:  ██████████░░░░░░░░░░  50/100
Performance:      ████████████████████  90/100
Effort:           ██████████████░░░░░░  65/100
Frustration:      ██████░░░░░░░░░░░░░░  30/100

Overall TLX:      55/100
```

---

## 📊 Dataset Output Options

### Option A: Without TLX (Skip Button 7)
**File:** `dataset_features.csv`

**Columns:**
- Window
- EEG features (Delta, Theta, Alpha, Beta, Gamma × 4 channels)
- Beta/Alpha ratios
- PPG metrics (HR, HRV, SDNN, RMSSD, pNN50)
- BP deltas
- StressScore
- **Stress_Label** (0, 1, 2)

### Option B: With TLX (Use Button 7)
**Files:** 
1. `dataset_features.csv` (enhanced)
2. `tlx_scores.csv` (separate)

**Additional Columns in dataset_features.csv:**
- TLX_Mental_Demand
- TLX_Physical_Demand
- TLX_Temporal_Demand
- TLX_Performance
- TLX_Effort
- TLX_Frustration
- TLX_Overall
- **Combined_Stress_TLX_Label** (0, 1, 2)

---

## 🎯 Combined Label Logic

When TLX scores are available, a **Combined_Stress_TLX_Label** is calculated:

```python
if TLX_Overall > 70 AND Stress_Label == 2:
    Combined_Label = 2  # High stress/workload
    
elif TLX_Overall < 40 AND Stress_Label == 0:
    Combined_Label = 0  # Low stress/workload
    
else:
    Combined_Label = 1  # Medium stress/workload
```

### Examples

| Physiological Stress | TLX Overall | Combined Label | Interpretation |
|---------------------|-------------|----------------|----------------|
| 2 (High) | 85 | **2** | Confirmed high stress |
| 0 (Low) | 30 | **0** | Confirmed low stress |
| 2 (High) | 35 | **1** | Mixed signals |
| 0 (Low) | 80 | **1** | Mixed signals |
| 1 (Medium) | Any | **1** | Medium state |

---

## 🚀 Automated Session Still Works

The **Complete Automated Session** button:
- Runs all 8 steps automatically
- Shows TLX dialog automatically after BP
- Generates dataset with TLX scores
- Everything happens in sequence

---

## 📝 Usage Examples

### Example 1: Quick Recording (No TLX)
```
1. Click "Baseline BP" → measure
2. Click "Start Stream"
3. Click "Start Recording" → wait 2 minutes
4. Click "Stop Recording"
5. Click "Stop Stream"
6. Click "Question-End BP" → measure
7. Skip TLX button
8. Click "Generate Dataset" → get basic labels
```

**Output:** Physiological stress labels only

---

### Example 2: Full Assessment (With TLX)
```
1. Click "Baseline BP" → measure
2. Click "Start Stream"
3. Click "Start Recording" → wait 2 minutes
4. Click "Stop Recording"
5. Click "Stop Stream"
6. Click "Question-End BP" → measure
7. Click "NASA-TLX Form" → fill sliders → submit
8. Click "Generate Dataset" → get combined labels
```

**Output:** Combined physiological + subjective labels

---

### Example 3: Automated (Easiest)
```
1. Click "Complete Automated Session"
2. Wait for Baseline BP prompt
3. Task runs for 60 seconds automatically
4. Wait for Question-End BP prompt
5. Fill TLX form when it appears
6. Done! Check output files
```

**Output:** All files with TLX integration

---

## 💡 Tips

### When to Use TLX
✅ **Use TLX when:**
- Doing formal research studies
- Comparing subjective and objective stress
- Need validated workload assessment
- Want richer feature set for ML

❌ **Skip TLX when:**
- Quick testing/debugging
- Only need physiological data
- Time constrained
- Pilot testing equipment

### TLX Best Practices
1. **Complete immediately** after task (while memory fresh)
2. **Be honest** with ratings (no right/wrong answers)
3. **Use full scale** (0-100, not just 40-60)
4. **Consider all aspects** of each dimension
5. **Performance is reverse** (higher = better, but still rated)

### Button States
- 🔵 **Blue** = Ready to use
- ⚫ **Gray** = Disabled (prerequisites not met)
- ✅ **Enabled after:** Each button enables the next step

---

## 📂 File Structure

```
D:/DataSet/
└── UserName(UserID)_20260202_143000/
    ├── eeg_raw.csv                  # Raw EEG data
    ├── ppg_raw.csv                  # Raw PPG data  
    ├── bp_measurements.csv          # BP readings
    ├── tlx_scores.csv              # ← NEW: TLX responses
    └── dataset_features.csv         # Features + labels
                                     # (includes TLX if filled)
```

---

## 🔄 Workflow Comparison

| Aspect | Manual (No TLX) | Manual (With TLX) | Automated |
|--------|----------------|-------------------|-----------|
| Steps | 7 buttons | 8 buttons | 1 button |
| Duration | User controlled | User controlled | 60s fixed |
| TLX | Optional | Yes | Yes (automatic) |
| Output | Basic labels | Combined labels | Combined labels |
| Flexibility | High | High | Low |
| Consistency | Variable | Variable | High |

---

## ⚠️ Important Notes

1. **TLX button is disabled** until Question-End BP is measured
2. **Can skip TLX** - dataset generation still works
3. **TLX scores saved separately** in tlx_scores.csv
4. **Combined labels only created** if TLX is filled
5. **Automated workflow always includes TLX**

---

## 🆘 Troubleshooting

### TLX Button Grayed Out
**Problem:** Can't click NASA-TLX button

**Solution:** 
1. Ensure Baseline BP measured (Button 1)
2. Complete recording workflow (Buttons 2-5)
3. Measure Question-End BP (Button 6)
4. Now Button 7 should be enabled

### TLX Form Won't Submit
**Problem:** OK button does nothing

**Solution:**
- All sliders have default value (50)
- Just click OK to accept defaults
- Or adjust sliders and click OK

### Dataset Generated Without TLX
**Problem:** Wanted TLX features but they're missing

**Solution:**
- Must click Button 7 (NASA-TLX) BEFORE Button 8
- Re-generate dataset after filling TLX
- Or use Automated workflow (includes TLX automatically)

---

*For detailed technical documentation, see README.md*
