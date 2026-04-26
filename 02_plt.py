import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os

# ==========================================
# ⚙️ CONFIG: ตั้งค่า Path สำหรับ Input / Output
# ==========================================
# 1. Path ของไฟล์ผลลัพธ์จากขั้นตอนเทรน (CSV)
INPUT_CSV_PATH = r'D:\Senioryear\FN_ProJ_HydroML\Result\test_results.csv'

# 2. Path ของโมเดลที่บันทึกไว้ (เพื่อนำมาดู Feature Importance)
INPUT_MODEL_PATH = r'D:\Senioryear\FN_ProJ_HydroML\Result\Model\xgboost_bestT_model.json'

# 3. Path สำหรับบันทึกรูปภาพกราฟ
OUTPUT_FOLDER = r'D:\Senioryear\FN_ProJ_HydroML\Result\Plots'
# ==========================================

def main():
    # สร้างโฟลเดอร์สำหรับเก็บรูปภาพหากยังไม่มี
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 สร้างโฟลเดอร์เก็บกราฟที่: {OUTPUT_FOLDER}")

    # 1. โหลดข้อมูลจาก CSV
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์ CSV ที่: {INPUT_CSV_PATH}")
        return

    # กรองเอาบรรทัด "OVERALL SUMMARY" ออก เพื่อไม่ให้รบกวนการพล็อตจุด
    df_plot = df[df['Pumpingrat'] != 'OVERALL SUMMARY'].copy()
    
    # แปลงข้อมูลเป็นตัวเลข (Numeric)
    cols_to_convert = ['Actual_T', 'Predicted_T']
    for col in cols_to_convert:
        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
    
    df_plot = df_plot.dropna(subset=['Actual_T', 'Predicted_T'])
    
    # คำนวณค่า Error (Actual - Predicted)
    df_plot['Error'] = df_plot['Actual_T'] - df_plot['Predicted_T']

    print("📊 กำลังเริ่มสร้างกราฟประเมินผล...")

    # --- กราฟที่ 1: Predicted vs. Actual Plot (เส้น 45 องศา) ---
    plt.figure(figsize=(8, 6))
    max_val = max(df_plot['Actual_T'].max(), df_plot['Predicted_T'].max())
    min_val = min(df_plot['Actual_T'].min(), df_plot['Predicted_T'].min())
    
    plt.scatter(df_plot['Actual_T'], df_plot['Predicted_T'], alpha=0.6, color='#3498db', edgecolors='w', s=80)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (45°)')
    
    plt.title('1. Predicted vs. Actual Plot', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Transmissivity (T)', fontsize=12)
    plt.ylabel('Predicted Transmissivity (T)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_FOLDER, '01_predicted_vs_actual.png'), dpi=300)
    print("✅ บันทึกกราฟ 1: Predicted vs Actual")

    # --- กราฟที่ 2: Residual Plot (ตรวจสอบความลำเอียง) ---
    plt.figure(figsize=(8, 6))
    plt.scatter(df_plot['Predicted_T'], df_plot['Error'], alpha=0.6, color='#9b59b6', edgecolors='w', s=80)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    
    plt.title('2. Residual Plot (Model Bias Check)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Value (Predicted_T)', fontsize=12)
    plt.ylabel('Error (Actual - Predicted)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_FOLDER, '02_residual_plot.png'), dpi=300)
    print("✅ บันทึกกราฟ 2: Residual Plot")

    # --- กราฟที่ 3: Distribution of Errors (Histogram) ---
    plt.figure(figsize=(8, 6))
    sns.histplot(df_plot['Error'], kde=True, color='#2ecc71', bins=15)
    
    plt.title('3. Distribution of Errors (Normal Distribution Check)', fontsize=14, fontweight='bold')
    plt.xlabel('Error Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_FOLDER, '03_error_distribution.png'), dpi=300)
    print("✅ บันทึกกราฟ 3: Error Distribution")

    # --- กราฟที่ 4: Feature Importance ---
    if os.path.exists(INPUT_MODEL_PATH):
        try:
            # โหลดโมเดลเพื่อดึงค่าความสำคัญของฟีเจอร์
            bst = xgb.Booster()
            bst.load_model(INPUT_MODEL_PATH)
            
            # ดึงคะแนนความสำคัญ
            importance = bst.get_score(importance_type='weight')
            # จับคู่ชื่อฟีเจอร์ (XGBoost อาจจะใช้ f0, f1... ถ้าไม่ได้ระบุชื่อตอนโหลด DMatrix)
            # ในกรณีนี้เราจะใช้ชื่อจาก DataFrame
            feat_importances = pd.Series(importance)
            feat_importances = feat_importances.sort_values(ascending=True)

            plt.figure(figsize=(10, 6))
            feat_importances.plot(kind='barh', color='#f1c40f')
            plt.title('4. Feature Importance (XGBoost Weight)', fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tick_params(axis='y', labelsize=9, pad=-0.5)
            plt.yticks(fontsize=8)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(OUTPUT_FOLDER, '04_feature_importance.png'), dpi=300)
            print("✅ บันทึกกราฟ 4: Feature Importance")
        except Exception as e:
            print(f"⚠️ ไม่สามารถวาดกราฟ Feature Importance ได้: {e}")
    else:
        print(f"⚠️ ไม่พบไฟล์โมเดลที่ {INPUT_MODEL_PATH} จึงข้ามกราฟ Feature Importance")

    print(f"\n🎉 สร้างกราฟทั้งหมดเสร็จสิ้น! ตรวจสอบไฟล์ได้ที่โฟลเดอร์: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()