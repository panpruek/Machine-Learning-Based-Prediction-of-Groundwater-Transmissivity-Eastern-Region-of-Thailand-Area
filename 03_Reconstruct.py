import pandas as pd

# ==========================================
# ⚙️ CONFIG: ตั้งค่า Path ไฟล์
# ==========================================
PATH_TEST_RESULTS = r"D:\Senioryear\FN_ProJ_HydroML\Result\test_results.csv"  # ไฟล์ผลลัพธ์จากการเทรน
PATH_ORIGINAL_TXT = r"D:\Senioryear\FN_ProJ_HydroML\47_48.txt"         # ไฟล์ข้อมูลดิบดั้งเดิม
PATH_OUTPUT_FULL = r"D:\Senioryear\FN_ProJ_HydroML\Result\test_results_full_columns.csv" # ไฟล์ใหม่ที่จะบันทึก
# ==========================================

def main():
    print("🔍 กำลังเริ่มต้นการดึงข้อมูลคอลัมน์จากไฟล์ต้นฉบับ...")

    # 1. โหลดผลลัพธ์การ Test
    df_test = pd.read_csv(PATH_TEST_RESULTS)
    
    # ตัดบรรทัด "OVERALL SUMMARY" ออก เพื่อให้ข้อมูลสะอาดก่อนรวม
    df_test = df_test[df_test['Pumpingrat'] != 'OVERALL SUMMARY'].copy()
    
    # 2. โหลดข้อมูลต้นฉบับจาก TXT
    df_orig = pd.read_csv(PATH_ORIGINAL_TXT)
    
    # ลบ Header ที่ซ้ำซ้อนในไฟล์ TXT (ถ้ามี)
    if 'FID' in df_orig.columns:
        df_orig = df_orig[df_orig['FID'] != 'FID'].reset_index(drop=True)

    # แปลงชนิดข้อมูลคอลัมน์ที่เป็นกุญแจให้เป็นตัวเลข เพื่อให้ Matching แม่นยำ
    key_cols = ['Pumpingrat', 'Drawdown_m']
    for col in key_cols:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
        df_orig[col] = pd.to_numeric(df_orig[col], errors='coerce')

    # 3. รวมข้อมูล (Merge) 
    # เราใช้ Left Join เพื่อรักษาลำดับของบ่อใน Test Set ไว้ และดึงข้อมูลที่ตรงกันจาก Orig มาใส่
    # เราจะ Match ด้วย Pumpingrat และ Drawdown_m
    df_combined = pd.merge(
        df_test, 
        df_orig, 
        on=['Pumpingrat', 'Drawdown_m'], 
        how='left',
        suffixes=('', '_orig') # ถ้ามีคอลัมน์ซ้ำกัน ตัวจากไฟล์ TXT จะถูกเติมท้ายด้วย _orig
    )

    # 4. จัดระเบียบข้อมูลเบื้องต้น
    # ตรวจสอบว่ามีคอลัมน์ที่ซ้ำซ้อน (เช่น Average_T_ กับ Actual_T) หรือไม่
    # ในขั้นตอนนี้เราจะเก็บทุกคอลัมน์ไว้ตามที่โจทย์ต้องการ
    
    # 5. บันทึกผลลัพธ์
    df_combined.to_csv(PATH_OUTPUT_FULL, index=False)
    
    print(f"✅ รวมข้อมูลสำเร็จ!")
    print(f"📊 จำนวนข้อมูลใน Test Set: {len(df_test)} บ่อ")
    print(f"📂 บันทึกไฟล์ที่มีคอลัมน์ครบถ้วนแล้วที่: {PATH_OUTPUT_FULL}")
    print("-" * 30)
    print("คอลัมน์ที่ได้เพิ่มมา:")
    # แสดงรายชื่อคอลัมน์ใหม่ๆ ที่ดึงมาจากไฟล์ต้นฉบับได้
    new_cols = [c for c in df_orig.columns if c not in ['Pumpingrat', 'Drawdown_m']]
    print(new_cols)

if __name__ == "__main__":
    main()