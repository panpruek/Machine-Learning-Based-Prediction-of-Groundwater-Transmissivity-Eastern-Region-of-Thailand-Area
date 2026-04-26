import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIG: ตั้งค่า Path สำหรับ Input / Output
# ==========================================
INPUT_DATA_PATH = r'D:\Senioryear\FN_ProJ_HydroML\47_48.txt'                 
OUTPUT_MODEL_PATH = r'D:\Senioryear\FN_ProJ_HydroML\Result\Model\xgboost_bestT_model.json' 
OUTPUT_TEST_RESULTS_CSV = r'D:\Senioryear\FN_ProJ_HydroML\Result\test_results.csv'  
OUTPUT_LOSS_PLOT = r'D:\Senioryear\FN_ProJ_HydroML\Result\loss_plot.png'
OUTPUT_ACC_PLOT = r'D:\Senioryear\FN_ProJ_HydroML\Result\accuracy_plot.png'
# ==========================================

def main():
    print("="*50)
    print("🚀 เริ่มต้นระบบวิเคราะห์ Transmissivity Pipeline (No Staticwate, No DeepDevelo, With Q/s)")
    print("="*50)

    # ---------------------------------------------------------
    # STEP 1: Data Preprocessing & Augmentation
    # ---------------------------------------------------------
    print("\n[Step 1] กำลังโหลดและเตรียมข้อมูล...")
    
    df = pd.read_csv(INPUT_DATA_PATH)
    df = df[df['FID'] != 'FID'].reset_index(drop=True)
    df = df.drop(columns=['Average_K_', 'Average_S'])
    
    # ตัด Staticwate และ DeepDevelo ออก
    numeric_cols = ['Average_T_', 'Depthofpip', 'Pumpingrat', 'Drawdown_m']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    
    # เพิ่มตัวแปร Specific Capacity (Q/s)
    df['Specific_Capacity'] = df['Pumpingrat'] / df['Drawdown_m']
    
    le = LabelEncoder()
    df['UNIT_encoded'] = le.fit_transform(df['UNIT'].astype(str))
    
    # ✅ แก้ไขตรงนี้: เพิ่ม 'Specific_Capacity' ลงใน Features
    features = ['Pumpingrat', 'Drawdown_m', 'UNIT_encoded', 'Specific_Capacity']
    X = df[features]
    y = np.log1p(df['Average_T_'])
    
    y_binned = pd.qcut(y, q=3, labels=False, duplicates='drop')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y_binned)
    
    y_temp_binned = pd.qcut(y_temp, q=2, labels=False, duplicates='drop')
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp_binned)
    
    X_aug_list = [X_train]; y_aug_list = [y_train]
    n_copies = 100
    for _ in range(n_copies):
        X_new = X_train.copy()
        X_new['Drawdown_m'] += np.random.uniform(-0.01, 0.01, len(X_new))
        X_new['Pumpingrat'] += np.random.uniform(-0.01, 0.01, len(X_new))
        
        X_new['Drawdown_m'] = np.clip(X_new['Drawdown_m'], 0.001, None)
        X_new['Pumpingrat'] = np.clip(X_new['Pumpingrat'], 0.001, None)
        
        # คำนวณ Specific_Capacity ใหม่สำหรับข้อมูลที่เพิ่ม
        X_new['Specific_Capacity'] = X_new['Pumpingrat'] / X_new['Drawdown_m']
        
        X_aug_list.append(X_new); y_aug_list.append(y_train)
        
    X_train_aug = pd.concat(X_aug_list, ignore_index=True)
    y_train_aug = pd.concat(y_aug_list, ignore_index=True)
    print(f"✅ ขยายข้อมูลสำเร็จ! Train Set: {len(X_train_aug)} บรรทัด")

    # ---------------------------------------------------------
    # STEP 2: XGBoost Training & Automated Tuning
    # ---------------------------------------------------------
    print("\n[Step 2] เริ่มต้นฝึกสอนโมเดล (Training & Tuning)...")
    
    dtrain = xgb.DMatrix(X_train_aug, label=y_train_aug)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    best_overall_val_loss = float('inf'); best_overall_depth = None
    best_booster_overall = None; best_history = None  

    lr_schedule = [0.001, 0.0001, 0.00001]; patience = 15
    
    for depth in range(2, 9):
        print(f"  👉 ทดสอบ Tree Depth = {depth} ...", end=" ")
        current_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        current_lr_idx = 0
        params = {
            'max_depth': depth, 'objective': 'reg:squarederror', 'eval_metric': 'mae',
            'eta': lr_schedule[current_lr_idx], 'subsample': 0.7, 'colsample_bytree': 0.8,
            'lambda': 1.5, 'alpha': 0.5, 'random_state': 42
        }
        
        booster = xgb.Booster(params, [dtrain, dval])
        best_val_for_depth = float('inf'); best_booster_for_depth = None; epochs_no_improve = 0
        
        for epoch in range(8500):
            booster.update(dtrain, epoch)
            train_eval = booster.eval(dtrain); val_eval = booster.eval(dval)
            t_mae = float(train_eval.split(':')[1]); v_mae = float(val_eval.split(':')[1])
            t_r2 = r2_score(y_train_aug, booster.predict(dtrain))
            v_r2 = r2_score(y_val, booster.predict(dval))
            
            current_history['train_loss'].append(t_mae); current_history['val_loss'].append(v_mae)
            current_history['train_acc'].append(t_r2); current_history['val_acc'].append(v_r2)
            
            if v_mae < best_val_for_depth:
                best_val_for_depth = v_mae; best_booster_for_depth = booster.copy(); epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                current_lr_idx += 1
                if current_lr_idx < len(lr_schedule):
                    booster = best_booster_for_depth.copy()
                    booster.set_param({'eta': lr_schedule[current_lr_idx]})
                    epochs_no_improve = 0
                else:
                    break

        print(f"(Best MAE: {best_val_for_depth:.4f})")
        if best_val_for_depth < (best_overall_val_loss - 0.05) or best_overall_depth is None:
            best_overall_val_loss = best_val_for_depth; best_overall_depth = depth
            best_booster_overall = best_booster_for_depth.copy(); best_history = current_history  

    best_booster_overall.save_model(OUTPUT_MODEL_PATH)
    
    # --- ส่วนการสร้างกราฟ ---
    epochs_range = range(1, len(best_history['train_loss']) + 1)
    plt.figure(figsize=(10, 5)); plt.plot(epochs_range, best_history['train_loss'], label='Train MAE'); plt.plot(epochs_range, best_history['val_loss'], label='Val MAE')
    plt.title('Loss Plot'); plt.savefig(OUTPUT_LOSS_PLOT); plt.close()
    plt.figure(figsize=(10, 5)); plt.plot(epochs_range, best_history['train_acc'], label='Train R2'); plt.plot(epochs_range, best_history['val_acc'], label='Val R2')
    plt.title('Accuracy Plot'); plt.savefig(OUTPUT_ACC_PLOT); plt.close()

    # ---------------------------------------------------------
    # STEP 3: Model Evaluation & Saving Results
    # ---------------------------------------------------------
    print("\n[Step 3] ประเมินผลบนข้อมูล Test Set...")
    dtest = xgb.DMatrix(X_test)
    y_pred_real = np.expm1(best_booster_overall.predict(dtest))
    y_test_real = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)
    
    # รายงานผลที่คุณต้องการ
    mean_actual = np.mean(y_test_real); mean_pred = np.mean(y_pred_real)
    error_percentage = (mae / mean_actual) * 100 if mean_actual != 0 else 0
    
    results_df = X_test.copy()
    results_df['Actual_T'] = y_test_real.values
    results_df['Predicted_T'] = y_pred_real
    results_df['MAE_point'] = np.abs(y_test_real.values - y_pred_real)
    results_df['RMSE_point'] = np.sqrt(results_df['MAE_point']**2)
    results_df['UNIT_name'] = le.inverse_transform(results_df['UNIT_encoded'])
    results_df['Overall_MAE'] = mae; results_df['Overall_RMSE'] = rmse; results_df['Overall_R2'] = r2
    
    summary_row = pd.DataFrame([{
        'Pumpingrat': 'OVERALL SUMMARY', 'Actual_T': mean_actual, 'Predicted_T': mean_pred, 
        'MAE_point': mae, 'RMSE_point': rmse, 'UNIT_name': f'R2: {r2:.4f}'
    }])
    final_output_df = pd.concat([results_df, summary_row], ignore_index=True)
    final_output_df.to_csv(OUTPUT_TEST_RESULTS_CSV, index=False)

    print("\n" + "=" * 55)
    print("📊 REPORT: ผลการประเมินประสิทธิภาพโมเดล (Test Set)")
    print("=" * 55)
    print(f"📈 ค่าเฉลี่ย Transmissivity (T) ของจริง   : {mean_actual:.4f}")
    print(f"📉 ค่าเฉลี่ย Transmissivity (T) ที่ทำนายได้ : {mean_pred:.4f}")
    print("-" * 55)
    print(f"🎯 MAE (Mean Absolute Error)          : {mae:.4f}")
    print(f"   >> โมเดลทำนายค่า T ผิดพลาดเฉลี่ย {mae:.2f} ตารางเมตร/วัน")
    print(f"   >> (คิดเป็นความคลาดเคลื่อนเฉลี่ยประมาณ {error_percentage:.2f}% จากค่าจริง)")
    print("-" * 55)
    print(f"   RMSE (Root Mean Squared Error)     : {rmse:.4f}")
    print(f"   R-Squared (R²)                     : {r2:.4f}")
    print("=" * 55)

if __name__ == "__main__":
    main()