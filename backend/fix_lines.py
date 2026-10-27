import pathlib
p = pathlib.Path(r'train.py')
s = p.read_text(encoding='utf-8')
old = '''    print("Saved SHAP Explainability Plots (reports/):")
    print(f"  [OK] Logistic Regression -> {REPORTS_DIR / 'shap_lr.png'} (summary), {REPORTS_DIR / 'shap_lr_force.png'} (force)")
    print(f"  [OK] Random Forest       -> {REPORTS_DIR / 'shap_rf.png'} (summary), {REPORTS_DIR / 'shap_rf_force.png'} (force)")
    print(f"  [OK] XGBoost             -> {REPORTS_DIR / 'shap_xgb.png'} (summary), {REPORTS_DIR / 'shap_xgb_force.png'} (force)")
'''
new = '''    print("Saved SHAP Explainability Plots (reports/):")
    lr_s = REPORTS_DIR / 'shap_lr.png'
    lr_f = REPORTS_DIR / 'shap_lr_force.png'
    rf_s = REPORTS_DIR / 'shap_rf.png'
    rf_f = REPORTS_DIR / 'shap_rf_force.png'
    xg_s = REPORTS_DIR / 'shap_xgb.png'
    xg_f = REPORTS_DIR / 'shap_xgb_force.png'
    print(f"  [OK] Logistic Regression -> {lr_s} (summary), {lr_f} (force)")
    print(f"  [OK] Random Forest       -> {rf_s} (summary), {rf_f} (force)")
    print(f"  [OK] XGBoost             -> {xg_s} (summary), {xg_f} (force)")
'''
if old not in s:
    print('MISS')
    raise SystemExit(1)
p.write_text(s.replace(old, new), encoding='utf-8')
print('OK')
