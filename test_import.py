import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import xgboost as xgb
    print(f"XGBoost imported successfully! Version: {xgb.__version__}")
    XGBOOST_AVAILABLE = True
except ImportError as e:
    print(f"XGBoost import failed: {e}")
    XGBOOST_AVAILABLE = False

print(f"\nXGBOOST_AVAILABLE = {XGBOOST_AVAILABLE}") 