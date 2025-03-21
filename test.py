# test_import.py
try:
    import models.machine_learning
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
