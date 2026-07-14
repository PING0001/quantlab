import sys, importlib
sys.path.insert(0, r'C:\Users\cui\.codex\worktrees\ce37\quantlab0.0.3\.venv\Lib\site-packages')
for n in ['duckdb','sklearn','scipy','joblib','lightgbm','polars']:
    try:
        m = importlib.import_module(n)
        print(f'{n}: {getattr(m, "__version__", "OK")}')
    except Exception as e:
        print(f'{n}: ERROR - {e}')
