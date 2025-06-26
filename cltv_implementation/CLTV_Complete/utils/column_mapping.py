import difflib

def auto_map(col_list, candidates):
    for name in candidates:
        match = difflib.get_close_matches(name, col_list, n=1, cutoff=0.6)
        if match:
            return match[0]
    return None

def auto_map_columns(df, expected_cols):
    return {
        k: auto_map(df.columns.tolist(), v) or ""
        for k, v in expected_cols.items()
    }
