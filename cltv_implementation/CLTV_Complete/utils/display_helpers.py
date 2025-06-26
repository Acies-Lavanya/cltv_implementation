import streamlit as st

def display_file_summary(df, label):
    with st.expander(f"📄 {label} CSV Info", expanded=True):
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.markdown("**🔤 Columns and Data Types:**")
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}))
        st.markdown("**❗ Missing Values:**")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing"}))
        st.markdown("**👁️ Preview (First 5 Rows):**")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("**📈 Summary Statistics (Numeric Columns):**")
        st.dataframe(df.describe().T, use_container_width=True)
