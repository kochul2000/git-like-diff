import datetime as dt
from io import BytesIO
import pandas as pd
import streamlit as st
from gld.git_like_diff import GitLikeDiff
from gld.gld_spec import dd_crf_spec, dd_choice_group_spec, dd_tag_spec, dd_visit_spec

CONFIGS = {
    "CRF": dd_crf_spec,
    "ChoiceGroup": dd_choice_group_spec,
    "Tag": dd_tag_spec,
    "Visit": dd_visit_spec,
}

# ────────────── Streamlit UI ──────────────
st.set_page_config(page_title="Excel Git-Diff", layout="wide")
st.title("DD2 Excel 비교 기능 (git-like-diff, LCS 정렬)")

c1, c2 = st.columns(2)
with c1:
    old_file = st.file_uploader("OLD (.xlsx)", type="xlsx")
with c2:
    new_file = st.file_uploader("NEW (.xlsx)", type="xlsx")

if old_file and new_file:
    old_xls = pd.ExcelFile(old_file)
    new_xls = pd.ExcelFile(new_file)
    sheets  = sorted(set(old_xls.sheet_names) & set(new_xls.sheet_names)
                     & CONFIGS.keys())

    if not sheets:
        st.error("두 파일에서 CONFIG에 정의된 공통 시트를 찾지 못했습니다.")
        st.stop()

    st.success(f"비교 대상 시트: {', '.join(sheets)}")
    if st.button("Diff 실행 & 다운로드"):
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for sheet in sheets:
                cfg = CONFIGS[sheet]
                df_old = old_xls.parse(sheet, keep_default_na=False, dtype=str)
                df_new = new_xls.parse(sheet, keep_default_na=False, dtype=str)

                diff = GitLikeDiff(**cfg).execute(df_old, df_new, baseline="old")
                diff.to_excel(writer, sheet_name=sheet, index=False)

        buf.seek(0)
        today = dt.date.today().isoformat()
        st.download_button(
            label="📥 Diff 결과 다운로드",
            data=buf,
            file_name=f"diff_{today}.xlsx",
            mime=("application/vnd.openxmlformats-officedocument."
                  "spreadsheetml.sheet"),
        )
else:
    st.info("왼쪽 → OLD, 오른쪽 → NEW 파일을 업로드해주세요.")
