"""
git_like_diff.py  –  path 제거 + dict 반환 + LCS(최장 공통 부분열) 기반 정렬

• 결과 컬럼   : change( "A" | "D" | "M-" | "M+" | "" )
• 정렬 규칙   : LCS 순서에 따라 삭제 블록 뒤에 바로 추가 블록이 오도록
• 성능        :  Patience-diff O((m+n) log n) / O(n) 메모리
"""

from bisect import bisect_left
from typing import Callable, Literal, Sequence, Union, Optional

import pandas as pd

from gld.const import test_new_path, test_old_path, test_result_path
from gld.gld_spec import dd_crf_spec

DiffCallable = Callable[[pd.DataFrame], pd.Series]
DiffSpec     = Union[Sequence[str], DiffCallable]


class Change:
    DELETE = "D"  # 삭제
    MODIFIED_FROM = "M-"  # 수정 (이전 값)
    MODIFIED = "M"  # 수정 (변경된 값, M- M+ 사이에 삽입. GitLikeDiff 의 유틸함수로 추후 추가됨. optional)
    MODIFIED_TO = "M+"  # 수정 (새 값)
    ADD = "A"  # 추가
    EQUAL = ""  # 동일


CHANGE_PRIORITY = {
    Change.DELETE: 0,
    Change.MODIFIED_FROM: 1,
    Change.MODIFIED: 2,  # M- M+ 사이에 삽입됨
    Change.MODIFIED_TO: 3,
    Change.ADD: 4,
    Change.EQUAL: 5,  # 사실상 겹치는 경우 없음
}

MODIFIED_SEP = " >> "  # MODIFIED 에서 전 후 값 사이 구분자
CHANGE_COL = "__CHANGE__"  # 첫번째 컬럼으로 추가되어 chage 타입을 표시합니다.


class GitLikeDiff:
    """
    id 와 attr 기반으로 비교 분석하고, 차이점을 LCS(최장 공통 부분열) 기반 정렬하여 반환합니다.
    id 가 같으면 같은 레코드로 간주하고, attr 가 다르면 변경된 것으로 간주합니다.

    • 결과 컬럼   : change( "A" | "D" | "M-" | "M+" | "" )
    • 정렬 규칙   : LCS 순서에 따라 삭제 블록 뒤에 바로 추가 블록이 오도록
    """

    # ───────────────────── 초기화 ─────────────────────
    def __init__(self, id_spec: "DiffSpec", attr_spec: Optional["DiffSpec"] = None, is_add_m: bool = True):
        self._check_specs(id_spec, attr_spec)
        self.id_spec = id_spec
        self.attr_spec = attr_spec
        self.is_add_m = is_add_m  # M- M+ 사이에 M 행을 추가할지 여부.

    # ───────────────────── public API ─────────────────────
    def execute(
        self,
        df_old: pd.DataFrame,
        df_new: pd.DataFrame,
        baseline: Literal["old", "new"] = "old",
    ) -> pd.DataFrame:
        if baseline not in {"old", "new"}:
            raise ValueError('baseline must be "old" or "new"')

        old = df_old.copy()
        new = df_new.copy()

        cols = self._resolve_columns(old, new)  # dict
        diff = self._build_diff(old, new, cols["id_cols"], cols["attr_cols"], baseline)
        diff.drop(columns=cols["tmp_cols"], inplace=True)
        if self.is_add_m:
            diff = self._add_mid_row_for_diff(diff)
        return diff

    # ────────────────────── 내부 헬퍼 ──────────────────────
    @staticmethod
    def _neq(a, b) -> bool:
        """NaN 동등 취급"""
        return not (pd.isna(a) and pd.isna(b)) and a != b

    @staticmethod
    def _add_tmp(
        d1: pd.DataFrame,
        d2: pd.DataFrame,
        func: "DiffCallable",
        prefix: str,
    ) -> dict[str, str | list[str]]:
        base = f"__tmp_{prefix}"
        col = base
        k = 1
        while col in d1.columns or col in d2.columns:
            col = f"{base}_{k}"
            k += 1
        d1[col] = func(d1)
        d2[col] = func(d2)
        return {"col": col, "tmp_cols": [col]}

    # ───────── spec 검증 ─────────
    @staticmethod
    def _check_specs(id_spec: "DiffSpec", attr_spec: Optional["DiffSpec"]) -> None:
        def is_seq(obj):
            from collections.abc import Sequence

            return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))

        def val(spec, name, allow_none):
            if spec is None:
                if not allow_none:
                    raise TypeError(f"{name} cannot be None")
                return
            if callable(spec):
                return
            if not is_seq(spec) or not all(isinstance(x, str) for x in spec):
                raise TypeError(f"{name} must be Sequence[str] or callable")

        val(id_spec, "id_spec", False)
        val(attr_spec, "attr_spec", True)

        if isinstance(id_spec, Sequence) and isinstance(attr_spec, Sequence):
            dup = set(id_spec) & set(attr_spec)
            if dup:
                raise ValueError(f"id_spec overlaps attr_spec: {sorted(dup)}")

    # ───────── spec → cols ─────────
    def _resolve_spec(self, d1: pd.DataFrame, d2: pd.DataFrame, spec: "DiffSpec", prefix: str) -> dict[str, list[str]]:
        if callable(spec):
            info = self._add_tmp(d1, d2, spec, prefix)
            return {"cols": [info["col"]], "tmp": info["tmp_cols"]}
        return {"cols": list(spec), "tmp": []}

    def _resolve_columns(self, d1: pd.DataFrame, d2: pd.DataFrame) -> dict[str, list[str]]:
        tmp: list[str] = []

        id_info = self._resolve_spec(d1, d2, self.id_spec, "id")
        id_cols = id_info["cols"]
        tmp.extend(id_info["tmp"])

        if self.attr_spec is None:
            attr_cols = sorted((set(d1.columns) | set(d2.columns)) - set(id_cols))
        else:
            attr_info = self._resolve_spec(d1, d2, self.attr_spec, "attr")
            attr_cols = attr_info["cols"]
            tmp.extend(attr_info["tmp"])

        return {"id_cols": id_cols, "attr_cols": attr_cols, "tmp_cols": tmp}

    # ───────── LCS (patience-diff 방식) ─────────
    @staticmethod
    def _lcs(a: list[tuple], b: list[tuple]) -> list[tuple]:
        pos_in_a = {v: i for i, v in enumerate(a)}
        seq = [pos_in_a[v] for v in b if v in pos_in_a]

        piles: list[int] = []
        links: list[tuple[int, int, int | None]] = []

        for idx in seq:
            j = bisect_left(piles, idx)
            if j == len(piles):
                piles.append(idx)
            else:
                piles[j] = idx
            prev = links[-1][2] if links and j else None
            links.append((j, idx, prev))

        k = len(piles) - 1
        lcs: list[int] = []
        for j, idx, prev in reversed(links):
            if j == k:
                lcs.append(idx)
                k -= 1
            if prev is not None:
                links[j] = (j, prev, links[prev][2] if prev < len(links) else None)
            if k < 0:
                break
        lcs_idx = set(lcs[::-1])
        return [a[i] for i in lcs_idx]

    # ───────── diff 생성 ─────────
    @staticmethod
    def _reorder_by_id(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
        # ID 그룹 안에서 우선순위 정렬
        return (
            df.groupby(id_cols, sort=False, group_keys=False)
            .apply(lambda g: g.sort_values(by=CHANGE_COL, key=lambda s: s.map(CHANGE_PRIORITY)))
            .reset_index(drop=True)
        )

    def _build_diff(
        self,
        d_old: pd.DataFrame,
        d_new: pd.DataFrame,
        id_cols: list[str],
        attr_cols: list[str],
        baseline: Literal["old", "new"],
    ) -> pd.DataFrame:
        col_order = list(d_old.columns)
        for c in d_new.columns:
            if c not in col_order:
                col_order.append(c)

        id_set = set(id_cols)
        row_bucket: dict[tuple, dict[str, list]] = {}

        merged = d_old.merge(d_new, on=id_cols, how="outer", suffixes=("_old", "_new"), indicator=True)

        for _, r in merged.iterrows():
            key = tuple(r[id_cols])

            def build(side: Literal["old", "new"], tag: str) -> list:
                suff = f"_{side}"
                return [tag] + [r[c] if c in id_set else r.get(f"{c}{suff}", pd.NA) for c in col_order]

            if r["_merge"] == "left_only":
                row_bucket[key] = {"old": build("old", Change.DELETE)}
            elif r["_merge"] == "right_only":
                row_bucket[key] = {"new": build("new", Change.ADD)}
            else:
                changed = any(self._neq(r[f"{c}_old"], r[f"{c}_new"]) for c in attr_cols)
                if changed:
                    row_bucket[key] = {
                        "old": build("old", Change.MODIFIED_FROM),
                        "new": build("new", Change.MODIFIED_TO),
                    }
                else:
                    row_bucket[key] = {baseline: build(baseline, Change.EQUAL)}

        old_seq = [tuple(r[id_cols]) for _, r in d_old.iterrows()]
        new_seq = [tuple(r[id_cols]) for _, r in d_new.iterrows()]
        common = set(self._lcs(old_seq, new_seq))

        out: list[list] = []
        i = j = 0
        while i < len(old_seq) or j < len(new_seq):
            if i < len(old_seq) and old_seq[i] not in common:
                out.append(row_bucket[old_seq[i]]["old"])  # 삭제 D
                i += 1
            elif j < len(new_seq) and new_seq[j] not in common:
                out.append(row_bucket[new_seq[j]]["new"])  # 추가 A
                j += 1
            else:
                # 공통 키 → M- / M+ 또는 = (baseline)
                key = old_seq[i]  # == new_seq[j]
                rec = row_bucket[key]

                if "old" in rec:  # M-  or =
                    out.append(rec["old"])
                if "new" in rec:  # M+  or =
                    out.append(rec["new"])
                i += 1
                j += 1

        out_df = pd.DataFrame(out, columns=[CHANGE_COL] + col_order)

        # ID 그룹 안에서 우선순위 정렬 후 반환
        return self._reorder_by_id(out_df, id_cols)

    @staticmethod
    def _add_mid_row_for_diff(diff: pd.DataFrame) -> pd.DataFrame:
        """
        GitLikeDiff 의 추가 유틸 함수.
        M- · M+  사이에  'M'  행을 삽입.
        값이 실제로 달라진 셀만  "old -> new"  형식으로 채움.
        """
        out = []
        i = 0
        n = len(diff)
        change_col = CHANGE_COL

        while i < n:
            row = diff.iloc[i]
            out.append(row)

            # 바로 다음 행이 짝이 되는 M+ 라는 가정
            if (
                row[change_col] == Change.MODIFIED_FROM
                and i + 1 < n
                and diff.iloc[i + 1][change_col] == Change.MODIFIED_TO
            ):
                old_r = row
                new_r = diff.iloc[i + 1]

                mid = new_r.copy()  # 헤더·순서 유지
                mid[change_col] = Change.MODIFIED

                for col in diff.columns.difference([change_col]):
                    old_val = old_r[col]
                    new_val = new_r[col]
                    if (pd.isna(old_val) and pd.isna(new_val)) or old_val == new_val:
                        mid[col] = ""
                    else:
                        mid[col] = f"{old_val}{MODIFIED_SEP}{new_val}"

                out.append(mid)  # 요약 행 삽입
                # 이어서 M+ 도 자연히 out 에 추가됨 (루프 증가 전)

            i += 1

        return pd.DataFrame(out, columns=diff.columns)


if __name__ == "__main__":
    # 테스트 코드
    df_old = pd.read_excel(test_old_path)
    df_new = pd.read_excel(test_new_path)

    dd_diff_engine = GitLikeDiff(**dd_crf_spec)
    diff = dd_diff_engine.execute(df_old, df_new, baseline="old")

    print(diff)
    diff.to_excel(test_result_path, index=False)
