"""
git_like_diff.py  –  path 제거 + dict 반환 + LCS(최장 공통 부분열) 기반 정렬

• 결과 컬럼   : change( "A" | "D" | "M-" | "M+" | "" )
• 정렬 규칙   : LCS 순서에 따라 삭제 블록 뒤에 바로 추가 블록이 오도록
• 성능        :  Patience-diff O((m+n) log n) / O(n) 메모리
"""

from bisect import bisect_left
from typing  import Callable, List, Literal, Sequence, Tuple, Union, Optional, Dict

import pandas as pd

from gld.const import test_new_path, test_old_path, test_result_path
from gld.gld_spec import dd_crf_spec

DiffCallable = Callable[[pd.DataFrame], pd.Series]
DiffSpec     = Union[Sequence[str], DiffCallable]


class GitLikeDiff:
    CHANGE_COL = "__CHANGE__"
    ADD, DELETE, MODIFIED_FROM, MODIFIED_TO, EQUAL = "A", "D", "M-", "M+", ""

    # ───────────────────── 초기화 ─────────────────────
    def __init__(self, id_spec: DiffSpec, attr_spec: Optional[DiffSpec] = None):
        self._check_specs(id_spec, attr_spec)
        self.id_spec   = id_spec
        self.attr_spec = attr_spec

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

        cols = self._resolve_columns(old, new)        # dict
        diff = self._build_diff(
            old, new,
            cols["id_cols"], cols["attr_cols"],
            baseline
        )
        diff.drop(columns=cols["tmp_cols"], inplace=True)
        return diff

    # ────────────────────── 내부 헬퍼 ──────────────────────
    @staticmethod
    def _neq(a, b) -> bool:
        """NaN 동등 취급"""
        return not (pd.isna(a) and pd.isna(b)) and a != b

    @staticmethod
    def _add_tmp(
        d1: pd.DataFrame, d2: pd.DataFrame,
        func: DiffCallable, prefix: str,
    ) -> Dict[str, Union[str, List[str]]]:
        base = f"__tmp_{prefix}"
        col  = base
        k    = 1
        while col in d1.columns or col in d2.columns:
            col = f"{base}_{k}"
            k += 1
        d1[col] = func(d1)
        d2[col] = func(d2)
        return {"col": col, "tmp_cols": [col]}

    # ───────── spec 검증 ─────────
    @staticmethod
    def _check_specs(id_spec: DiffSpec, attr_spec: Optional[DiffSpec]) -> None:
        def is_seq(obj):
            from typing import Sequence
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
    def _resolve_spec(
        self, d1: pd.DataFrame, d2: pd.DataFrame, spec: DiffSpec, prefix: str
    ) -> Dict[str, List[str]]:
        if callable(spec):
            info = self._add_tmp(d1, d2, spec, prefix)
            return {"cols": [info["col"]], "tmp": info["tmp_cols"]}
        return {"cols": list(spec), "tmp": []}

    def _resolve_columns(self, d1: pd.DataFrame, d2: pd.DataFrame) -> Dict[str, List[str]]:
        tmp: List[str] = []

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
    def _lcs(a: List[Tuple], b: List[Tuple]) -> List[Tuple]:
        pos_in_a = {v: i for i, v in enumerate(a)}
        seq = [pos_in_a[v] for v in b if v in pos_in_a]

        piles: List[int] = []
        links: List[Tuple[int, int, Optional[int]]] = []

        for idx in seq:
            j = bisect_left(piles, idx)
            if j == len(piles):
                piles.append(idx)
            else:
                piles[j] = idx
            prev = links[-1][2] if links and j else None
            links.append((j, idx, prev))

        k = len(piles) - 1
        lcs: List[int] = []
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
    def _build_diff(
        self,
        d_old: pd.DataFrame,
        d_new: pd.DataFrame,
        id_cols: List[str],
        attr_cols: List[str],
        baseline: Literal["old", "new"],
    ) -> pd.DataFrame:

        col_order = list(d_old.columns)
        for c in d_new.columns:
            if c not in col_order:
                col_order.append(c)

        id_set = set(id_cols)
        row_bucket: Dict[Tuple, Dict[str, List]] = {}

        merged = d_old.merge(
            d_new, on=id_cols,
            how="outer", suffixes=("_old", "_new"), indicator=True
        )

        for _, r in merged.iterrows():
            key = tuple(r[id_cols])

            def build(side: Literal["old", "new"], tag: str) -> List:
                suff = f"_{side}"
                return [tag] + [
                    r[c] if c in id_set else r.get(f"{c}{suff}", pd.NA)
                    for c in col_order
                ]

            if r["_merge"] == "left_only":
                row_bucket[key] = {"old": build("old", self.DELETE)}
            elif r["_merge"] == "right_only":
                row_bucket[key] = {"new": build("new", self.ADD)}
            else:
                changed = any(self._neq(r[f"{c}_old"], r[f"{c}_new"]) for c in attr_cols)
                if changed:
                    row_bucket[key] = {
                        "old": build("old", self.MODIFIED_FROM),
                        "new": build("new", self.MODIFIED_TO),
                    }
                else:
                    row_bucket[key] = {baseline: build(baseline, self.EQUAL)}

        old_seq = [tuple(r[id_cols]) for _, r in d_old.iterrows()]
        new_seq = [tuple(r[id_cols]) for _, r in d_new.iterrows()]
        common  = set(self._lcs(old_seq, new_seq))

        out: List[List] = []
        i = j = 0
        while i < len(old_seq) or j < len(new_seq):
            if i < len(old_seq) and old_seq[i] not in common:
                out.append(row_bucket[old_seq[i]]["old"])
                i += 1
            elif j < len(new_seq) and new_seq[j] not in common:
                out.append(row_bucket[new_seq[j]]["new"])
                j += 1
            else:
                key = old_seq[i]  # == new_seq[j] == common
                rec = row_bucket[key]
                if "old" in rec:
                    out.append(rec["old"])
                if "new" in rec:
                    out.append(rec["new"])
                i += 1
                j += 1

        return pd.DataFrame(out, columns=[self.CHANGE_COL] + col_order)


if __name__ == "__main__":
    # 테스트 코드
    df_old = pd.read_excel(test_old_path)
    df_new = pd.read_excel(test_new_path)

    dd_diff_engine = GitLikeDiff(**dd_crf_spec)
    diff = dd_diff_engine.execute(df_old, df_new, baseline="old")

    print(diff)
    diff.to_excel(test_result_path, index=False)
