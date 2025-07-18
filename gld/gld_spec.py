import pandas as pd

def make_crf_id(df: pd.DataFrame) -> pd.Series:
    """
    description  ➜  question__uid
    그 외        ➜  uid|sequence
    DD 파일의 CRF 시트용도
    """
    is_desc = df["question__type"] == "description"
    id_key = pd.Series(index=df.index, dtype=object)

    id_key[is_desc] = df.loc[is_desc, "question__uid"].astype(str)
    id_key[~is_desc] = df.loc[~is_desc, "uid"].astype(str) + "|" + df.loc[~is_desc, "sequence"].astype(str)
    return id_key


# DD CRF 시트
dd_crf_spec = {
    "id_spec": make_crf_id,
    "attr_spec": [
        # page
        "page__uid",
        "page__name",
        "page__is_enroll",
        # form
        "form__uid",
        "form__name",
        # question
        "question__uid",
        "question__name",
        "question__name",
        "question__type",
        "question__description_value",
        "question__is_vertical",
        # field
        "name",
        "type",
        "is_missing_query",
        "max_length",
        "decimal_places",
        "is_autocalculated",
        "choice_group__uid",
        "constant_value",
        "is_signed",
        "uk_limit",
        "time_format",
        "placeholder",
        "is_vertical",
        "width",
        "viewer",
        "max_file_size",
        "tag__uid",
    ],
}

# DD ChoiceGroup 시트
dd_choice_group_spec = {
    "id_spec": ["choice_group__uid", "choice__value"],
    "attr_spec": ["choice_group__name", "choice__name", "choice__score"],
}

# DD Tag 시트
dd_tag_spec = {
    "id_spec": ["uid"],
    "attr_spec": ["description", "color", "is_display"],
}


# DD Visit 시트
dd_visit_spec = {
    "id_spec": ["visit__uid"],
    "attr_spec": [
        "cycle__name",
        "cycle__uid",
        "cycle__is_primary",  # cycle
        "visit_group__name",
        "visit_group__uid",
        "visit_group__is_primary",
        "visit_group__is_repeatable",  # visit group
        "visit__name",
        "visit__is_primary",  # visit
    ],
}