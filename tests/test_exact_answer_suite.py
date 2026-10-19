import json
from pathlib import Path

import pytest

from hive.contracts import SPLITS, AgentResult, Suite, Task, Usage
from hive.suites.exact_answer import ExactAnswerSuite

BUNDLED_TASKS = Path(__file__).resolve().parents[1] / "hive" / "suites" / "exact_answer" / "tasks.jsonl"


def completed(final_text: str) -> AgentResult:
    return AgentResult(status="completed", final_text=final_text, usage=Usage(), wall_clock_s=0.1)


def write_tasks(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def task_row(**overrides) -> dict:
    row = {"id": "t1", "split": "search", "question": "What is 2 + 2?", "answers": ["4"], "numeric": True}
    row.update(overrides)
    return row


def suite_with(tmp_path: Path, rows: list[dict]) -> ExactAnswerSuite:
    return ExactAnswerSuite({"tasks_file": str(write_tasks(tmp_path / "tasks.jsonl", rows))})


def only_task(suite: ExactAnswerSuite) -> Task:
    (task,) = [task for split in SPLITS for task in suite.tasks(split)]
    return task


def bundled_rows() -> list[dict]:
    return [json.loads(line) for line in BUNDLED_TASKS.read_text().splitlines() if line.strip()]


def grade_text(suite: ExactAnswerSuite, task: Task, text: str, tmp_path: Path):
    return suite.grade(task, completed(text), tmp_path)


# --- contract and bundled data ---


def test_suite_satisfies_the_suite_contract():
    suite = ExactAnswerSuite()
    assert isinstance(suite, Suite)
    assert suite.name == "exact_answer"


def test_constructor_accepts_none_and_empty_settings():
    assert ExactAnswerSuite(None).tasks("search") == ExactAnswerSuite({}).tasks("search")


def test_bundled_tasks_have_six_search_four_heldout_and_no_validation():
    suite = ExactAnswerSuite()
    assert len(suite.tasks("search")) == 6
    assert len(suite.tasks("heldout")) == 4
    assert suite.tasks("validation") == []


def test_bundled_splits_are_disjoint_by_task_id():
    suite = ExactAnswerSuite()
    ids_by_split = [{task.id for task in suite.tasks(split)} for split in SPLITS]
    all_ids = [task.id for split in SPLITS for task in suite.tasks(split)]
    assert len(all_ids) == len(set(all_ids))
    assert sum(len(ids) for ids in ids_by_split) == len(all_ids)


def test_tasks_carry_their_split_and_no_workspace_files():
    suite = ExactAnswerSuite()
    for split in SPLITS:
        for task in suite.tasks(split):
            assert task.split == split
            assert task.workspace_files == {}


def test_instruction_contains_the_question_and_the_answer_format():
    suite = ExactAnswerSuite()
    rows = {row["id"]: row for row in bundled_rows()}
    for task in suite.tasks("search"):
        assert rows[task.id]["question"] in task.instruction
        assert "ANSWER:" in task.instruction


def test_instruction_never_contains_an_accepted_answer_line():
    suite = ExactAnswerSuite()
    rows = {row["id"]: row for row in bundled_rows()}
    for split in SPLITS:
        for task in suite.tasks(split):
            for answer in rows[task.id]["answers"]:
                assert f"ANSWER: {answer}" not in task.instruction


def test_tasks_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="split"):
        ExactAnswerSuite().tasks("train")


# --- grader trust checks (P2) ---


def test_reference_answer_scores_one_on_every_bundled_task(tmp_path):
    suite = ExactAnswerSuite()
    rows = {row["id"]: row for row in bundled_rows()}
    for split in SPLITS:
        for task in suite.tasks(split):
            grade = grade_text(suite, task, f"ANSWER: {rows[task.id]['answers'][0]}", tmp_path)
            assert grade.reward == 1.0, (task.id, grade.detail)


def test_empty_final_text_scores_zero_on_every_bundled_task(tmp_path):
    suite = ExactAnswerSuite()
    for split in SPLITS:
        for task in suite.tasks(split):
            assert grade_text(suite, task, "", tmp_path).reward == 0.0


# --- answer extraction ---


def test_missing_answer_line_scores_zero_with_explanation(tmp_path):
    suite = suite_with(tmp_path, [task_row()])
    grade = grade_text(suite, only_task(suite), "I think it is 4.", tmp_path)
    assert grade.reward == 0.0
    assert grade.detail == 'no "ANSWER:" line found'


def test_last_answer_line_wins_over_earlier_ones(tmp_path):
    suite = suite_with(tmp_path, [task_row()])
    text = "ANSWER: 5\nWait, let me recheck.\nANSWER: 4"
    assert grade_text(suite, only_task(suite), text, tmp_path).reward == 1.0


def test_earlier_correct_answer_does_not_rescue_a_wrong_final_answer(tmp_path):
    suite = suite_with(tmp_path, [task_row()])
    text = "ANSWER: 4\nActually no.\nANSWER: 5"
    assert grade_text(suite, only_task(suite), text, tmp_path).reward == 0.0


def test_answer_marker_is_case_insensitive(tmp_path):
    suite = suite_with(tmp_path, [task_row()])
    assert grade_text(suite, only_task(suite), "answer: 4", tmp_path).reward == 1.0


def test_bold_markdown_marker_is_tolerated(tmp_path):
    suite = suite_with(tmp_path, [task_row()])
    assert grade_text(suite, only_task(suite), "**ANSWER:** 4", tmp_path).reward == 1.0


def test_backticks_around_the_whole_line_are_tolerated(tmp_path):
    suite = suite_with(tmp_path, [task_row()])
    assert grade_text(suite, only_task(suite), "`ANSWER: 4`", tmp_path).reward == 1.0


def test_answer_text_after_a_prefix_on_the_same_line_is_used(tmp_path):
    suite = suite_with(tmp_path, [task_row()])
    assert grade_text(suite, only_task(suite), "So my final ANSWER: 4", tmp_path).reward == 1.0


def test_empty_answer_after_marker_scores_zero(tmp_path):
    suite = suite_with(tmp_path, [task_row()])
    grade = grade_text(suite, only_task(suite), "ANSWER:\n4", tmp_path)
    assert grade.reward == 0.0
    assert grade.detail == 'parsed answer "": incorrect'


# --- normalization of text answers ---


def text_suite(tmp_path: Path) -> ExactAnswerSuite:
    return suite_with(tmp_path, [task_row(question="Who?", answers=["Carol Ann", "C. Ann"], numeric=False)])


def test_text_answer_match_ignores_case(tmp_path):
    suite = text_suite(tmp_path)
    assert grade_text(suite, only_task(suite), "ANSWER: CAROL ANN", tmp_path).reward == 1.0


def test_text_answer_match_strips_surrounding_quotes(tmp_path):
    suite = text_suite(tmp_path)
    assert grade_text(suite, only_task(suite), 'ANSWER: "Carol Ann"', tmp_path).reward == 1.0


def test_text_answer_match_strips_bold_asterisks_and_trailing_period(tmp_path):
    suite = text_suite(tmp_path)
    assert grade_text(suite, only_task(suite), "ANSWER: **Carol Ann**.", tmp_path).reward == 1.0


def test_text_answer_match_collapses_internal_whitespace(tmp_path):
    suite = text_suite(tmp_path)
    assert grade_text(suite, only_task(suite), "ANSWER:   Carol \t  Ann  ", tmp_path).reward == 1.0


def test_any_accepted_answer_scores_one(tmp_path):
    suite = text_suite(tmp_path)
    assert grade_text(suite, only_task(suite), "ANSWER: c. ann", tmp_path).reward == 1.0


def test_text_answer_with_extra_words_scores_zero(tmp_path):
    suite = text_suite(tmp_path)
    assert grade_text(suite, only_task(suite), "ANSWER: Carol Ann owns it", tmp_path).reward == 0.0


def test_correct_detail_reports_the_parsed_answer(tmp_path):
    suite = text_suite(tmp_path)
    grade = grade_text(suite, only_task(suite), "ANSWER: Carol Ann", tmp_path)
    assert grade.detail == 'parsed answer "carol ann": correct'


# --- numeric answers ---


def numeric_suite(tmp_path: Path, answer: str = "1234.5") -> ExactAnswerSuite:
    return suite_with(tmp_path, [task_row(answers=[answer])])


def test_numeric_answer_compares_as_a_number(tmp_path):
    suite = numeric_suite(tmp_path)
    assert grade_text(suite, only_task(suite), "ANSWER: 1234.50", tmp_path).reward == 1.0


def test_numeric_answer_accepts_thousands_separators(tmp_path):
    suite = numeric_suite(tmp_path)
    assert grade_text(suite, only_task(suite), "ANSWER: 1,234.5", tmp_path).reward == 1.0


def test_numeric_answer_rejects_misplaced_commas(tmp_path):
    suite = numeric_suite(tmp_path, answer="15")
    assert grade_text(suite, only_task(suite), "ANSWER: 1,5", tmp_path).reward == 0.0


def test_numeric_answer_within_relative_tolerance_scores_one(tmp_path):
    suite = numeric_suite(tmp_path, answer="1000000")
    assert grade_text(suite, only_task(suite), "ANSWER: 1000000.5", tmp_path).reward == 1.0


def test_numeric_answer_outside_tolerance_scores_zero(tmp_path):
    suite = numeric_suite(tmp_path, answer="2")
    assert grade_text(suite, only_task(suite), "ANSWER: 2.001", tmp_path).reward == 0.0


def test_numeric_answer_with_units_scores_zero(tmp_path):
    suite = numeric_suite(tmp_path, answer="42")
    assert grade_text(suite, only_task(suite), "ANSWER: 42 apples", tmp_path).reward == 0.0


def test_numeric_answer_with_currency_symbol_scores_zero(tmp_path):
    suite = numeric_suite(tmp_path, answer="10")
    assert grade_text(suite, only_task(suite), "ANSWER: $10", tmp_path).reward == 0.0


def test_numeric_answer_rejects_python_only_number_spellings(tmp_path):
    suite = numeric_suite(tmp_path, answer="1000")
    assert grade_text(suite, only_task(suite), "ANSWER: 1_000", tmp_path).reward == 0.0


def test_numeric_answer_accepts_negative_numbers(tmp_path):
    suite = numeric_suite(tmp_path, answer="-3")
    assert grade_text(suite, only_task(suite), "ANSWER: -3.0", tmp_path).reward == 1.0


# --- feedback never leaks the expected answer ---


def test_incorrect_detail_reports_the_parsed_answer(tmp_path):
    suite = numeric_suite(tmp_path, answer="388")
    grade = grade_text(suite, only_task(suite), "ANSWER: 17", tmp_path)
    assert grade.detail == 'parsed answer "17": incorrect'


def test_incorrect_detail_never_contains_any_expected_answer(tmp_path):
    suite = ExactAnswerSuite()
    rows = {row["id"]: row for row in bundled_rows()}
    for split in SPLITS:
        for task in suite.tasks(split):
            grade = grade_text(suite, task, "ANSWER: zzz-wrong", tmp_path)
            assert grade.reward == 0.0
            for answer in rows[task.id]["answers"]:
                assert answer.lower() not in grade.detail.lower(), task.id


def test_grade_rejects_a_task_from_another_suite(tmp_path):
    suite = ExactAnswerSuite()
    stranger = Task(id="not-a-real-task", split="search", instruction="?")
    with pytest.raises(ValueError, match="not-a-real-task"):
        grade_text(suite, stranger, "ANSWER: 1", tmp_path)


# --- describe and metric ---


def test_describe_explains_the_answer_format_and_grading():
    description = ExactAnswerSuite().describe()
    assert "ANSWER:" in description
    assert "accuracy" in description.lower()


def test_describe_contains_no_question_text():
    description = ExactAnswerSuite().describe()
    for row in bundled_rows():
        assert row["question"] not in description, row["id"]


def test_describe_contains_no_answer_of_three_or_more_characters():
    description = ExactAnswerSuite().describe().lower()
    for row in bundled_rows():
        for answer in row["answers"]:
            if len(answer) >= 3:
                assert answer.lower() not in description, (row["id"], answer)


def test_metric_is_accuracy_with_a_description():
    metric = ExactAnswerSuite().metric()
    assert metric.name == "accuracy"
    assert metric.description.strip()


# --- data validation ---


def test_duplicate_ids_raise_naming_the_line(tmp_path):
    with pytest.raises(ValueError, match=r":2:.*duplicate"):
        suite_with(tmp_path, [task_row(), task_row()])


def test_invalid_split_raises_naming_the_line(tmp_path):
    with pytest.raises(ValueError, match=r":1:.*split"):
        suite_with(tmp_path, [task_row(split="train")])


def test_empty_question_raises_naming_the_line(tmp_path):
    with pytest.raises(ValueError, match=r":1:.*question"):
        suite_with(tmp_path, [task_row(question="   ")])


def test_empty_answers_list_raises_naming_the_line(tmp_path):
    with pytest.raises(ValueError, match=r":1:.*answers"):
        suite_with(tmp_path, [task_row(answers=[])])


def test_blank_answer_string_raises_naming_the_line(tmp_path):
    with pytest.raises(ValueError, match=r":1:.*answers"):
        suite_with(tmp_path, [task_row(answers=["  "])])


def test_non_numeric_answer_on_numeric_task_raises_naming_the_line(tmp_path):
    with pytest.raises(ValueError, match=r":1:.*number"):
        suite_with(tmp_path, [task_row(answers=["four"])])


def test_non_boolean_numeric_flag_raises_naming_the_line(tmp_path):
    with pytest.raises(ValueError, match=r":1:.*numeric"):
        suite_with(tmp_path, [task_row(numeric="yes")])


def test_missing_field_raises_naming_the_line(tmp_path):
    row = task_row()
    del row["answers"]
    with pytest.raises(ValueError, match=r":1:.*answers"):
        suite_with(tmp_path, [row])


def test_unknown_field_raises_naming_the_line(tmp_path):
    with pytest.raises(ValueError, match=r":1:.*answer\b"):
        suite_with(tmp_path, [task_row(answer="4")])


def test_malformed_json_raises_naming_the_line(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(task_row()) + "\n{not json\n")
    with pytest.raises(ValueError, match=r":2:.*JSON"):
        ExactAnswerSuite({"tasks_file": str(path)})


def test_non_object_line_raises_naming_the_line(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match=r":1:.*object"):
        ExactAnswerSuite({"tasks_file": str(path)})


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n" + json.dumps(task_row()) + "\n\n")
    assert [task.id for task in ExactAnswerSuite({"tasks_file": str(path)}).tasks("search")] == ["t1"]


def test_file_with_no_tasks_raises(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no tasks"):
        ExactAnswerSuite({"tasks_file": str(path)})


def test_tasks_file_setting_overrides_the_bundled_tasks(tmp_path):
    suite = suite_with(tmp_path, [task_row(id="custom", split="validation")])
    assert [task.id for task in suite.tasks("validation")] == ["custom"]
    assert suite.tasks("search") == []
