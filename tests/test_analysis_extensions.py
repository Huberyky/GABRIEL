import pandas as pd

from gabriel.analysis import reliability, validate, robustness
from gabriel.utils.model_utils import strip_reasoning_tags


def test_strip_reasoning_tags_removes_think_blocks():
    payload = '<think>internal</think>{"score": 5}'
    assert strip_reasoning_tags(payload) == '{"score": 5}'


def test_reliability_computes_basic_metrics():
    df = pd.DataFrame(
        {
            'id': ['a', 'a', 'b', 'b'],
            'run': [1, 2, 1, 2],
            'score': [10, 12, 80, 82],
        }
    )
    out = reliability(df, value_cols=['score'])
    assert out.loc[0, 'attribute'] == 'score'
    assert out.loc[0, 'n_runs'] == 2


def test_validate_continuous_and_categorical():
    cont = pd.DataFrame({'gpt': [1, 2, 3], 'human': [1, 2, 4], 'bucket': ['s', 's', 'l']})
    report = validate(cont, gpt_col='gpt', human_col='human', task_type='continuous', strata=['bucket'])
    assert 'metrics' in report and 'stratified' in report

    cat = pd.DataFrame({'gpt': ['a', 'b', 'a'], 'human': ['a', 'b', 'b']})
    report2 = validate(cat, gpt_col='gpt', human_col='human', task_type='categorical')
    assert 'confusion_matrix' in report2


def test_robustness_bootstrap_summary():
    df = pd.DataFrame({'score': [1, 2, 1.5, 2.5], 'group': ['p1', 'p1', 'p2', 'p2']})
    out = robustness(df, score_col='score', group_col='group', reference_group='p1', n_bootstrap=10)
    assert 'group_summary' in out
    assert 'vs_reference' in out
