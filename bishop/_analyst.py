"""
Finding all the functions in the pd. namespace:

pd_functions = [name for name, obj in inspect.getmembers(pd, inspect.isfunction)]
len(pd_functions)

All methods in the pandas.DataFrame object:

df_functions = [name for name, obj in inspect.getmembers(df, predicate=inspect.ismethod)]
len(df_functions)
"""

import dspy



PD_WHITELIST = ['array', 'bdate_range', 'concat', 'crosstab', 'cut', 'date_range',
                'factorize', 'from_dummies', 'get_dummies', 'infer_freq', 'interval_range',
                'isna', 'isnull', 'json_normalize', 'lreshape', 'melt', 'merge', 'merge_asof',
                'merge_ordered', 'notna', 'notnull', 'period_range', 'pivot', 'pivot_table', 
                'qcut', 'timedelta_range', 'to_datetime', 'to_numeric', 'to_timedelta',
                'unique', 'value_counts', 'wide_to_long']

DF_WHITELIST = ['abs', 'add', 'add_prefix', 'add_suffix', 'agg', 'aggregate', 'align', 'all',
                'any', 'asfreq', 'asof', 'assign', 'astype', 'at_time', 'backfill', 'between_time',
                'bfill', 'bool', 'clip', 'combine', 'combine_first', 'compare', 'convert_dtypes', 
                'corr', 'corrwith', 'count', 'cov', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe',
                'diff', 'div', 'divide', 'dot', 'drop', 'drop_duplicates', 'droplevel', 'dropna',
                'duplicated', 'eq', 'equals', 'ewm', 'expanding', 'explode', 'ffill', 'fillna', 'filter',
                'first', 'first_valid_index', 'floordiv', 'ge', 'get', 'groupby', 'gt', 'head', 'hist',
                'idxmax', 'idxmin', 'infer_objects', 'info', 'insert', 'interpolate', 'isetitem', 'isin',
                'isna', 'isnull', 'items', 'iterrows', 'itertuples', 'join', 'keys', 'kurt', 'kurtosis', 'last', 
                'last_valid_index', 'le', 'lt', 'mask', 'max', 'mean', 'median', 'melt', 'memory_usage', 'merge',
                'min', 'mod', 'mode', 'mul', 'multiply', 'ne', 'nlargest', 'notna', 'notnull', 'nsmallest', 
                'nunique', 'pad', 'pct_change', 'pivot', 'pivot_table', 'pop', 'pow', 'prod', 'product', 'quantile',
                'query', 'radd', 'rank', 'rdiv', 'reindex', 'reindex_like', 'rename', 'rename_axis', 
                'reorder_levels', 'replace', 'resample', 'reset_index', 'rfloordiv', 'rmod', 'rmul', 'rolling',
                'round', 'rpow', 'rsub', 'rtruediv', 'sample', 'select_dtypes', 'sem', 'set_axis', 'set_flags',
                'set_index','shift', 'skew', 'sort_index', 'sort_values', 'squeeze', 'stack', 'std', 'sub',
                'subtract', 'sum', 'swapaxes', 'swaplevel', 'tail', 'take', 'transform', 'transpose', 'truediv',
                'truncate', 'tz_convert', 'tz_localize', 'unstack', 'update', 'value_counts', 'var', 'where', 'xs']

def return_pandas_query_tool(df, strict=False):
    def pandas_query(command:str) -> str:
        """
        Call any function in the pandas API, starting with 'pd.' or 'df.' on the 
        dataframe 'df' and retrieve results. lambda functions and df.eval() are 
        not allowed!! The code will be run using eval(), so intermediate steps 
        like assigning a new column won't work.
        """
        _ = df.describe()
        exec("import pandas as pd")
        print("\ncommand:", command)
        if ("lambda" in command.lower())|("pd.eval" in command.lower()):
            result = "command not allowed!"
        elif ";" in command:
            result = "multi-line commands not allowed!"
        elif not command.startswith("pd.")|command.startswith("df"):
            result ="command not allowed! must start with `pd.` or `df.`"
        elif command.startswith("pd.read")|command.startswith("pd.to_"):
            result = "not allowed to read from or write to disk"
        else:
            try:
                allowed = True
                # strict case: only permit whitelisted function calls
                if strict:
                    # split out every case that looks like pd.FUNCTION(), df.FUNCTION(), 
                    # df["colum_name"].FUNCTION(), etc
                    for f in command.split(".")[1:]:
                        func = f.split("(")[0]
                        if func not in PD_WHITELIST+DF_WHITELIST:
                             allowed = False
                             result = f"function {func} not permitted"
                if allowed:
                    result = eval(command)
            except Exception as e:
                result = f"Command failed with this error: {e}"
        print("result:", result)
        return result
    return pandas_query


class AnalystSig(dspy.Signature):
    """
    You are a curious and rigorous AI scientist, specializing in data analysis. It is your
    ethical and professional duty to pose difficult questions and challenge assumptions.

    Input a question about a dataset along with the background/context for the question. Return 
    your best answer, including caveats and an explanation of your confidence in the 
    assessment. You can use your python skills to analyze the dataset using the pandas API.
    """
    background:str = dspy.InputField()
    question:str = dspy.InputField()
    description:str = dspy.InputField(desc="df.describe()")
    answer:str = dspy.OutputField()


def get_analyst(df, strict=True):
    """
    """
    return dspy.ReAct(AnalystSig, tools=[return_pandas_query_tool(df.copy(), strict=strict)])