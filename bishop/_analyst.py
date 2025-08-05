"""
Finding all the functions in the pd. namespace:

pd_functions = [name for name, obj in inspect.getmembers(pd, inspect.isfunction)]
len(pd_functions)

All methods in the pandas.DataFrame object:

df_functions = [name for name, obj in inspect.getmembers(df, predicate=inspect.ismethod)]
len(df_functions)
"""
import pandas as pd
import dspy
import typing



PD_WHITELIST = ['array', 'bdate_range', 'concat', 'crosstab', 'cut', 'date_range',
                'factorize', 'from_dummies', 'get_dummies', 'infer_freq', 'interval_range',
                'isna', 'isnull', 'json_normalize', 'lreshape', 'melt', 'merge', 'merge_asof',
                'merge_ordered', 'notna', 'notnull', 'period_range', 'pivot', 'pivot_table', 
                'qcut', 'timedelta_range', 'to_datetime', 'to_numeric', 'to_timedelta',
                'unique', 'value_counts', 'wide_to_long']

DF_WHITELIST = ['abs', 'add', 'add_prefix', 'add_suffix', 'agg', 'aggregate', 'align', 'all',
                'any', 'asfreq', 'asof', 'astype', 'at_time', 'backfill', 'between_time',
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


foo = """
        Call any function in the pandas API, starting with 'pd.' or 'df.' on the 
        dataframe 'df' and retrieve results. lambda functions and df.eval() are 
        not allowed!! 
        The query must be a single call to the pandas API; multiple lines combined
        with a semicolon will return an error. Intermediate steps like assigning a 
        new column won't work.
        """

def return_pandas_query_tool(df, strict=False):
    def pandas_query(command:str) -> str:
        """
        Query the dataset using pandas
        """
        
        _ = df.describe()
        exec("import pandas as pd")
        print("\ncommand:", command)
        if "import " in command.lower():
            result = "FAIL: import statements not allowed!"
        elif "lambda" in command.lower():
            result = "FAIL: command 'lambda' not allowed!"
        elif "pd.eval" in command.lower():
            result = "FAIL: command 'eval' not allowed!"
        elif "np." in command:
            result = "FAIL: numpy calls not allowed; only pandas"
        #if ("lambda" in command.lower())|("pd.eval" in command.lower()):
        #    result = "command not allowed!"
        elif ";" in command:
            result = "FAIL: multiple statements not allowed!"
        # remove "(" for this test because "((df['y'] - df['x']**2)**2).mean()" is OK
        elif not command.replace("(","").startswith("pd.")|command.replace("(","").startswith("df"):
            result ="FAIL: command not allowed! must start with `pd.` or `df.`"
        elif command.startswith("pd.read")|command.startswith("pd.to_"):
            result = "FAIL: not allowed to read from or write to disk"
        else:
            try:
                allowed = True
                # strict case: only permit whitelisted function calls
                if strict:
                    # split out every case that looks like pd.FUNCTION(), df.FUNCTION(), 
                    # df["colum_name"].FUNCTION(), etc
                    for f in command.split(".")[1:]:
                        # if it's a function call there should be a parenthesis- without this
                        # check, the function will flag on decimals
                        if "(" in f:
                            func = f.split("(")[0]
                            # make sure we're not flagging on decimal numbers
                            #if not all(char.isdigit() for char in func):
                            if func not in PD_WHITELIST+DF_WHITELIST:
                                allowed = False
                                result = f"FAIL: function {func} not permitted"
                if allowed:
                    result = eval(command)
            except Exception as e:
                result = f"FAIL: Command failed with this error: {e}"
        print("result:", result)
        return result
    return pandas_query


def _pandas_query(command:str, df:pd.core.frame.DataFrame, strict:bool=True, maxlines:int=15) -> str:
    """
    Query the dataset using pandas
    """
    failures = []
    _ = df.describe() # still need this?
    exec("import pandas as pd") # still need this?
    print("\ncommand:", command)
    if "import" in command.lower():
        failures.append("import statements not allowed")
    if "lambda" in command.lower():
        failures.append("lambda commands not allowed")
    if "pd.eval" in command.lower():
        failures.append("eval comands not allowed")
    if "np." in command:
        failures.append("numpy commands not allowed; only pandas")
    if ";" in command:
        failures.append("multiple lines connected by `;` not allowed")
    if not command.replace("(","").startswith("pd.")|command.replace("(","").startswith("df"):
        failures.append("command not allowed! must start wtih `pd.` or `df.`")
    if command.startswith("pd.read")|command.startswith("pd.to_"):
        failures.append("not allowed to read from or write to disk")
    # strict mode- find everything that looks like a function and check to 
    # see if it's whitelisted
    if strict:
        # split out every case that looks like pd.FUNCTION(), df.FUNCTION(), 
        # df["colum_name"].FUNCTION(), etc
        for f in command.split(".")[1:]:
            # if it's a function call there should be a parenthesis- without this
            # check, the function will flag on decimals
            if "(" in f:
                func = f.split("(")[0]
                if func not in PD_WHITELIST+DF_WHITELIST:
                    failures.append(f"function {func} not permitted")
    if len(failures) == 0:
        try:
            result = eval(command)
            if len(str(result).split("\n")) > maxlines:
                result = f"WARNING: result too long; truncating to {maxlines} lines. Please try a different query.\n{result.head(maxlines)}"
        except Exception as e:
            failures.append(f"error: {e}")
    if len(failures) > 0:
        result = "Command failed for the following reasons:"
        for f in failures:
            result += f"\n* {f}"
        result += "\n**Please reframe your query or ask a different question.**"
    return result


class AnalystSig(dspy.Signature):
    """
    You are a curious and rigorous AI scientist, specializing in data analysis. It is your
    ethical and professional duty to pose difficult questions and challenge assumptions.

    Input a question about a dataset along with the background/context for the question. Return 
    your best answer, including caveats and an explanation of your confidence in the 
    assessment. You can use your python skills to analyze the dataset using single-line calls
    to the pandas API.

    If you get a "Command failed" error, try asking a different question! Do not import anything,
    do not try to make multiple calls with a ";", don't create new columns, and do not use 
    "eval" or "lambda".

    Example acceptable calls:
    df["foo"].mean()
    df.groupby(["foo", "bar"]).size()

    Example unacceptable calls:
    df["bar"] = df["foo"]+1
    data = df[["foo", "bar"]]; means = data.mean()
    df["foo"].plot()
    df["foo"].apply(lambda x: x**2)
    """
    background:str = dspy.InputField()
    question:str = dspy.InputField()
    description:str = dspy.InputField(desc="df.describe()")
    answer:str = dspy.OutputField()


def get_analyst(df, strict=True, max_iters=25):
    """
    Return a dspy ReAct agent that can answer analytic questions about your dataset.

    :df: pandas DataFrame; your dataset
    :strict: bool; whether to enforce a set of whitelisted pandas functions
    :max_iters: int; max number of times the agent can query the dataset
    """
    return dspy.ReAct(AnalystSig, tools=[return_pandas_query_tool(df.copy(), strict=strict)], 
                      max_iters=max_iters)


class Analyst(dspy.Module):
    """
    General-purpose data analysis agent
    """
    def __init__(self, max_iters:int=25, strict:bool=True,
                 df:typing.Union[None,pd.core.frame.DataFrame]=None,
                 verbose:bool=False):
        """
        :max_iters: max number of ReAct iterations to query dataset for analysis
        :strict: if True, only permit explicitly whitelisted pandas functions
        :df: pandas DataFrame to use for analysis
        :verbose: if True, print out each stage of analysis
        """
        self.max_iters = max_iters
        self.strict = strict
        self.df = df
        self.verbose = verbose
        self.react = dspy.ReAct(AnalystSig, tools=[self.pandas_query], 
                      max_iters=max_iters)
        
    def set_dataframe(self, df=pd.core.frame.DataFrame):
        self.df = df

    def pandas_query(self, command:str) -> str:
        """
        Use a single line of pandas code to probe the dataset
        """
        if self.verbose:
            print(f"analyst command: {command}")
        result = _pandas_query(command, self.df, strict=self.strict)
        # this is where we could potentially change the output when the LLM
        # keeps repeating a failed query
        if self.verbose:
            print(f"analyst response: {result}")
        return result
    
    def forward(self, question:str, background:str="None", 
                df:typing.Union[None,pd.core.frame.DataFrame]=None, **kwargs) -> dspy.Prediction:
        """
        do analysis
        """
        if df is not None:
            self.set_dataframe(df)
          
        return self.react(question=question,
                          background=background, 
                          description=self.df.describe().to_markdown())
