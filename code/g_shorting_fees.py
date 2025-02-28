#%%
# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.lines as mlines  # For custom legend handles


# Random pip below for specific pip install
# C:\Users\Wiingaard\AppData\Local\Programs\Python\Python311\python.exe -m pip install seaborn

# Create mean and median indicative_fee plots for each dcbs group? Set True or False
plots = False

# Global output directory and pickle file names
Output_Dir = r"C:\Master"
Unfiltered_PKL = "short_fees_unfiltered.pkl"
Filtered_PKL = "short_fees_filtered.pkl"
Linked_PKL = "short_fees_linked.pkl"
Processed_PKL = "sf_last.pkl"
Merged_Analysis_PKL = "merged_analysis.pkl"


#%% 
# Shorting fee data -----------------------------------------------------
def load_short_fees(data_path, file_name):
    """
    Load and filter shorting fee data, keeping all columns.
    Renames columns to more convenient names.
    
    Parameters:
        data_path (str): Path to the directory containing the file.
        file_name (str): Name of the short fees CSV file.
    
    Returns:
        pd.DataFrame: Unfiltered DataFrame with renamed columns
                      and a 'date' column in YYYY-MM-DD format (string).
    """
    import os
    import pandas as pd

    file_path = os.path.join(data_path, file_name)
    short_fees = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=False)

    col_rename = {
        "Date": "date",
        "DXL Identifier": "dxl_identifier",
        "ISIN": "isin",
        "SEDOL": "sedol",
        "CUSIP": "cusip",
        "Stock Description": "stock_description",
        "Market Area (Country level)": "market_area",
        "Record Type": "record_type",
        "Open Loan Transactions": "open_loan_transactions",
        "SAF": "saf",
        "SAR": "sar",
        "DCBS": "dcbs",
        "Indicative Fee": "indicative_fee",
        "Indicative Rebate": "indicative_rebate"
    }
    short_fees.rename(columns=col_rename, inplace=True)
    short_fees["date"] = short_fees["date"].dt.strftime("%Y-%m-%d")

    # Generate pickle file
    out_path = os.path.join(Output_Dir, Unfiltered_PKL)
    short_fees.to_pickle(out_path)
    print(f"Saved unfiltered short_fees to {out_path}")

    return short_fees


#%%
def short_fees_filters(short_fees):
    """
    Filter short fees data:
      - Keep rows where 'dcbs' is not null,
      - 'record_type' == 1,
      - 'market_area' == "USA Equity" (or change as needed).
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered = short_fees[
        short_fees["dcbs"].notna() &
        (short_fees["record_type"] == 1) &
        (short_fees["market_area"] == "USA Equity")
    ]

    # Generate pickle file
    out_path = os.path.join(Output_Dir, "short_fees_filtered.pkl")
    filtered.to_pickle(out_path)
    print(f"Saved filtered short_fees to {out_path}")

    return filtered


#%%
if True:
    def short_fees_link(short_fees_filtered):
        """
        Link each row in short_fees_filtered to a CRSP permno via CUSIP.
        
        Steps:
          1. Read the CRSP_Compustat_Mapping file.
          2. Merge on 'cusip' to find the corresponding 'LPERMNO' (CRSP permno).
          3. Rename 'LPERMNO' to 'permno'.
          4. Convert 'permno' to string.
        
        Returns:
            pd.DataFrame: DataFrame with a new 'permno' column.
        """
        import os
        import pandas as pd

        mapping_dir = r"C:\Master"
        mapping_file = "CRSP_Compustat_Mapping.csv"
        mapping_path = os.path.join(mapping_dir, mapping_file)
        mapping_df = pd.read_csv(mapping_path, usecols=["cusip", "LPERMNO"])

        linked = short_fees_filtered.merge(
            mapping_df,
            how="left",
            on="cusip"
        )
        linked.rename(columns={"LPERMNO": "permno"}, inplace=True)
        linked["permno"] = linked["permno"].astype(str)

        # Generate pickle file
        out_path = os.path.join(Output_Dir, "short_fees_linked.pkl")
        linked.to_pickle(out_path)
        print(f"Saved linked short_fees to {out_path}")

        return linked


#%%
def short_fees_2(short_fees_linked):
    """
    Further process the linked short fees data:
      - Compute 'sfee': If 'indicative_fee' is non-missing, use it;
                        otherwise, use group mean of 'indicative_fee' for each 'dcbs' group.
      - Compute end-of-month (eom) for each observation.
      - For each stock (by 'permno') and each month ('eom'), take the last observation.
      - Remove duplicates (keep groups with exactly one observation).
      - Print statistics on the use of own indicative_fee vs. dcbs group mean.
    
    Returns:
        pd.DataFrame: DataFrame with columns: 'permno', 'eom', 'sfee', 'dcbs'.
    """
    import pandas as pd

    # Convert 'date' to datetime
    short_fees_linked["date"] = pd.to_datetime(short_fees_linked["date"])

    # Flag for using own fee
    short_fees_linked["used_own_fee"] = short_fees_linked["indicative_fee"].notna()

    # Compute sfee using own fee if available, else group mean by dcbs
    short_fees_linked["sfee"] = short_fees_linked.groupby("dcbs")["indicative_fee"].transform(
        lambda grp: grp.fillna(grp.mean())
    )

    # Compute end-of-month (eom)
    short_fees_linked["eom"] = short_fees_linked["date"] + pd.offsets.MonthEnd(0)

    # For each stock (by 'permno') and each month ('eom'), compute maximum date.
    short_fees_linked["max_date"] = short_fees_linked.groupby(["permno", "eom"])["date"].transform("max")

    # Keep rows where date equals max_date
    sf_last = short_fees_linked[short_fees_linked["date"] == short_fees_linked["max_date"]].copy()

    # Print statistics
    own_count = sf_last["used_own_fee"].sum()
    total = len(sf_last)
    group_mean_count = total - own_count
    print(f"Observations using their own indicative_fee: {own_count}")
    print(f"Observations using dcbs group mean: {group_mean_count}")

    # Select required columns.
    sf_last = sf_last[["permno", "eom", "sfee", "dcbs"]]

    # Remove duplicates.
    sf_last["n"] = sf_last.groupby(["permno", "eom"])["permno"].transform("count")
    sf_last = sf_last[sf_last["n"] == 1].copy()
    sf_last.drop(columns=["n"], inplace=True)

    # Generate pickle file
    out_path = os.path.join(Output_Dir, "sf_last.pkl")
    sf_last.to_pickle(out_path)
    print(f"Saved processed short_fees (sf_last) to {out_path}")

    return sf_last


#%% 
def analyze_short_fees(short_fees, chars):
    """
    Merge short fees with stock characteristics and perform analysis.
    
    Steps:
      1. Merge short fees data with characteristics on 'permno' and 'eom'.
      2. Compute summary statistics (proportion non-missing sfee by eom, dcbs frequencies).
      3. Run regressions to examine the predictive power of characteristics on sfee.
      4. Plot selected relationships.
    
    Parameters:
        short_fees (pd.DataFrame): Processed short fees data with columns 'permno', 'eom', 'sfee', 'dcbs'.
        chars (pd.DataFrame): Stock characteristics data; must include 'permno', 'eom', 'market_equity', 
                              'rvol_252d', and 'dolvol_126d'.
    
    Returns:
        pd.DataFrame: The merged DataFrame used for analysis.
    """
    import pandas as pd
    import statsmodels.formula.api as smf
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure that 'permno' in chars is a string and 'eom' is datetime
    chars['permno'] = chars['permno'].astype(str)
    chars['eom'] = pd.to_datetime(chars['eom'])
    
    # Merge on 'permno' and 'eom'
    merged = pd.merge(short_fees, chars[['permno', 'eom', 'market_equity', 'rvol_252d', 'dolvol_126d']],
                      on=['permno', 'eom'], how='left')
    
    # Summary Statistics: Proportion of non-missing sfee by eom
    # "# How many are non-missing? Very few"
    prop_non_missing = merged.groupby('eom')['sfee'].apply(lambda x: x.notna().mean()).reset_index(name='non_miss')
    print("Proportion of non-missing sfee by eom (non-zero only):")
    print(prop_non_missing[prop_non_missing['non_miss'] != 0])
    
    sns.scatterplot(data=prop_non_missing, x='eom', y='non_miss')
    plt.xticks(rotation=45)
    plt.title("Proportion of non-missing sfee by eom")
    plt.show()
    
    # Frequency of dcbs groups
    # "# What is the typical dcbs group? 96% of the sample are in the "easiest to borrow" group"
    freq_dcbs = merged.loc[merged['dcbs'].notna(), 'dcbs'].value_counts(normalize=True)
    print("Frequency of dcbs groups:")
    print(freq_dcbs)
    
    # Regression Analysis
    print("Regression: sfee ~ market_equity")
    mod1 = smf.ols('sfee ~ market_equity', data=merged).fit()
    print(mod1.summary())
    # The output here shows how much variance in sfee is explained by market_equity
    
    print("Regression: sfee ~ market_equity + rvol_252d")
    mod2 = smf.ols('sfee ~ market_equity + rvol_252d', data=merged).fit()
    print(mod2.summary())
    
    print("Regression: sfee ~ market_equity + rvol_252d + dolvol_126d")
    mod3 = smf.ols('sfee ~ market_equity + rvol_252d + dolvol_126d', data=merged).fit()
    print(mod3.summary())
    
    # Outlier Analysis: Plot sfee vs. market_equity for maximum eom
    max_eom = merged['eom'].max()
    subset = merged[(merged['sfee'].notna()) & (merged['eom'] == max_eom)]
    sns.scatterplot(data=subset, x='market_equity', y='sfee')
    plt.title("market_equity vs sfee at max(eom)")
    plt.show()
    print("max_eom:", max_eom)
    print("subset shape:", subset.shape)
    print(subset.head())
    
    # # Regressions on subset excluding top 1% of sfee
    # sfee_99 = merged['sfee'].quantile(0.99)
    # subset99 = merged[merged['sfee'] <= sfee_99]
    # print("Regression on subset (sfee <= 99th percentile): sfee ~ market_equity")
    # mod1_99 = smf.ols('sfee ~ market_equity', data=subset99).fit()
    # print(mod1_99.summary())
    
    # print("Regression on subset: sfee ~ market_equity + rvol_252d")
    # mod2_99 = smf.ols('sfee ~ market_equity + rvol_252d', data=subset99).fit()
    # print(mod2_99.summary())
    
    # print("Regression on subset: sfee ~ market_equity + rvol_252d + dolvol_126d")
    # mod3_99 = smf.ols('sfee ~ market_equity + rvol_252d + dolvol_126d', data=subset99).fit()
    # print(mod3_99.summary())
    
    # # Summary statistics by dcbs
    # summary_dcbs = merged.loc[merged['sfee'].notna()].groupby('dcbs')['sfee'].agg(['count', 'mean', 'median'])
    # total_n = summary_dcbs['count'].sum()
    # summary_dcbs['prop'] = summary_dcbs['count'] / total_n
    # print("Summary statistics by dcbs:")
    # print(summary_dcbs.sort_values('dcbs'))
    
    # # Generate pickle file    
    # out_path = os.path.join(Output_Dir, "merged_analysis.pkl")
    # merged.to_pickle(out_path)
    # print(f"Saved merged analysis data to {out_path}")

    return merged


#%% 
def get_shorting_fees():
    data_path = r"C:\Master"
    file_name = "Shortfees_US_2002_2024.csv"
    
    # 1. Load the short fees data (or load from pickle if exists)
    unfiltered_path = os.path.join(Output_Dir, Unfiltered_PKL)
    if os.path.exists(unfiltered_path):
        short_fees_df = pd.read_pickle(unfiltered_path)
        print("Loaded unfiltered short fees from pickle")
    else:
        short_fees_df = load_short_fees(data_path, file_name)
    
    # 2. Apply initial filtering
    filtered_path = os.path.join(Output_Dir, Filtered_PKL)
    if os.path.exists(filtered_path):
        short_fees_filtered = pd.read_pickle(filtered_path)
        print("Loaded filtered short fees from pickle")
    else:
        short_fees_filtered = short_fees_filters(short_fees_df)
    
    print("After filtering:")
    print(short_fees_filtered.head())
    
    # 3. Link short fees to CRSP permno via CUSIP
    linked_path = os.path.join(Output_Dir, Linked_PKL)
    if os.path.exists(linked_path):
        short_fees_linked = pd.read_pickle(linked_path)
        print("Loaded linked short fees from pickle")
    else:
        short_fees_linked = short_fees_link(short_fees_filtered)
    
    print("After linking (with permno):")
    print(short_fees_linked.head())
    
    # 4. Process the short fees: compute sfee, eom, take last observation, remove duplicates, and print statistics.
    processed_path = os.path.join(Output_Dir, Processed_PKL)
    if os.path.exists(processed_path):
        sf_last = pd.read_pickle(processed_path)
        print("Loaded processed short fees (sf_last) from pickle")
    else:
        sf_last = short_fees_2(short_fees_linked)
    
    print("After processing (sf_last):")
    print(sf_last.head())
    
    # 5. Merge with characteristics for analysis.
    chars_path = r"C:\Master\usa.csv"
    chars = pd.read_csv(chars_path)
    chars['eom'] = pd.to_datetime(chars['eom'])
    chars['permno'] = chars['permno'].astype(str)
    
    merged_analysis = analyze_short_fees(sf_last, chars)
    
    # Generate pickle file    
    merged_analysis_path = os.path.join(Output_Dir, Merged_Analysis_PKL)
    merged_analysis.to_pickle(merged_analysis_path)
    print(f"Saved merged analysis data to {merged_analysis_path}")
    
    print("Merged analysis data:")
    print(merged_analysis.head())

#%%
get_shorting_fees()








#%%
def get_merged_test(merged):
    data_path = r"C:\Master"
    file_name = "merged_analysis.pkl"
    merged_test_print = os.path.join(data_path, file_name)

    return merged_test_print
#%%
path = get_merged_test(merged)
print("Merged analysis pickle path:", path)
#%%
import pandas as pd
df = pd.read_pickle(path)
print("DataFrame shape:", df.shape)
print(df.head())
#%%












#%%
import os
import pandas as pd

Output_Dir = r"C:\Master"
file_name = "merged_analysis.pkl"
file_path = os.path.join(Output_Dir, file_name)
csv_file = "merged_analysis.csv"

# Load the pickle file into a DataFrame
merged_analysis = pd.read_pickle(file_path)
csv_path = os.path.join(Output_Dir, csv_file)
merged_analysis.to_csv(csv_path, index=False)
print(f"Exported merged_analysis to CSV: {csv_path}")
# View the first few rows of the DataFrame
print(merged_analysis.head())
#%%













#%%
# Plots -----------------------------------------------------
# Expand shorting data
if plots:
    print('Creating plots for mean and median indicative for per DCBS group')

    data_path = r"C:\Master"
    file_name = "Shortfees_US_2002_2024.csv"
    short_fees_df = load_short_fees(data_path, file_name) # 1: Get the fees
    short_fees_filtered = short_fees_filters(short_fees_df) # 2: Do the first filtering

    # Copy and ensure 'date' is datetime  
    df = short_fees_filtered.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["indicative_fee"].notna()]   # Filter for rows with non-missing 'indicative_fee'

    # Compute mean and median of 'indicative_fee' by 'date' and 'dcbs'
    grouped = df.groupby(["date", "dcbs"])["indicative_fee"].agg(mean="mean", median="median").reset_index()

    # Pivot to long format (columns: mean, median -> variable, value)
    melted = pd.melt(
        grouped,
        id_vars=["date", "dcbs"],
        value_vars=["mean", "median"],
        var_name="name",
        value_name="value"
    )

    # Create a FacetGrid with subplots for each dcbs value
    g = sns.FacetGrid(
        melted,
        col="dcbs",
        col_wrap=4,
        sharex=False,    # each subplot has its own x-axis
        sharey=False,
        height=3,
        aspect=1.2
    )

    # Plot scatter in each facet, disabling the auto-legend
    g.map_dataframe(
        sns.scatterplot,
        x="date",
        y="value",
        hue="name",
        alpha=0.5,
        s=10,
        marker="o",
        legend=False
    )

    # Create custom legend handles with larger marker sizes
    mean_handle = mlines.Line2D(
        [], [], color='tab:blue', marker='o', linestyle='None',
        markersize=8, label='mean'
    )
    median_handle = mlines.Line2D(
        [], [], color='tab:orange', marker='o', linestyle='None',
        markersize=8, label='median'
    )

    # Make room at the top for the legend
    g.fig.subplots_adjust(top=0.88, bottom=0.15)

    # Place a custom legend above all subplots (top-left corner)
    g.fig.legend(
        handles=[mean_handle, median_handle],
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=True
    )

    # Label each subplot's x-axis and format date ticks
    for ax in g.axes.flat:
        ax.set_xlabel("Date")  # Show 'Date' label under each subplot
        ax.xaxis.set_major_locator(mdates.YearLocator())  # Yearly ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45, labelbottom=True)

    plt.tight_layout()
    plt.show()

#%%








