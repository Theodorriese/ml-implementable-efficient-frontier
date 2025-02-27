#%%
# Imports
import os
import pandas as pd

# Random pip below for specific pip install
# C:\Users\Wiingaard\AppData\Local\Programs\Python\Python311\python.exe -m pip install seaborn

# Loading data
def load_short_fees(data_path, file_name):
    print('Loading initial data...')
    """
    Load and filter shorting fee data, keeping all columns.
    Renames columns to more convenient names
    
    Parameters:
        data_path (str): Path to the directory containing the file.
        file_name (str): Name of the short fees CSV file.

    Returns:
        pd.DataFrame: Unfiltered DataFrame with renamed columns
                      a 'date' column in YYYY-MM-DD format (string).
    """

    # Construct full file path
    file_path = os.path.join(data_path, file_name)

    # Read all columns from the CSV; parse 'Date' as datetime
    # Adjust 'dayfirst' if your CSV uses a different format.
    short_fees = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=False)

    # Dictionary for renaming columns to be more Python-friendly
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

    # Rename columns in the DataFrame
    short_fees.rename(columns=col_rename, inplace=True)

    # Convert date column to string in 'YYYY-MM-DD' format
    short_fees["date"] = short_fees["date"].dt.strftime("%Y-%m-%d")


    return short_fees

#%%


def short_fees_filters(short_fees):
    print('Filtering...')
    """
    Starts filtering process:
      - DCBS not null
      - record_type == 1
      - market_area == "USA Equity" (currently; can be changed to Eurostoxx 600 countries)
      
    Parameters:
        data_path (str): Path to the directory containing the file.
        file_name (str): Name of the short fees CSV file.

    Returns:
        pd.DataFrame: Filtered DataFrame
    """

    # Filter rows:
    # 1. 'dcbs' is not null
    # 2. 'record_type' == 1
    # 3. 'market_area' == "USA Equity" for now (can change to "Eurostoxx 600 countries")
    short_fees_filtered = short_fees[
        short_fees["dcbs"].notna() &
        (short_fees["record_type"] == 1) &
        (short_fees["market_area"] == "USA Equity")  # Change to "Eurostoxx 600 countries" later if needed
    ]

    return short_fees_filtered


#%%
if False:
    def short_fees_link(short_fees_filtered):
        print('Linking')
        """
        Link each row in short_fees_filtered to a CRSP permno via CUSIP.

        Steps:
            1. Read the CRSP_Compustat_Mapping file from your specified path.
            2. Merge on 'cusip' to find the corresponding 'LPERMNO' (CRSP permno).
            3. Rename 'LPERMNO' to 'permno' in the merged dataframe.
            4. Convert 'permno' to string (if you want it stored as text).
        
        Parameters:
            short_fees_filtered (pd.DataFrame): Filtered IHS Markit short fees data,
                                                containing a 'cusip' column.

        Returns:
            pd.DataFrame: DataFrame with a new 'permno' column matching each row's CUSIP
                        to CRSP permno (if a match is found).
        """

        import os
        import pandas as pd

        mapping_dir = r"C:\Master"
        mapping_file = "CRSP_Compustat_Mapping.csv"

        mapping_path = os.path.join(mapping_dir, mapping_file)
        mapping_df = pd.read_csv(mapping_path, usecols=["cusip", "LPERMNO"])

        # Merge short_fees_filtered with the mapping, matching on 'cusip'
        short_fees_linked = short_fees_filtered.merge(
            mapping_df,
            how="left",   # 'left' ensures we keep all rows in short_fees_filtered
            on="cusip"
        )

        short_fees_linked.rename(columns={"LPERMNO": "permno"}, inplace=True) # Rename 'LPERMNO' -> 'permno'
        
        # Convert 'permno' to string
        short_fees_linked["permno"] = short_fees_linked["permno"].astype(str)

        return short_fees_linked




#%%
def short_fees_2(short_fees_filtered):
    print('Handling...')
    """
    Further process the filtered short fees data:
      - Compute 'sfee': If 'indicative_fee' is non-missing, use it;
                        otherwise, use group mean of 'indicative_fee' for each 'dcbs' group.
      - Compute end-of-month (eom) for each observation as the last day of its month.
      - For each stock (identified by 'cusip') and each month ('eom'), take the last observation (i.e. the maximum date).
      - Remove duplicates: retain only groups with exactly one observation.
      
    Returns:
        pd.DataFrame: DataFrame with columns: 'cusip', 'eom', 'sfee', 'dcbs'.
    """
    # Ensure the 'date' column is datetime (it was previously converted to string)
    short_fees_filtered["date"] = pd.to_datetime(short_fees_filtered["date"])

    # Compute sfee: If indicative_fee is not missing, use it;
    # otherwise, fill with the mean of indicative_fee within the same dcbs group.
    short_fees_filtered["sfee"] = short_fees_filtered.groupby("dcbs")["indicative_fee"].transform(
        lambda grp: grp.fillna(grp.mean())
    )

    # Compute end-of-month (eom) as the last day of the month.
    short_fees_filtered["eom"] = short_fees_filtered["date"] + pd.offsets.MonthEnd(0)

    # For each stock (by 'cusip') and each month (by 'eom'),
    # compute the maximum date (i.e. the last observation in that month).
    short_fees_filtered["max_date"] = short_fees_filtered.groupby(["cusip", "eom"])["date"].transform("max")

    # Retain only the rows where 'date' equals the 'max_date'
    sf_last = short_fees_filtered[short_fees_filtered["date"] == short_fees_filtered["max_date"]].copy()

    # Select only the required columns.
    sf_last = sf_last[["cusip", "eom", "sfee", "dcbs"]]

    # Remove duplicates:
    # Count number of rows per (cusip, eom) group.
    sf_last["n"] = sf_last.groupby(["cusip", "eom"])["cusip"].transform("count")
    # Keep only groups with exactly one observation.
    sf_last = sf_last[sf_last["n"] == 1].copy()
    # Drop the temporary count column.
    sf_last.drop(columns=["n"], inplace=True)

    return sf_last





#%%

def get_shorting_fees():
    print('Showing final boss df...')
    # Example usage

    data_path = r"C:\Master"
    file_name = "Shortfees_US_2002_2024.csv"

    short_fees_df = load_short_fees(data_path, file_name) # 1: Get the fees
    short_fees_filtered = short_fees_filters(short_fees_df) # 2: Do the first filtering
    # short_fees_linked = short_fees_link(short_fees_filtered) #3: Link CRSP permno via CUSIP
    sf_last = short_fees_2(short_fees_filtered) # 4: Use means if indicative_fee is missing, find and use eom fee and remove duplicates


    print('Showing short_fees_filtered DataFrame...')
    print(short_fees_filtered.head())
    # print('Showing short_fees_linked DataFrame...')
    # print(short_fees_linked.head())
    print('Showing sf_last DataFrame...')
    print(sf_last)



#%%
get_shorting_fees()

#%%



















#%%
# Expand shorting data
if False:
    print('Creating plots for mean and median indicative for per DCBS group')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as mdates
    import matplotlib.lines as mlines  # For custom legend handles

    #
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








