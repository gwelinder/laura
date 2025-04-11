import pandas as pd
import re
import numpy as np

print("--- Loading Data ---")
# Load the datasets
df_katalog_full = pd.read_excel('ANDF11042025105914807Shot bearsRovbase.xlsx',sheet_name="Katalog", header=2)
df_rapport = pd.read_excel('ANDF11042025105914807Shot bearsRovbase.xlsx',sheet_name="Rapport")

print("\nOriginal df_katalog columns:")
print(df_katalog_full.columns)
print("\nOriginal df_rapport columns:")
print(df_rapport.columns)

print("\n--- Initial Data Cleaning ---")
df_katalog = df_katalog_full.dropna(how='all').copy()
df_rapport = df_rapport.dropna(how='all').copy()
print(f"df_katalog shape after dropna: {df_katalog.shape}")
print(f"df_rapport shape after dropna: {df_rapport.shape}")

# --- Investigate Microchip Column within df_rapport --- 
print("\n--- Investigating Microchip Column (within df_rapport) ---")
if 'Microchip' in df_rapport.columns:
    # Clean the Microchip ID - attempt to extract alphanumeric code
    def clean_microchip(chip_str):
        if pd.isna(chip_str):
            return pd.NA
        # Find alphanumeric sequences (allowing hex like A-F)
        matches = re.findall(r'\b([a-f0-9]{6,})\b', str(chip_str).lower())
        if matches:
            return matches[-1] # Take the last one if multiple found
        return pd.NA # Return NA if no suitable code found
        
    df_rapport['Microchip_clean'] = df_rapport['Microchip'].apply(clean_microchip).astype("string")
    
    nan_count_rapport = df_rapport['Microchip_clean'].isna().sum()
    total_rapport = df_rapport.shape[0]
    unique_count_rapport = df_rapport['Microchip_clean'].nunique()
    print(f"Cleaned Microchip NaNs: {nan_count_rapport}/{total_rapport} ({nan_count_rapport/total_rapport:.2%})")
    print(f"Unique non-NaN Cleaned Microchips: {unique_count_rapport}")

    # Find duplicates
    microchip_counts = df_rapport.dropna(subset=['Microchip_clean'])['Microchip_clean'].value_counts()
    duplicate_microchips = microchip_counts[microchip_counts > 1]
    
    print(f"\nFound {len(duplicate_microchips)} unique Microchip IDs appearing in multiple records.")
    if not duplicate_microchips.empty:
        print("Examples of duplicate Microchip IDs and their counts (Top 10):")
        print(duplicate_microchips.head(10).to_string())
        
        # Show example records for a few duplicate chips
        print("\nExample records for some duplicate Microchip IDs:")
        example_chips = duplicate_microchips.head(3).index.tolist()
        for chip_id in example_chips:
            print(f"\n--- Records for Microchip: {chip_id} ---")
            display_cols = ['RovbaseID', 'Individ', 'Dødsdato', 'Observasjons/Jaktdato', 'Bakgrunn/årsak', 'Microchip']
            print(df_rapport.loc[df_rapport['Microchip_clean'] == chip_id, display_cols].to_markdown(index=False))
else:
    print("'Microchip' column NOT found in df_rapport.")

# --- Check df_katalog for Microchip (as before, just for logging) --- 
print("\n--- Checking Microchip Column (in df_katalog) ---")
possible_katalog_chip_cols = [col for col in df_katalog.columns if 'chip' in str(col).lower()]
if not possible_katalog_chip_cols:
     print("No obvious 'Microchip' column found in df_katalog.")

# --- Prepare Rapport Key --- 
print("\n--- Preparing Rapport Key (Multi-Pattern Extraction from Individ) ---")
id_pattern_1 = re.compile(r'\b(\d{2}[a-z]{1,2}\d{2})\b')
id_pattern_2 = re.compile(r'\b([a-z]{1,3}\d{3,})\b')
# Add more patterns here if needed

def extract_id_multi_pattern(text):
    # Clean the text first
    cleaned_text = str(text).lower()
    cleaned_text = re.sub(r'[^a-z0-9\s]+', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    if not cleaned_text:
        return pd.NA
        
    match1 = id_pattern_1.search(cleaned_text)
    if match1:
        return match1.group(1)
    match2 = id_pattern_2.search(cleaned_text)
    if match2:
        return match2.group(1)
    return pd.NA

df_rapport['Extracted_ID'] = df_rapport['Individ'].apply(extract_id_multi_pattern).astype("string")
extracted_count = df_rapport['Extracted_ID'].notna().sum()
print(f"Extracted potential IDs (multi-pattern) from 'Individ': {extracted_count}/{df_rapport.shape[0]}")

# --- Prepare Katalog Keys --- 
print("\n--- Preparing Katalog Keys (ID No Space, ID, BAG no) ---")
def clean_katalog_key(series):
    return series.astype(str).str.lower().str.strip().astype("string").replace(['nan', ''], pd.NA)

df_katalog['ID_katalog_clean_NoSpace'] = clean_katalog_key(df_katalog['ID (No Space)'])
print(f"Unique cleaned Katalog 'ID (No Space)': {df_katalog['ID_katalog_clean_NoSpace'].nunique()}")

if 'ID' in df_katalog.columns:
    df_katalog['ID_katalog_clean_ID'] = clean_katalog_key(df_katalog['ID'])
    print(f"Unique cleaned Katalog 'ID': {df_katalog['ID_katalog_clean_ID'].nunique()}")
else:
    df_katalog['ID_katalog_clean_ID'] = pd.NA
    print("Katalog 'ID' column not found.")

if 'BAG no' in df_katalog.columns:
    df_katalog['ID_katalog_clean_BagNo'] = clean_katalog_key(df_katalog['BAG no'])
    print(f"Unique cleaned Katalog 'BAG no': {df_katalog['ID_katalog_clean_BagNo'].nunique()}")
else:
    df_katalog['ID_katalog_clean_BagNo'] = pd.NA
    print("Katalog 'BAG no' column not found.")

# --- Attempt Merges --- 
print("\n--- Attempting Merges --- ")
merge_results = {}

# Merge 1: Extracted ID vs ID (No Space)
katalog_cols_for_merge = [col for col in df_katalog.columns if col not in ['Alder på dødt individ']]
merge1_df = pd.merge(df_rapport, df_katalog[katalog_cols_for_merge], 
                     left_on='Extracted_ID', right_on='ID_katalog_clean_NoSpace', 
                     how='inner', suffixes=['_rapport', '_katalog'])
merge_results['ID (No Space)'] = merge1_df.shape[0]
print(f"Merge 1 (vs ID No Space) rows: {merge1_df.shape[0]}")

# Merge 2: Extracted ID vs ID
if 'ID_katalog_clean_ID' in df_katalog.columns:
    merge2_df = pd.merge(df_rapport, df_katalog[katalog_cols_for_merge], 
                         left_on='Extracted_ID', right_on='ID_katalog_clean_ID', 
                         how='inner', suffixes=['_rapport', '_katalog'])
    merge_results['ID'] = merge2_df.shape[0]
    print(f"Merge 2 (vs ID) rows: {merge2_df.shape[0]}")
else:
    merge_results['ID'] = 0

# Merge 3: Extracted ID vs BAG no
if 'ID_katalog_clean_BagNo' in df_katalog.columns:
    merge3_df = pd.merge(df_rapport, df_katalog[katalog_cols_for_merge], 
                         left_on='Extracted_ID', right_on='ID_katalog_clean_BagNo', 
                         how='inner', suffixes=['_rapport', '_katalog'])
    merge_results['BAG no'] = merge3_df.shape[0]
    print(f"Merge 3 (vs BAG no) rows: {merge3_df.shape[0]}")
else:
     merge_results['BAG no'] = 0

# --- Select Best Merge --- 
# NOTE: Forcing the use of 'ID (No Space)' merge as it seems most conceptually correct
# despite 'BAG no' potentially yielding more (likely incorrect many-to-many) rows.
best_merge_key = 'ID (No Space)' 

if best_merge_key not in merge_results or merge_results[best_merge_key] == 0:
    print(f"ERROR: Selected merge key '{best_merge_key}' failed or yielded 0 rows. Cannot proceed.")
    exit()
else:
    best_merge_rows = merge_results[best_merge_key]
    print(f"\nProceeding with merge using Katalog column: '{best_merge_key}' with {best_merge_rows} rows.")

# Re-perform the selected merge to get the final matches_df
# Fix key construction bug - use the original key name for lookup
katalog_key_col_name = {
    'ID (No Space)': 'ID_katalog_clean_NoSpace',
    'ID': 'ID_katalog_clean_ID',
    'BAG no': 'ID_katalog_clean_BagNo'
}.get(best_merge_key) # Get the correct column name based on the selected key

if not katalog_key_col_name or katalog_key_col_name not in df_katalog.columns:
     print(f"ERROR: Could not find the correct key column '{katalog_key_col_name}' for the selected merge.")
     exit()
     
matches_df = pd.merge(df_rapport, df_katalog[katalog_cols_for_merge], 
                     left_on='Extracted_ID', right_on=katalog_key_col_name, 
                     how='inner', suffixes=['_rapport', '_katalog'])


# --- Proceed with the matches_df (from selected merge) --- 
print(f"\n--- Proceeding with DataFrame of shape {matches_df.shape} from '{best_merge_key}' merge ---")
# --- Cleaning and Transforming 'Alder, vurdert' ---
print("\n--- Processing 'Alder, vurdert' column ---")
# Check for potentially suffixed column name from rapport
alder_col = 'Alder, vurdert_rapport' if 'Alder, vurdert_rapport' in matches_df.columns else 'Alder, vurdert'
if alder_col in matches_df.columns:
    print(f"Original '{alder_col}' value counts (Top 20):")
    print(matches_df[alder_col].value_counts().head(20).to_string())
    print(f"Original '{alder_col}' NaN count: {matches_df[alder_col].isna().sum()}")
    
    def clean_age(age_str):
        if pd.isna(age_str):
            return np.nan
        original_value = age_str
        try:
            age_str = str(age_str).replace(',', '.')
            if '-' in age_str:
                ages = age_str.split('-')
                mean_age = np.mean([float(age) for age in ages])
                return mean_age
            else:
                return float(age_str)
        except ValueError:
            return np.nan
    
    matches_df['Alder, vurdert_cleaned'] = matches_df[alder_col].apply(clean_age)
    nan_after_cleaning = matches_df['Alder, vurdert_cleaned'].isna().sum()
    print(f"NaN count in 'Alder, vurdert' after cleaning: {nan_after_cleaning}")
else:
    print(f"Column '{alder_col}' (or suffixed version) not found for age cleaning.")
    matches_df['Alder, vurdert_cleaned'] = np.nan

# --- Logging 'Helvekt' --- 
print("\n--- Logging 'Helvekt' column ---")
# Check for potentially suffixed column name from rapport
helvekt_col = 'Helvekt_rapport' if 'Helvekt_rapport' in matches_df.columns else 'Helvekt'
if helvekt_col in matches_df.columns:
    print(f"Original '{helvekt_col}' value counts (Top 20):")
    print(matches_df[helvekt_col].value_counts().head(20).to_string())
    print(f"Original '{helvekt_col}' NaN count: {matches_df[helvekt_col].isna().sum()}")
    # Note: Actual cleaning happens in analysis.py
else:
    print(f"Column '{helvekt_col}' (or suffixed version) not found.")

# --- Converting Date Columns ---
print("\n--- Processing Date Columns ---")
# Check for potentially suffixed column names from rapport
dods_col = 'Dødsdato_rapport' if 'Dødsdato_rapport' in matches_df.columns else 'Dødsdato'
obs_col = 'Observasjons/Jaktdato_rapport' if 'Observasjons/Jaktdato_rapport' in matches_df.columns else 'Observasjons/Jaktdato'
if dods_col in matches_df.columns:
    original_dods_nat = matches_df[dods_col].isna().sum()
    matches_df['Dødsdato_cleaned'] = pd.to_datetime(matches_df[dods_col], errors='coerce')
    dods_nat_after = matches_df['Dødsdato_cleaned'].isna().sum()
    print(f"'{dods_col}': Original NaNs: {original_dods_nat}, NaNs/NaT after conversion: {dods_nat_after}")
else:
    print(f"Column '{dods_col}' not found.")
    matches_df['Dødsdato_cleaned'] = pd.NaT
    
if obs_col in matches_df.columns:
    original_obs_nat = matches_df[obs_col].isna().sum()
    matches_df['Observasjons/Jaktdato_cleaned'] = pd.to_datetime(matches_df[obs_col], errors='coerce')
    obs_nat_after = matches_df['Observasjons/Jaktdato_cleaned'].isna().sum()
    print(f"'{obs_col}': Original NaNs: {original_obs_nat}, NaNs/NaT after conversion: {obs_nat_after}")
else:
    print(f"Column '{obs_col}' not found.")
    matches_df['Observasjons/Jaktdato_cleaned'] = pd.NaT

# --- Feature Engineering ---
print("\n--- Feature Engineering ---")
# Use cleaned date columns
if 'Dødsdato_cleaned' in matches_df.columns and 'Observasjons/Jaktdato_cleaned' in matches_df.columns:
    matches_df['Tid_Mellom_Observasjon_Død'] = (matches_df['Dødsdato_cleaned'] - matches_df['Observasjons/Jaktdato_cleaned']).dt.days
    print("Created 'Tid_Mellom_Observasjon_Død'.")
else:
    print("Could not create 'Tid_Mellom_Observasjon_Død'.")
    matches_df['Tid_Mellom_Observasjon_Død'] = np.nan
    
# Check for potentially suffixed column name from rapport
funnsted_col = 'Funnsted_rapport' if 'Funnsted_rapport' in matches_df.columns else 'Funnsted'
if funnsted_col in matches_df.columns:
    matches_df['Funnsted_kort'] = matches_df[funnsted_col].apply(lambda x: x.split(',')[0] if isinstance(x, str) else x)
    print("Created 'Funnsted_kort'.")
else:
    print(f"Column '{funnsted_col}' (or suffixed version) not found.")
    matches_df['Funnsted_kort'] = np.nan

# --- Prepare Final DataFrame ---
print("\n--- Preparing Final DataFrame ---")
# Adjust rapport_cols to look for suffixed names if they exist
rapport_cols_base = [
    'RovbaseID', 'DNAID', 'Art', 'Bakgrunn/årsak', 
    'Kjønn', 'Helvekt', 'Funnsted', 'Fylke',
    # Add new columns needed for further analysis:
    'Nord (UTM33/SWEREF99 TM)', 'Øst (UTM33/SWEREF99 TM)', 
    'Alder på dødt individ', 'Slaktevekt', 'Kommune'
]
rapport_cols = []
for base_col in rapport_cols_base:
    # Prioritize suffixed column if merge added it
    suffixed_col = f"{base_col}_rapport"
    if suffixed_col in matches_df.columns:
        rapport_cols.append(suffixed_col)
    elif base_col in matches_df.columns:
         rapport_cols.append(base_col)
    else:
         print(f"Warning: Base column '{base_col}' not found in matches_df (nor suffixed version).")
         
cleaned_engineered_cols = {
    'Alder, vurdert_cleaned': 'Alder, vurdert',
    'Dødsdato_cleaned': 'Dødsdato',
    'Observasjons/Jaktdato_cleaned': 'Observasjons/Jaktdato',
    'Tid_Mellom_Observasjon_Død': 'Tid_Mellom_Observasjon_Død',
    'Funnsted_kort': 'Funnsted_kort'
}
    
final_cols_needed = list(cleaned_engineered_cols.keys())
rename_dict = cleaned_engineered_cols.copy()
    
for col in rapport_cols:
    # Figure out the base name to check against rename_dict.values()
    base_name = col.replace('_rapport', '')
    if col in matches_df.columns and base_name not in rename_dict.values():
         final_cols_needed.append(col)
         # Add renaming for suffixed columns if needed
         if col.endswith('_rapport'):
             rename_dict[col] = base_name
         # No explicit else needed: if col == base_name, no rename needed
         elif col in rename_dict: # Avoid adding if it's already a key to be renamed
             print(f"Debug: Column '{col}' skipped as it's already in rename_dict keys")
             pass 
                 
# Ensure essential analysis columns (numeric/datetime cleaned + others) are selected
# Use the *final* names after potential renaming
analysis_essential_final_names = [
    'Alder, vurdert', 'Helvekt', 'Dødsdato', 'Observasjons/Jaktdato', 
    'Kjønn', 'Fylke', 'Tid_Mellom_Observasjon_Død', 'Funnsted_kort', 
    'Bakgrunn/årsak', 
    # Add newly required columns (final names)
    'Nord (UTM33/SWEREF99 TM)', 'Øst (UTM33/SWEREF99 TM)',
    'Alder på dødt individ', 'Slaktevekt', 'Kommune' 
] 

# Check which essential columns are available using the original/suffixed names
for final_name in analysis_essential_final_names:
    original_col = final_name
    suffixed_col = f"{final_name}_rapport"
    # Find the column name present in matches_df
    col_to_add = None
    if suffixed_col in matches_df.columns:
        col_to_add = suffixed_col
    elif original_col in matches_df.columns:
        col_to_add = original_col
        
    # Add if found and not already covered by rename_dict or final_cols_needed
    if col_to_add and col_to_add not in final_cols_needed and col_to_add not in rename_dict:
        final_cols_needed.append(col_to_add)
        # Ensure correct renaming if suffixed
        if col_to_add.endswith('_rapport'):
            rename_dict[col_to_add] = final_name

final_cols_needed = list(dict.fromkeys(final_cols_needed))
# Ensure we only select columns that actually exist in matches_df
final_cols_present = [col for col in final_cols_needed if col in matches_df.columns]
    
final_df = matches_df[final_cols_present].rename(columns=rename_dict)

# Final check
missing_essentials = [col for col in analysis_essential_final_names if col not in final_df.columns]
if missing_essentials:
    print(f"\nWARNING: Essential columns for analysis missing in final_df: {missing_essentials}")

print(f"Final DataFrame columns before saving: {final_df.columns.tolist()}")
print(f"Final DataFrame shape before saving: {final_df.shape}")

# --- Output the result ---
print("\n--- Final Output ---")
# ... (output markdown and info as before) ...
print("First 5 rows of final_df:")
with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
    print(final_df.head().to_markdown(index=False, numalign="left", stralign="left"))

print("\nfinal_df Info:")
final_df.info()

# --- Save the result ---
final_df.to_csv('matches_df.csv', index=False)
print("\nfinal_df saved to matches_df.csv")
