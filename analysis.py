import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats # Import scipy
import geopandas as gpd # Import geopandas

# Ensure plots directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the dataset
try:
    df = pd.read_csv('matches_df.csv')
    print("matches_df.csv loaded successfully.")
except FileNotFoundError:
    print("Error: matches_df.csv not found. Please run clean_up.py first.")
    exit()

# Set plot style
sns.set_theme(style="whitegrid")

# 1. Histogram of 'Alder, vurdert' (Estimated Age)
plt.figure(figsize=(10, 6))
sns.histplot(df['Alder, vurdert'].dropna(), kde=True, bins=15)
plt.title('Distribution of Estimated Age (Alder, vurdert)')
plt.xlabel('Estimated Age')
plt.ylabel('Frequency')
plt.savefig('plots/estimated_age_distribution.png')
plt.close()
print("Saved plot: plots/estimated_age_distribution.png")

# 2. Count plot of 'Kjønn' (Gender)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Kjønn', order = df['Kjønn'].value_counts().index)
plt.title('Gender Distribution (Kjønn)')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('plots/gender_distribution.png')
plt.close()
print("Saved plot: plots/gender_distribution.png")

# 3. Histogram of 'Tid_Mellom_Observasjon_Død'
# Filter out potential outliers or incorrect data (e.g., negative days) if necessary
time_diff = df['Tid_Mellom_Observasjon_Død'][df['Tid_Mellom_Observasjon_Død'] >= 0]
plt.figure(figsize=(10, 6))
sns.histplot(time_diff.dropna(), kde=False, bins=30) # Using more bins, KDE might be less informative here
plt.title('Distribution of Days Between Observation/Hunt and Death')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.xlim(left=0) # Ensure x-axis starts at 0
plt.savefig('plots/time_difference_distribution.png')
plt.close()
print("Saved plot: plots/time_difference_distribution.png")

# 4. Count plot of 'Fylke' (County)
plt.figure(figsize=(12, 8))
sns.countplot(data=df, y='Fylke', order = df['Fylke'].value_counts().index)
plt.title('Distribution by County (Fylke)')
plt.xlabel('Count')
plt.ylabel('County')
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.savefig('plots/county_distribution.png')
plt.close()
print("Saved plot: plots/county_distribution.png")

# --- Additional Analysis ---

# Clean 'Helvekt' (Full Weight)
def clean_weight(weight):
    if pd.isna(weight):
        return np.nan
    try:
        # Convert to string, remove potential thousand separators (like spaces), replace comma with dot
        weight_str = str(weight).replace(' ', '').replace(',', '.')
        return float(weight_str)
    except ValueError:
        return np.nan # Return NaN if conversion fails

df['Helvekt_numeric'] = df['Helvekt'].apply(clean_weight)

# 5. Scatter plot of Age vs. Weight
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Alder, vurdert', y='Helvekt_numeric', hue='Kjønn', alpha=0.7)
plt.title('Estimated Age vs. Full Weight (Helvekt)')
plt.xlabel('Estimated Age')
plt.ylabel('Full Weight (kg)')
plt.savefig('plots/age_vs_weight_scatter.png')
plt.close()
print("Saved plot: plots/age_vs_weight_scatter.png")

# 6. Count plot of 'Bakgrunn/årsak' (Reason/Background)
plt.figure(figsize=(10, 10)) # Increased height for better label visibility
reason_counts = df['Bakgrunn/årsak'].value_counts()
# Limit to top N reasons if there are too many
# N = 20
# sns.countplot(data=df[df['Bakgrunn/årsak'].isin(reason_counts.index[:N])], y='Bakgrunn/årsak', order = reason_counts.index[:N])
sns.countplot(data=df, y='Bakgrunn/årsak', order = reason_counts.index)
plt.title('Distribution by Reason/Background (Bakgrunn/årsak)')
plt.xlabel('Count')
plt.ylabel('Reason/Background')
plt.tight_layout()
plt.savefig('plots/reason_distribution.png')
plt.close()
print("Saved plot: plots/reason_distribution.png")

# --- Further Analysis ---

# 7. Correlation Heatmap
numeric_cols = ['Alder, vurdert', 'Helvekt_numeric', 'Tid_Mellom_Observasjon_Død']
# Ensure Tid_Mellom_Observasjon_Død is already numeric from clean_up.py
# Remove: df['Tid_Mellom_Observasjon_Død'] = pd.to_numeric(df['Tid_Mellom_Observasjon_Død'], errors='coerce')

# Select only the numeric columns that actually exist in the dataframe after loading
numeric_cols_present = [col for col in numeric_cols if col in df.columns]
if not numeric_cols_present:
    print("\nWarning: No numeric columns found for correlation heatmap.")
else:
    print(f"\nCalculating correlation for: {numeric_cols_present}")
    # Calculate correlation only on existing numeric columns
    correlation_matrix = df[numeric_cols_present].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()
    print("Saved plot: plots/correlation_heatmap.png")

# Filter for known genders for comparison plots
df_gender_known = df[df['Kjønn'].isin(['Hane', 'Hona'])]

# 8. Age Distribution by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_gender_known, x='Kjønn', y='Alder, vurdert', order=['Hane', 'Hona'])
plt.title('Estimated Age Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Estimated Age')
plt.savefig('plots/age_distribution_by_gender.png')
plt.close()
print("Saved plot: plots/age_distribution_by_gender.png")

# 9. Weight Distribution by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_gender_known, x='Kjønn', y='Helvekt_numeric', order=['Hane', 'Hona'])
plt.title('Full Weight Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Full Weight (kg)')
plt.savefig('plots/weight_distribution_by_gender.png')
plt.close()
print("Saved plot: plots/weight_distribution_by_gender.png")

# 10. Records Over Time (using Dødsdato)
# Ensure Dødsdato is datetime
df['Dødsdato'] = pd.to_datetime(df['Dødsdato'], errors='coerce')
df['Death_Year'] = df['Dødsdato'].dt.year
yearly_counts = df['Death_Year'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
yearly_counts.plot(kind='line', marker='o')
plt.title('Number of Records (Deaths) Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Records')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/records_over_time.png')
plt.close()
print("Saved plot: plots/records_over_time.png")

# 11. Age Distribution by Top N Counties
top_n_counties = df['Fylke'].value_counts().nlargest(5).index
df_top_counties = df[df['Fylke'].isin(top_n_counties)]

plt.figure(figsize=(12, 7))
sns.boxplot(data=df_top_counties, x='Alder, vurdert', y='Fylke', order=top_n_counties)
plt.title('Estimated Age Distribution by Top 5 Counties')
plt.xlabel('Estimated Age')
plt.ylabel('County')
plt.tight_layout()
plt.savefig('plots/age_distribution_by_county.png')
plt.close()
print("Saved plot: plots/age_distribution_by_county.png")

# --- Statistical Tests ---
print("\n--- Performing Statistical Tests ---")

# Filter out NaNs for tests
df_test = df.dropna(subset=['Alder, vurdert', 'Helvekt_numeric', 'Kjønn', 'Fylke']).copy()

# 1. T-test: Age by Gender
male_age = df_test[df_test['Kjønn'] == 'Hane']['Alder, vurdert']
female_age = df_test[df_test['Kjønn'] == 'Hona']['Alder, vurdert']

if len(male_age) > 1 and len(female_age) > 1:
    t_stat_age, p_val_age = stats.ttest_ind(male_age, female_age, equal_var=False) # Welch's t-test
    print(f"\nT-test for Age (Hane vs. Hona):")
    print(f"  T-statistic: {t_stat_age:.3f}")
    print(f"  P-value: {p_val_age:.3f}")
    if p_val_age < 0.05:
        print("  Result: Statistically significant difference in mean age between genders (p < 0.05).")
    else:
        print("  Result: No statistically significant difference in mean age between genders (p >= 0.05).")
else:
    print("\nCould not perform t-test for Age by Gender (insufficient data).")

# 2. T-test: Weight by Gender
male_weight = df_test[df_test['Kjønn'] == 'Hane']['Helvekt_numeric']
female_weight = df_test[df_test['Kjønn'] == 'Hona']['Helvekt_numeric']

if len(male_weight) > 1 and len(female_weight) > 1:
    t_stat_weight, p_val_weight = stats.ttest_ind(male_weight, female_weight, equal_var=False)
    print(f"\nT-test for Weight (Hane vs. Hona):")
    print(f"  T-statistic: {t_stat_weight:.3f}")
    print(f"  P-value: {p_val_weight:.3f}")
    if p_val_weight < 0.05:
        print("  Result: Statistically significant difference in mean weight between genders (p < 0.05).")
    else:
        print("  Result: No statistically significant difference in mean weight between genders (p >= 0.05).")
else:
     print("\nCould not perform t-test for Weight by Gender (insufficient data).")

# 3. ANOVA: Age by Top 5 Counties
# Prepare data for ANOVA: list of age arrays for each county
top_counties_data = df_test[df_test['Fylke'].isin(top_n_counties)] # Use df_test with NaNs dropped
anova_groups = [group['Alder, vurdert'].values for name, group in top_counties_data.groupby('Fylke')]

# Check if we have at least 2 groups with data
if len(anova_groups) >= 2:
    f_stat_county, p_val_county = stats.f_oneway(*anova_groups)
    print(f"\nANOVA for Age by Top 5 Counties:")
    print(f"  F-statistic: {f_stat_county:.3f}")
    print(f"  P-value: {p_val_county:.3f}")
    if p_val_county < 0.05:
        print("  Result: Statistically significant difference in mean age across counties (p < 0.05).")
        # Post-hoc tests could be added here if desired (e.g., Tukey HSD)
    else:
        print("  Result: No statistically significant difference in mean age across counties (p >= 0.05).")
else:
    print("\nCould not perform ANOVA for Age by County (insufficient groups/data).")

# --- Further Detailed Analysis ---
print("\n--- Performing Further Detailed Analysis ---")

# 12. Municipality Distribution
if 'Kommune' in df.columns:
    plt.figure(figsize=(12, 10)) # Adjust size as needed
    # Limit to top N municipalities if too many
    top_n = 30 
    kommune_counts = df['Kommune'].value_counts()
    if len(kommune_counts) > top_n:
         top_kommune = kommune_counts.nlargest(top_n).index
         df_plot = df[df['Kommune'].isin(top_kommune)]
         plot_order = top_kommune
         title = f'Distribution by Top {top_n} Municipality (Kommune)'
    else:
         df_plot = df
         plot_order = kommune_counts.index
         title = 'Distribution by Municipality (Kommune)'
         
    sns.countplot(data=df_plot, y='Kommune', order=plot_order)
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Municipality')
    plt.tight_layout()
    plt.savefig('plots/municipality_distribution.png')
    plt.close()
    print("Saved plot: plots/municipality_distribution.png")
else:
    print("'Kommune' column not found. Skipping municipality distribution plot.")

# 13. Alternative Age Comparison
if 'Alder på dødt individ' in df.columns and 'Alder, vurdert' in df.columns:
    plt.figure(figsize=(10, 6))
    # Use original df for this comparison, before df_test filtering
    age_comp_df = df.dropna(subset=['Alder på dødt individ', 'Alder, vurdert'])
    sns.boxplot(data=age_comp_df, x='Alder, vurdert', y='Alder på dødt individ')
    plt.title('Numeric Estimated Age vs. Categorical Age')
    plt.xlabel('Numeric Estimated Age (Alder, vurdert)')
    plt.ylabel('Categorical Age (Alder på dødt individ)')
    plt.tight_layout()
    plt.savefig('plots/age_comparison.png')
    plt.close()
    print("Saved plot: plots/age_comparison.png")
else:
    print("Could not perform age comparison plot (missing columns).")

# 14. Weight Comparison
if 'Slaktevekt' in df.columns and 'Helvekt_numeric' in df.columns:
    # Clean 'Slaktevekt' like 'Helvekt'
    df['Slaktevekt_numeric'] = df['Slaktevekt'].apply(clean_weight) # Use clean_weight from earlier
    weight_comp_df = df.dropna(subset=['Helvekt_numeric', 'Slaktevekt_numeric'])
    
    if not weight_comp_df.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=weight_comp_df, x='Helvekt_numeric', y='Slaktevekt_numeric', alpha=0.7)
        # Add a y=x line for reference
        max_val = max(weight_comp_df['Helvekt_numeric'].max(), weight_comp_df['Slaktevekt_numeric'].max())
        min_val = min(weight_comp_df['Helvekt_numeric'].min(), weight_comp_df['Slaktevekt_numeric'].min())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
        plt.title('Full Weight (Helvekt) vs. Slaughter Weight (Slaktevekt)')
        plt.xlabel('Full Weight (kg)')
        plt.ylabel('Slaughter Weight (kg)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/weight_comparison.png')
        plt.close()
        print("Saved plot: plots/weight_comparison.png")
    else:
        print("No non-NaN data for weight comparison plot.")
else:
    print("Could not perform weight comparison plot (missing columns).")

# 15. Geospatial Plot
coord_cols = ['Nord (UTM33/SWEREF99 TM)', 'Øst (UTM33/SWEREF99 TM)']
if all(col in df.columns for col in coord_cols):
    try:
        # Convert relevant columns to numeric, coercing errors
        df['X_coord'] = pd.to_numeric(df[coord_cols[1]], errors='coerce')
        df['Y_coord'] = pd.to_numeric(df[coord_cols[0]], errors='coerce')
        df['Age_numeric'] = pd.to_numeric(df['Alder, vurdert'], errors='coerce') # Use already cleaned age
        
        # Drop rows where coordinates or age are NaN
        gdf = df.dropna(subset=['X_coord', 'Y_coord', 'Age_numeric']).copy()
        
        if not gdf.empty:
            # Create GeoDataFrame
            geometry = gpd.points_from_xy(gdf.X_coord, gdf.Y_coord)
            gdf = gpd.GeoDataFrame(gdf, geometry=geometry, crs="EPSG:3006") # SWEREF99 TM (EPSG:3006)
            
            # Basic plot - requires contextily for basemap, might need separate install
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            gdf.plot(column='Age_numeric', ax=ax, legend=True,
                     legend_kwds={'label': "Estimated Age",
                                  'orientation': "horizontal"},
                     markersize=20, cmap='magma', alpha=0.8, edgecolor='black', linewidth=0.5) # Larger markersize, less transparent, edge
            ax.set_title('Geographic Distribution of Records by Estimated Age (UTM33 / SWEREF99 TM)')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            # Try adding a basemap if contextily is available
            try:
                import contextily as ctx
                ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
                print("Added OpenStreetMap basemap using contextily.")
            except ImportError:
                print("Contextily not found, skipping basemap. Install with 'pip install contextily' for map backgrounds.")
                
            plt.tight_layout()
            plt.savefig('plots/geospatial_distribution_age.png')
            plt.close()
            print("Saved plot: plots/geospatial_distribution_age.png")
        else:
            print("No valid coordinate/age data for geospatial plot after cleaning.")
            
    except Exception as e:
        print(f"Error during geospatial plotting: {e}")
else:
    print("Coordinate columns not found. Skipping geospatial plot.")

# 16. Geospatial Plot by Cause
cause_col = 'Bakgrunn/årsak'
if all(col in df.columns for col in coord_cols) and cause_col in df.columns:
    try:
        gdf_cause = df.dropna(subset=['X_coord', 'Y_coord', cause_col]).copy()
        if not gdf_cause.empty:
            geometry = gpd.points_from_xy(gdf_cause.X_coord, gdf_cause.Y_coord)
            gdf_cause = gpd.GeoDataFrame(gdf_cause, geometry=geometry, crs="EPSG:3006")
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            gdf_cause.plot(column=cause_col, ax=ax, legend=True, categorical=True,
                           legend_kwds={'title': "Cause", 'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
                           markersize=20, cmap='tab10', alpha=0.8, edgecolor='black', linewidth=0.5) # Larger markersize, less transparent, edge
            ax.set_title('Geographic Distribution of Records by Cause (UTM33 / SWEREF99 TM)')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            try:
                import contextily as ctx
                ctx.add_basemap(ax, crs=gdf_cause.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
                print("Added OpenStreetMap basemap (Cause plot).")
            except ImportError:
                print("Contextily not found, skipping basemap (Cause plot).")
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
            plt.savefig('plots/geospatial_distribution_cause.png')
            plt.close()
            print("Saved plot: plots/geospatial_distribution_cause.png")
        else:
            print("No valid coordinate/cause data for geospatial plot after cleaning.")
    except Exception as e:
        print(f"Error during geospatial plotting by cause: {e}")
else:
    print(f"Coordinate or '{cause_col}' columns not found. Skipping geospatial plot by cause.")

# 17. Geospatial Plot by Year
date_col = 'Dødsdato'
if all(col in df.columns for col in coord_cols) and date_col in df.columns:
    try:
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['Death_Year'] = df[date_col].dt.year
        gdf_year = df.dropna(subset=['X_coord', 'Y_coord', 'Death_Year']).copy()
        gdf_year['Death_Year'] = gdf_year['Death_Year'].astype(int) # Convert year to int for plotting
        
        if not gdf_year.empty:
            geometry = gpd.points_from_xy(gdf_year.X_coord, gdf_year.Y_coord)
            gdf_year = gpd.GeoDataFrame(gdf_year, geometry=geometry, crs="EPSG:3006")
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            gdf_year.plot(column='Death_Year', ax=ax, legend=True,
                          legend_kwds={'label': "Year of Death",
                                       'orientation': "horizontal"},
                          markersize=20, cmap='cividis', alpha=0.8, edgecolor='black', linewidth=0.5) # Larger markersize, less transparent, edge
            ax.set_title('Geographic Distribution of Records by Year (UTM33 / SWEREF99 TM)')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            try:
                import contextily as ctx
                ctx.add_basemap(ax, crs=gdf_year.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
                print("Added OpenStreetMap basemap (Year plot).")
            except ImportError:
                print("Contextily not found, skipping basemap (Year plot).")
            plt.tight_layout()
            plt.savefig('plots/geospatial_distribution_year.png')
            plt.close()
            print("Saved plot: plots/geospatial_distribution_year.png")
        else:
            print("No valid coordinate/year data for geospatial plot after cleaning.")
    except Exception as e:
        print(f"Error during geospatial plotting by year: {e}")
else:
    print(f"Coordinate or '{date_col}' columns not found. Skipping geospatial plot by year.")

print("\nAnalysis complete.") 