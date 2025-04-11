# Bear Data Analysis Project

This project cleans, merges, and analyzes data about bears from an Excel file, generating various visualizations and statistical summaries.

## Setup Instructions (macOS)

1.  **Prerequisites:**

    - Ensure you have Python 3 installed. You can check by opening Terminal and running `python3 --version`.
    - Ensure you have `pip` (Python package installer) available.

2.  **Clone or Download Project:**

    - Obtain the project files, including `clean_up.py`, `analysis.py`, and the input Excel file (`ANDF11042025105914807Shot bearsRovbase.xlsx`).

3.  **Navigate to Project Directory:**

    - Open Terminal and use the `cd` command to navigate into the project folder.

    ```bash
    cd path/to/your/project/folder
    ```

4.  **Create and Activate Virtual Environment (Recommended):**

    - Create a virtual environment to isolate project dependencies:

    ```bash
    python3 -m venv .venv
    ```

    - Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

    - You should see `(.venv)` at the beginning of your terminal prompt.

5.  **Install Dependencies:**

    - Install the required Python libraries using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

    - This file lists all necessary packages and their specific versions.
    - (Note: `geopandas` included in the file might still have system dependencies like GDAL; if installation fails, consult `geopandas` documentation for installing its prerequisites on your system before running the pip command again).

6.  **Input Data:**
    - Make sure the Excel file `ANDF11042025105914807Shot bearsRovbase.xlsx` is present in the same directory as the Python scripts.

## Running the Analysis

1.  **Ensure Virtual Environment is Active:** If not active, run `source .venv/bin/activate` in the project directory.

2.  **Run the Cleaning Script:** This script reads the Excel file, cleans the data, performs merges, and saves the result to `matches_df.csv`.

    ```bash
    python clean_up.py
    ```

3.  **Run the Analysis Script:** This script reads `matches_df.csv` and generates plots in the `plots/` directory.
    ```bash
    python analysis.py
    ```
    - Alternatively, run both in sequence:
    ```bash
    python clean_up.py && python analysis.py
    ```

## Output

- `matches_df.csv`: A CSV file containing the cleaned and merged data subset used for analysis.
- `plots/` directory: Contains various PNG images visualizing the data (histograms, scatter plots, box plots, maps, etc.).
- Console output: Logs from the cleaning process and results from statistical tests.

## Deactivating Virtual Environment

- When you are finished working on the project, you can deactivate the virtual environment:
  ```bash
  deactivate
  ```
