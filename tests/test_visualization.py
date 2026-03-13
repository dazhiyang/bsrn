import os
import sys
import matplotlib.pyplot as plt

# Add src to python path to ensure bsrn is importable
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from bsrn.visualization import plot_bsrn_availability
except ImportError:
    print("Error: Could not import bsrn. Make sure you are running from the project root.")
    sys.exit(1)

def main():
    # --- CONFIGURATION ---
    # Put your BSRN FTP credentials here
    BSRN_USER = "bsrnftp" 
    BSRN_PASSWORD = "bsrn1"
    
    # List of stations you want to check (can be a string or a list)
    # Examples: 'PAY' or ['PAY', 'NYA', 'GVN', 'ILO', 'TAT', 'QIQ']
    STATIONS_TO_CHECK = ['PAY', 'NYA', 'GVN', 'ILO', 'TAT', 'QIQ'] 
    
    # Year range
    START_YEAR = 1992
    
    # --- EXECUTION ---
    if BSRN_USER == "your_username":
        print("Please edit this script and add your BSRN FTP credentials!")
        return

    print(f"Initiating BSRN FTP search for: {STATIONS_TO_CHECK}")
    
    try:
        # Save the plot
        output_file = "bsrn_availability_heatmap.pdf"
        
        fig = plot_bsrn_availability(
            stations=STATIONS_TO_CHECK,
            username=BSRN_USER,
            password=BSRN_PASSWORD,
            start_year=START_YEAR,
            output_file=output_file
        )
        print(f"Successfully generated heatmap: {output_file}")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
