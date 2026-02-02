"""
Generate Dummy Data for Application Fraud Detection
====================================================

This script generates synthetic application data matching the schema
from the CIMB Niaga Instinct SQL queries.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
NUM_RECORDS = 1000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 1, 31)

# Helper function to hash strings
def hash_string(text):
    """Simulate hashed data"""
    return hashlib.sha256(str(text).encode()).hexdigest()[:32]

# Generate base data
data = {
    'Application_Number': [f'APP{str(i).zfill(8)}' for i in range(1, NUM_RECORDS + 1)],
    'Application_Date': pd.date_range(start=START_DATE, end=END_DATE, periods=NUM_RECORDS),
    'Application_Type': np.random.choice(['RCC', 'RPL'], NUM_RECORDS, p=[0.6, 0.4]),
    'Amount_Limit': np.random.uniform(10000000, 500000000, NUM_RECORDS).round(0),
    'Branch': np.random.choice(['Jakarta Pusat', 'Jakarta Selatan', 'Jakarta Barat', 
                                'Jakarta Utara', 'Jakarta Timur', 'Bandung', 
                                'Surabaya', 'Medan', 'Semarang'], NUM_RECORDS),
}

# Add application-specific fields
data['TUJUAN PINJAMAN'] = np.random.choice(['Renovasi Rumah', 'Pendidikan', 'Modal Usaha', 
                                            'Konsumsi', 'Investasi', 'Refinancing'], NUM_RECORDS)
data['BANK PENCAIRAN'] = np.random.choice(['CIMB Niaga', 'BCA', 'Mandiri', 'BNI', 'BRI'], NUM_RECORDS)
data['LOCATION CODE'] = np.random.choice([f'LOC{i:03d}' for i in range(1, 51)], NUM_RECORDS)
data['APO'] = np.random.choice([f'APO{i:03d}' for i in range(1, 21)], NUM_RECORDS)
data['PRIMARY/SECONDARY'] = np.random.choice(['Primary', 'Secondary'], NUM_RECORDS, p=[0.8, 0.2])

# Generate customer IDs (some customers will have multiple applications)
unique_customers = 600
customer_ids = [f'CUS{str(i).zfill(6)}' for i in range(1, unique_customers + 1)]
# Some customers apply multiple times
customer_distribution = np.random.choice(customer_ids, NUM_RECORDS, 
                                        p=np.random.dirichlet(np.ones(unique_customers)))

data['ID/KTP/PASPOR/KITAS'] = [hash_string(cid) for cid in customer_distribution]
data['NAMA'] = [hash_string(f'NAME_{cid}') for cid in customer_distribution]

# Generate demographic data
birth_dates = [datetime(1960, 1, 1) + timedelta(days=np.random.randint(0, 365*45)) 
               for _ in range(NUM_RECORDS)]
data['TANGGAL LAHIR'] = birth_dates
data['USIA'] = [(datetime.now() - bd).days // 365 for bd in birth_dates]

data['KODE POS RUMAH'] = np.random.choice([f'{i:05d}' for i in range(10000, 99999)], NUM_RECORDS)
data['NO HP'] = [f'08{np.random.randint(10000000, 99999999)}' for _ in range(NUM_RECORDS)]

# Company information
data['NAMA PERUSAHAAN'] = np.random.choice(['PT ABC', 'PT XYZ', 'CV Maju', 'UD Jaya', 
                                            'PT Sejahtera', 'Freelance', 'Wiraswasta', 
                                            'PNS', 'BUMN'], NUM_RECORDS)
data['KODE POS PERUSAHAAN'] = np.random.choice([f'{i:05d}' for i in range(10000, 99999)], NUM_RECORDS)
data['TELP KANTOR'] = [f'021{np.random.randint(1000000, 9999999)}' for _ in range(NUM_RECORDS)]

# Education and employment
data['PENDIDIKAN TERAKHIR'] = np.random.choice(['SD', 'SMP', 'SMA', 'D3', 'S1', 'S2', 'S3'], 
                                               NUM_RECORDS, p=[0.05, 0.10, 0.25, 0.15, 0.35, 0.08, 0.02])
data['GAJI/TAHUN'] = np.random.uniform(50000000, 1000000000, NUM_RECORDS).round(0)
data['PEKERJAAN'] = np.random.choice(['Karyawan Swasta', 'PNS', 'Wiraswasta', 'Profesional', 
                                      'Manager', 'Direktur', 'Freelance'], NUM_RECORDS,
                                     p=[0.3, 0.15, 0.2, 0.15, 0.1, 0.05, 0.05])

# Sales agent information
num_agents = 100
agent_ids = [f'AGT{i:04d}' for i in range(1, num_agents + 1)]
data['SALES CODE'] = np.random.choice(agent_ids, NUM_RECORDS)
data['NIP/NIK'] = [hash_string(f'NIP_{sid}') for sid in data['SALES CODE']]
data['MARKETING PROGRAM'] = np.random.choice(['Regular', 'Premium', 'Priority', 'Corporate'], 
                                             NUM_RECORDS, p=[0.5, 0.25, 0.15, 0.1])
data['SALES BRANCH'] = data['Branch']  # Same as application branch

# Join dates for agents (random dates in the past 5 years)
join_dates = [datetime.now() - timedelta(days=np.random.randint(30, 1825)) 
              for _ in range(NUM_RECORDS)]
data['JOIN DATE'] = join_dates

# Collateral information (mostly for RPL)
data['NO SERTIFIKAT'] = [f'CERT{np.random.randint(100000, 999999)}' if app_type == 'RPL' else None 
                         for app_type in data['Application_Type']]
data['KODE POS SERTIFIKAT'] = [data['KODE POS RUMAH'][i] if data['Application_Type'][i] == 'RPL' 
                               else None for i in range(NUM_RECORDS)]
data['PURCHASE PRICE'] = [data['Amount_Limit'][i] * np.random.uniform(0.8, 1.2) 
                         if data['Application_Type'][i] == 'RPL' else None 
                         for i in range(NUM_RECORDS)]
data['APPRISAL PRICE'] = [data['PURCHASE PRICE'][i] * np.random.uniform(0.9, 1.1) 
                         if data['PURCHASE PRICE'][i] is not None else None 
                         for i in range(NUM_RECORDS)]
data['SERTIFIKAT ATAS NAMA'] = [hash_string(f'OWNER_{data["ID/KTP/PASPOR/KITAS"][i]}') 
                                if data['Application_Type'][i] == 'RPL' else None 
                                for i in range(NUM_RECORDS)]

# Rejection codes (simulate ~20% rejection rate)
rejection_codes = ['R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 'R08', 'R09', 'R10']
data['REJECTION CODE'] = [np.random.choice(rejection_codes) if np.random.random() < 0.2 else None 
                         for _ in range(NUM_RECORDS)]

# Create DataFrame
df = pd.DataFrame(data)

# Sort by application date
df = df.sort_values('Application_Date').reset_index(drop=True)

# Save to CSV
output_file = 'application_data.csv'
df.to_csv(output_file, index=False)

print(f"Generated {NUM_RECORDS} records")
print(f"Date range: {df['Application_Date'].min()} to {df['Application_Date'].max()}")
print(f"Application types: RCC={sum(df['Application_Type']=='RCC')}, RPL={sum(df['Application_Type']=='RPL')}")
print(f"Rejection rate: {df['REJECTION CODE'].notna().sum() / len(df) * 100:.1f}%")
print(f"Unique customers: {df['ID/KTP/PASPOR/KITAS'].nunique()}")
print(f"Data saved to: {output_file}")
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
