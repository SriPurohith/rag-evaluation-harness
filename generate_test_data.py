import os
from fpdf import FPDF
import random

# Create directory if not exists
os.makedirs("data/policies", exist_ok=True)

states = ["Tennessee", "Washington", "California", "Texas", "New York"]
years = [2022, 2023, 2024]
policy_types = ["Remote Work", "Health Benefits", "Travel Reimbursement", "PTO Policy"]

def generate_policy_text(state, year, p_type):
    return f"""
    {state} Official {p_type} - {year}
    
    Section 1: Eligibility
    This policy applies to all full-time employees residing in {state} as of January {year}.
    
    Section 2: Guidelines
    Under the {year} regulations, {p_type} must be requested 30 days in advance.
    For {state} specific mandates, employees must maintain a 50Mbps internet connection
    for all {p_type} activities.
    
    Section 3: Compliance
    Failure to comply with the {year} {state} guidelines may result in a review 
    of employment status.
    """

print("🚀 Generating 100 policies...")

for i in range(100):
    state = random.choice(states)
    year = random.choice(years)
    p_type = random.choice(policy_types)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    text = generate_policy_text(state, year, p_type)
    for line in text.split('\n'):
        pdf.cell(200, 10, txt=line.strip(), ln=True)
    
    # Filename encodes metadata: e.g., Policy_Tennessee_2024_0.pdf
    filename = f"data/policies/Policy_{state}_{year}_{i}.pdf"
    pdf.output(filename)

print(f"✅ Successfully created 100 PDFs in 'data/policies/'")