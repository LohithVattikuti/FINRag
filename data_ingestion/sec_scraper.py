import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ✅ Your email added to identify you to SEC.gov servers
USER_EMAIL = "vattikutilohith@gmail.com"

# ✅ Create folder for storing downloaded filings
os.makedirs("sec_filings", exist_ok=True)

def fetch_sec_filings(ticker="AAPL", form_type="10-K", count=3):
    """
    Downloads the latest 10-K or 10-Q filings for a given company ticker from SEC EDGAR.
    Saves them as HTML files in the 'sec_filings' folder.
    """
    base_url = "https://www.sec.gov"

    # ✅ Header required by SEC to avoid getting rate-limited or blocked
    headers = {
        "User-Agent": f"Mozilla/5.0 ({USER_EMAIL})"
    }

    # 🔍 Step 1: Find the CIK (unique company identifier)
    cik_lookup = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&action=getcompany&output=atom"
    cik_res = requests.get(cik_lookup, headers=headers)
    if cik_res.status_code != 200:
        print("❌ Invalid ticker or SEC request blocked.")
        return

    # 🧠 Extract CIK from XML response
    cik_page = BeautifulSoup(cik_res.content, "lxml")
    cik = cik_page.find("company-info").find("cik").text.strip().zfill(10)

    # 🔍 Step 2: Get recent filings list in JSON format
    search_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(search_url, headers=headers)
    if res.status_code != 200:
        print("❌ Could not retrieve filings.")
        return

    data = res.json()
    recent_filings = data["filings"]["recent"]

    # 🎯 Step 3: Filter for specified form type (e.g., 10-K)
    matched = []
    for i, ftype in enumerate(recent_filings["form"]):
        if ftype == form_type:
            matched.append({
                "accession": recent_filings["accessionNumber"][i].replace("-", ""),
                "filingDate": recent_filings["filingDate"][i],
                "document": recent_filings["primaryDocument"][i]
            })
        if len(matched) >= count:
            break

    # 📥 Step 4: Download each matched document (HTML)
    for filing in tqdm(matched, desc=f"Downloading {form_type} filings"):
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{filing['accession']}/{filing['document']}"
        try:
            filing_res = requests.get(filing_url, headers=headers)
            if filing_res.status_code == 200:
                filename = f"{ticker}_{form_type}_{filing['filingDate']}.html"
                with open(os.path.join("sec_filings", filename), "w", encoding="utf-8") as f:
                    f.write(filing_res.text)
            else:
                print(f"⚠️ Failed to download: {filing_url}")
        except Exception as e:
            print(f"❌ Error fetching filing: {e}")

if __name__ == "__main__":
    # ✅ You can change the ticker, form type, or number of filings here
    fetch_sec_filings("AAPL", "10-K", count=3)
