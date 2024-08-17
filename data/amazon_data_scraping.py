### amazon_data_scraping.py
### Script to perform the webscraping loop for amazon.jobs data

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import sys
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Step 1: Scrape the URLs of the Amazon jobs

# Set up the Selenium WebDriver using WebDriver Manager
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Load the page
url = 'https://www.amazon.jobs/en-gb/teams/aws-software-development-engineer'
driver.get(url)

# Wait for the page to load
time.sleep(5) 

# Accept cookies
try:
    accept_cookies_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "btn-accept-cookies"))
    )
    accept_cookies_button.click()
except Exception as e:
    print("Accept cookies button not found or not clickable:", e)


# Extract the job URLs

url_list = []
base_url = "https://www.amazon.jobs"

page_number = 0

while True:
    page_number +=1
    sys.stdout.write(f"\rExtracting jobs from page {page_number}")
    sys.stdout.flush()

    # Find all job title elements
    job_title_elements = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "job-title")))

    # Iterate over each job title element to collect URLs
    for job_title_element in job_title_elements:
        try:
            # Get the job link
            job_link = job_title_element.find_element(By.TAG_NAME, "a").get_attribute("href")
        
            # Construct the full URL if the link is relative
            if job_link.startswith("/"):
                job_link = base_url + job_link
        
            # Add the URL to the list
            url_list.append(job_link)

        except Exception as e:
            print("Error processing job title element:", e)
   
    # Try to click the "Next page" button
    try:
        next_page_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.circle.right"))
        )
        next_page_button.click()
        
        # Wait for the next page to load
        time.sleep(5)

    except Exception as e:
        print("Next page button not found or not clickable:", e)
        break

# Close the WebDriver
driver.quit()

print("Number of URLs scraped:", len(url_list))


# Step 2: Scrape information for each URL in url_list


# Prepare an empty list for the URLS
columns = ["URL", "Job_Title", "Job_ID", "Basic_Qualifications", "Preferred_Qualifications", "Minimum_Pay", "Maximum_Pay"]
job_df = pd.DataFrame(columns=columns)

job_number = 0

for url in url_list:
    job_number += 1
    sys.stdout.write(f"\rExtracting job {job_number}")
    sys.stdout.flush()

    # Initialise all variables as NA
    job_id = pd.NA
    job_title = pd.NA
    basic_qualifications = pd.NA
    preferred_qualifications = pd.NA
    minimum_pay = pd.NA
    maximum_pay = pd.NA

    url = url
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract Job ID
    meta_info = soup.find('p', class_='meta')
    if meta_info:
        job_id_text = meta_info.get_text(strip=True)
        job_id = job_id_text.split()[2]
    else:
        print("Job ID not found")
        continue

    # Extract the Job Title
    job_title_tag = soup.find('h1', class_='title')
    if job_title_tag:
        job_title = job_title_tag.get_text(strip=True)
    else:
        print("Job title not found")
    
    # Extract the basic qualifications
    heading = soup.find('h2', text='BASIC QUALIFICATIONS')

    if heading:
        basic_qualifications_section = heading.find_parent('div', class_='section')
        basic_qualifications = basic_qualifications_section.get_text(separator="\n", strip=True)
    else:
        print("Job qualifications not found.")

    # Extract the preferred qualifications
    heading = soup.find('h2', text='PREFERRED QUALIFICATIONS')

    if heading:
        preferred_qualifications_section = heading.find_parent('div', class_='section')
        preferred_qualifications = preferred_qualifications_section.get_text(separator="\n", strip=True)
    else:
        print("Job qualifications not found.")

    # Extract the pay range
    section_text = preferred_qualifications_section.get_text(separator="\n", strip=True)
    pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\/year'

    matches = re.findall(pattern, section_text)

    if matches:
        minimum_pay = matches[0]
        maximum_pay = matches[1]
    else: 
        print("Payment information not found.")

    # Create a new row
    new_row = pd.DataFrame({
        "URL" : [url],
        "Job_Title": [job_title],
        "Job_ID": [job_id],
        "Basic_Qualifications": [basic_qualifications],
        "Preferred_Qualifications": [preferred_qualifications],
        "Minimum_Pay": [minimum_pay],
        "Maximum_Pay": [maximum_pay]
        })

    # Append the new row to the DataFrame
    job_df = pd.concat([job_df, new_row], ignore_index=True)

# Save the results to a CSV file
job_df.to_csv("raw_data/amazon_data_raw.csv")
print("Results saved to amazon_data_raw.csv")